#include <CadPL/ShaderLibrary.h>
#include <CadPL/ShaderGenerator.h>
#include <CadPL/DebugUtils.h>
#include <CadR/VulkanDevice.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <future>
#include <regex>

#include "glm/detail/setup.hpp"

using namespace std;
using namespace CadPL;


bool SharedShaderModule::isValid() const {
	if (!_smObject) {
		return false;
	}
	auto* smObject = static_cast<ShaderLibrary::AbstractShaderModuleObject*>(_smObject);
	return smObject->shaderModule != VK_NULL_HANDLE || smObject->identifier.identifierSize != 0;
}

bool SharedShaderModule::aquireCompileFlag()
{
	if (!_smObject) {
		return false;
	}
	auto* smObject = static_cast<ShaderLibrary::AbstractShaderModuleObject*>(_smObject);
	if (smObject->shaderModule == VK_NULL_HANDLE && smObject->identifier.identifierSize == 0 && !smObject->compiling.test_and_set()) {
		return true;
	}
	return false;
}

void SharedShaderModule::waitIfCompiling() {
	if (!_smObject) {
		return;
	}
	auto* smObject = static_cast<ShaderLibrary::AbstractShaderModuleObject*>(_smObject);
	const auto start = std::chrono::system_clock::now();
	smObject->compiling.wait(true);
	const auto end = std::chrono::system_clock::now();
	CadPL::Debug::log("", "waitIfCompiling()", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void ShaderLibrary::destroy() noexcept
{
	if(_device) {
		_device->destroy(_pipelineLayout);
		_device->destroy(_descriptorSetLayout);
		_pipelineLayout = nullptr;
		_descriptorSetLayout = nullptr;
	}
}


ShaderLibrary::~ShaderLibrary() noexcept
{
	assert(!containsShaderModuleObjects(_vertexShaderMap) && "ShaderLibrary::~ShaderLibrary(): All SharedShaderModules must be released before destroying ShaderLibrary.");
	assert(!containsShaderModuleObjects(_geometryShaderMap) && "ShaderLibrary::~ShaderLibrary(): All SharedShaderModules must be released before destroying ShaderLibrary.");
	assert(!containsShaderModuleObjects(_fragmentShaderMap) && "ShaderLibrary::~ShaderLibrary(): All SharedShaderModules must be released before destroying ShaderLibrary.");

	if(_device) {
		_device->destroy(_pipelineLayout);
		_device->destroy(_descriptorSetLayout);
	}
}


ShaderLibrary::ShaderLibrary(CadR::VulkanDevice& device, uint32_t maxTextures)
	: ShaderLibrary()  // make sure thay destructor will be called when exception is thrown
{
	init(device, maxTextures);
}


void ShaderLibrary::init(CadR::VulkanDevice& device, uint32_t maxTextures)
{
	destroy();

	_device = &device;

	_descriptorSetLayout =
		_device->createDescriptorSetLayout(
			vk::DescriptorSetLayoutCreateInfo(
				vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,  // flags
				1,  // bindingCount
				array<vk::DescriptorSetLayoutBinding,1>{  // pBindings
					vk::DescriptorSetLayoutBinding{
						0,  // binding
						vk::DescriptorType::eCombinedImageSampler,  // descriptorType
						maxTextures, // descriptorCount
						vk::ShaderStageFlagBits::eFragment,  // stageFlags
						nullptr  // pImmutableSamplers
					}
				}.data()
			).setPNext(
				&(const vk::DescriptorSetLayoutBindingFlagsCreateInfo&)vk::DescriptorSetLayoutBindingFlagsCreateInfo(
					1,  // bindingCount
					array<vk::DescriptorBindingFlags,1>{  // pBindingFlags
						vk::DescriptorBindingFlagBits::eUpdateAfterBind |
							vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending |
							vk::DescriptorBindingFlagBits::ePartiallyBound |
							vk::DescriptorBindingFlagBits::eVariableDescriptorCount
					}.data()
				)
			)
		);
	_pipelineLayout =
		_device->createPipelineLayout(
			vk::PipelineLayoutCreateInfo(
				vk::PipelineLayoutCreateFlags(),  // flags
				1,  // setLayoutCount
				&_descriptorSetLayout,  // pSetLayouts
				1,  // pushConstantRangeCount
				array{
					vk::PushConstantRange{  // pPushConstantRanges
						vk::ShaderStageFlagBits::eAllGraphics,  // stageFlags
						0,  // offset
						60  // size
					},
				}.data()
			)
		);
	_descriptorSetLayoutList.reserve(1);
	_descriptorSetLayoutList.push_back(_descriptorSetLayout);
}


void ShaderLibrary::destroyShaderModule(void* shaderModuleObject) noexcept
{
	AbstractShaderModuleObject* smObject = static_cast<AbstractShaderModuleObject*>(shaderModuleObject); 

	smObject->shaderLibrary->_device->destroy(smObject->shaderModule);

	switch(static_cast<uint32_t>(smObject->owningMap)) {
	case static_cast<uint32_t>(vk::ShaderStageFlagBits::eVertex):   smObject->shaderLibrary->_vertexShaderMap.erase(static_cast<ShaderModuleObject<VertexShaderMapKey>*>(smObject)->eraseIt); break;
	case static_cast<uint32_t>(vk::ShaderStageFlagBits::eGeometry): smObject->shaderLibrary->_geometryShaderMap.erase(static_cast<ShaderModuleObject<GeometryShaderMapKey>*>(smObject)->eraseIt); break;
	case static_cast<uint32_t>(vk::ShaderStageFlagBits::eFragment): smObject->shaderLibrary->_fragmentShaderMap.erase(static_cast<ShaderModuleObject<FragmentShaderMapKey>*>(smObject)->eraseIt); break;
	default:
		assert(0 && "ShaderModuleObject::owningMap contains unknown value.");
	}
}

vk::ShaderModuleIdentifierEXT ShaderLibrary::getShaderModuleIdentifier(vk::ShaderModule shaderModule)
{
	assert(vkGetShaderModuleIdentifierEXT && "Missing PFN_vkGetShaderModuleIdentifierEXT");
	const auto &directory = ShaderGenerator::getCacheDirectory();
	vk::ShaderModuleIdentifierEXT identifier;
	if (!directory.empty()) {
		vkGetShaderModuleIdentifierEXT(_device->handle(), shaderModule, reinterpret_cast<VkShaderModuleIdentifierEXT*>(&identifier));
		if (identifier.identifierSize != 0) {
			identifierDatabaseDirty = true;
		}
	}
	return identifier;
}

template <typename MapKey>
static void serialize(std::stringstream &output, const MapKey &key, const vk::ShaderModuleIdentifierEXT &identifier) {
	if (identifier.identifierSize == 0) {
		return;
	}
	output << key.serialize();
	output << ':';
	output << static_cast<char>(identifier.identifierSize);
	output.write(reinterpret_cast<const char*>(identifier.identifier.data()), identifier.identifierSize);
}

template <typename MapKey>
void ShaderLibrary::serializeMap(const std::string_view prefix, const std::map<MapKey, ShaderModuleObject<MapKey>> &map, std::stringstream &output)
{
	auto it = map.begin();
	for (; it != map.end(); ++it) {
		if (it->second.identifier.identifierSize != 0) {
			break;
		}
	}
	if (it == map.end()) {
		return;
	}
#ifndef NDEBUG
	std::map<std::string, size_t> validation;
#endif
	output << prefix;
	for (; it != map.end(); ++it) {
#ifndef NDEBUG
		if (it->second.identifier.identifierSize > 0) {
			validation[it->first.serialize()]++;
		}
#endif
		serialize(output, it->first, it->second.identifier);
	}
#ifndef NDEBUG
	for (auto &it : validation) {
		if (it.second > 1) {
			std::cerr << "Duplicate shader identifier under same key, this should not happen.\n";
		}
	}
#endif
}

void ShaderLibrary::saveIdentifierDatabase() {
	const auto &directory = ShaderGenerator::getCacheDirectory();
	if (directory.empty() || !identifierDatabaseDirty) {
		return;
	}
	identifierDatabaseDirty = false;

	std::stringstream output;
	serializeMap("@0", _vertexShaderMap, output);
	serializeMap("@1", _fragmentShaderMap, output);
	serializeMap("@2", _geometryShaderMap, output);

	auto str =  output.str();
	// std::cout << "Saving id database: " << str.size() << " B\n";
	if (!str.empty()) {
		auto fileName = directory + "\\ids.bin";
		std::ofstream hashfile(fileName, std::ios::binary);
		if (hashfile.is_open()) {
			hashfile << str;
			hashfile.close();
		}
	}
}

void ShaderLibrary::loadIdentifierDatabase() {
	const auto &directory = ShaderGenerator::getCacheDirectory();
	if (directory.empty()) {
		return;
	}

	auto fileName = directory + "\\ids.bin";

	std::ifstream file(fileName.data(), std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		return;
	}
	file.unsetf(std::ios::skipws);

	enum State {
		ENTRY_BEGIN,
		GROUP_TYPE,
		KEY,
		ID_SIZE,
		ID
	};
	State state = ENTRY_BEGIN;

	char c;
	std::string key;
	size_t idIndex = 0;
	vk::ShaderModuleIdentifierEXT identifier = {};
	std::function<void(const std::string&, const vk::ShaderModuleIdentifierEXT&)> func = nullptr;

	while (file.get(c)) {
		switch (state) {
			case ENTRY_BEGIN:
				if (c == '@') {
					state = GROUP_TYPE;
				}
				else {
					key.clear();
					if (c != ':') {
						key += c;
						state = KEY;
					}
					else {
						state = ID_SIZE;
					}
				}
				break;
			case GROUP_TYPE:
				switch (c) {
					case '0':
						func = std::bind(&ShaderLibrary::setIdentifierVertex, this, std::placeholders::_1, std::placeholders::_2);
						break;
					case '1':
						func = std::bind(&ShaderLibrary::setIdentifierFragment, this, std::placeholders::_1, std::placeholders::_2);
						break;
					case '2':
						func = std::bind(&ShaderLibrary::setIdentifierGeometry, this, std::placeholders::_1, std::placeholders::_2);
						break;
					default:
						std::cerr << "Invalid data in identifier database\n";
						return;
				}
				key.clear();
				state = KEY;
				break;
			case KEY:
				if (c == ':') {
					state = ID_SIZE;
				}
				else {
					key += c;
				}
				break;
			case ID_SIZE:
				if (c > VK_MAX_SHADER_MODULE_IDENTIFIER_SIZE_EXT) {
					std::cerr << "Invalid data in identifier database\n";
					return;
				}
				if (c == 0) {
					state = ENTRY_BEGIN;
				}
				else {
					identifier.identifierSize = c;
					idIndex = 0;
					state = ID;
				}
				break;
			case ID:
				identifier.identifier[idIndex] = c;
				idIndex++;
				if (idIndex == identifier.identifierSize) {
					if (func) {
						func(key, identifier);
					}
					state = ENTRY_BEGIN;
				}
				break;
			default:
				break;
		}
	}

}

void ShaderLibrary::setIdentifierVertex(const std::string& serializedState, const vk::ShaderModuleIdentifierEXT &identifier)
{
	getOrEmplace(VertexShaderMapKey(serializedState), _vertexShaderMap)->identifier = identifier;
}

void ShaderLibrary::setIdentifierFragment(const std::string& serializedState, const vk::ShaderModuleIdentifierEXT &identifier)
{
	getOrEmplace(FragmentShaderMapKey(serializedState), _fragmentShaderMap)->identifier = identifier;;
}

void ShaderLibrary::setIdentifierGeometry(const std::string& serializedState, const vk::ShaderModuleIdentifierEXT &identifier)
{
	getOrEmplace(GeometryShaderMapKey(serializedState), _geometryShaderMap)->identifier = identifier;;
}

template <typename MapKey>
bool ShaderLibrary::containsShaderModuleObjects(const std::map<MapKey, ShaderModuleObject<MapKey>> &map) const
{
	for (const auto &it : map) {
		if (it.second.shaderModule) {
			return true;
		}
	}
	return false;
}

template <typename MapKey>
SharedShaderModule ShaderLibrary::getOrCreateShader(const ShaderState& state, std::map<MapKey, ShaderModuleObject<MapKey>> &map, vk::ShaderModule (*create)(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName))
{
	MapKey key(state);
	auto [it, newRecord] = map.try_emplace(key);
	if(newRecord) {
		if (it->second.identifier.identifierSize == 0) {
			std::string cacheName;
#ifdef NDEBUG
			if (ShaderGenerator::useSpirVCache())
#endif
				cacheName = key.serialize();
			try {
				it->second.shaderModule = create(state, *_device, cacheName);
			} catch(...) {
				map.erase(it);
				throw;
			}
			if (_useShaderModuleIdentifier && it->second.shaderModule) {
				it->second.identifier = getShaderModuleIdentifier(it->second.shaderModule);
			}
		}
		it->second.referenceCounter = 0;
		it->second.shaderLibrary = this;
		it->second.owningMap = MapKey::ShaderStage;
		it->second.eraseIt = it;
		it->second.compiling.clear();
	}
	return SharedShaderModule(&it->second);
}

SharedShaderModule ShaderLibrary::getOrCreateVertexShader(const ShaderState& state)
{
	return getOrCreateShader<VertexShaderMapKey>(state, _vertexShaderMap, &ShaderGenerator::createVertexShader);
}

SharedShaderModule ShaderLibrary::getOrCreateGeometryShader(const ShaderState& state)
{
	return getOrCreateShader<GeometryShaderMapKey>(state, _geometryShaderMap, &ShaderGenerator::createGeometryShader);
}

SharedShaderModule ShaderLibrary::getOrCreateFragmentShader(const ShaderState& state)
{
	return getOrCreateShader<FragmentShaderMapKey>(state, _fragmentShaderMap, &ShaderGenerator::createFragmentShader);
}

template <typename MapKey>
ShaderLibrary::ShaderModuleObject<MapKey>* ShaderLibrary::getOrEmplace(const MapKey& key, std::map<MapKey, ShaderModuleObject<MapKey>> &map)
{
	auto [it, newRecord] = map.try_emplace(key);
	if(newRecord) {
		it->second.shaderModule = VK_NULL_HANDLE;
		it->second.referenceCounter = 0;
		it->second.shaderLibrary = this;
		it->second.owningMap = MapKey::ShaderStage;
		it->second.eraseIt = it;
		it->second.compiling.clear();
	}
	return &it->second;
}

SharedShaderModule ShaderLibrary::getOrEmplaceVertexShader(const ShaderState& state)
{
	return {getOrEmplace(VertexShaderMapKey(state), _vertexShaderMap)};
}

SharedShaderModule ShaderLibrary::getOrEmplaceGeometryShader(const ShaderState& state)
{
	return {getOrEmplace(GeometryShaderMapKey(state), _geometryShaderMap)};
}

SharedShaderModule ShaderLibrary::getOrEmplaceFragmentShader(const ShaderState& state)
{
	return {getOrEmplace(FragmentShaderMapKey(state), _fragmentShaderMap)};
}

template <typename MapKey>
void ShaderLibrary::createAsync(const ShaderState &state, vk::ShaderModule (*create)(const ShaderState& state, CadR::VulkanDevice& device, const std::string&), SharedShaderModule* shaderModule) {
	MapKey key(state);
	const auto s = state.serialize();
	Debug::enterThread(s);

	auto *object = reinterpret_cast<ShaderLibrary::AbstractShaderModuleObject*>(shaderModule->_smObject);
	try {
		std::string cacheName;
#ifdef NDEBUG
		if (ShaderGenerator::useSpirVCache())
#endif
			cacheName = key.serialize();
		auto shader = create(state, *_device, cacheName);

		if (object->shaderModule) {
			// should not happen with proper synchronization
			std::cerr << "Warning: duplicate shader creation\n";
			_device->destroyShaderModule(shader);
			return;
		}
		object->shaderModule = shader;
		if (_useShaderModuleIdentifier) {
			object->identifier = getShaderModuleIdentifier(object->shaderModule);
		}
	} catch(exception &e) {
		std::cerr << "ShaderLibrary::createAsync failed because of exception: " << e.what() << endl;
		object->compiling.clear();
		object->compiling.notify_all();
		Debug::exitThread(s);
		throw;
	} catch(...) {
		object->compiling.clear();
		object->compiling.notify_all();
		Debug::exitThread(s);
		std::cerr << "ShaderLibrary::createAsync failed because of unspecified exception." << endl;
		throw;
	}
	object->compiling.clear();
	object->compiling.notify_all();
	Debug::exitThread(s);
}

ShaderLibrary::ShaderSet ShaderLibrary::getShadersWithoutCompilation(const ShaderState& state)
{
	const bool geometry = ShaderGenerator::usesGeometryShader(state);
	ShaderSet set;
	std::unique_lock lock{this->mutex};
	set.vertex = getOrEmplaceVertexShader(state);
	set.fragment = getOrEmplaceFragmentShader(state);
	if (geometry) {
		set.geometry = getOrEmplaceGeometryShader(state);
	}
	return set;
}

std::future<void> ShaderLibrary::createVertexShaderAsync(const ShaderState& state, SharedShaderModule& shader)
{
	return std::async(std::launch::async, &ShaderLibrary::createAsync<VertexShaderMapKey>, this, state, &ShaderGenerator::createVertexShader, &shader);
}
std::future<void> ShaderLibrary::createGeometryShaderAsync(const ShaderState& state, SharedShaderModule& shader)
{
	return std::async(std::launch::async, &ShaderLibrary::createAsync<GeometryShaderMapKey>, this, state, &ShaderGenerator::createGeometryShader, &shader);
}
std::future<void> ShaderLibrary::createFragmentShaderAsync(const ShaderState& state, SharedShaderModule& shader)
{
	return std::async(std::launch::async, &ShaderLibrary::createAsync<FragmentShaderMapKey>, this, state, &ShaderGenerator::createFragmentShader, &shader);
}

void ShaderLibrary::createShaders(const ShaderState& state, SharedShaderModule &vertex, SharedShaderModule &geometry, SharedShaderModule &fragment)
{
	std::array<std::future<void>, 3> futures;
	if (vertex.aquireCompileFlag()) {
		futures[0] = createVertexShaderAsync(state, vertex);
	}
	if (fragment.aquireCompileFlag()) {
		futures[1] = createFragmentShaderAsync(state, fragment);
	}
	if (geometry.aquireCompileFlag()) {
		futures[2] = createGeometryShaderAsync(state, geometry);
	}
	for (auto &f : futures) {
		if (f.valid()) {
			f.wait();
		}
	}
}

ShaderLibrary::ShaderSet ShaderLibrary::getOrCreateShaders(const ShaderState& state, bool compile)
{
	ShaderSet set = getShadersWithoutCompilation(state);
	createShaders(state, set.vertex, set.geometry, set.fragment);
	return set;
}
