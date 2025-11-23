#pragma once

#include <future>
#include <map>
#include <vulkan/vulkan.hpp>

#include "ShaderState.h"

#include "CadR/VulkanDevice.h"

namespace CadR {
class VulkanDevice;
}

namespace CadPL {

class CADPL_EXPORT SharedShaderModule {
protected:
	void* _smObject = nullptr;
public:

	SharedShaderModule() = default;
	~SharedShaderModule() noexcept;

	SharedShaderModule(SharedShaderModule&& other) noexcept;
	SharedShaderModule(const SharedShaderModule& other) noexcept;
	SharedShaderModule& operator=(SharedShaderModule&& rhs) noexcept;
	SharedShaderModule& operator=(const SharedShaderModule& rhs) noexcept;

	vk::ShaderModule get() const;
	operator vk::ShaderModule() const;
	explicit operator bool() const;
	void reset() noexcept;

	bool isValid() const;
	bool aquireCompileFlag();
	void waitIfCompiling();

	vk::ShaderModuleIdentifierEXT* getIdentifier() const;

protected:
	friend class ShaderLibrary;
	SharedShaderModule(void* shaderModuleObject) noexcept;
};


class CADPL_EXPORT ShaderLibrary {
protected:

	CadR::VulkanDevice* _device = nullptr;
	PFN_vkGetShaderModuleIdentifierEXT vkGetShaderModuleIdentifierEXT = {};
	bool identifierDatabaseDirty = false;

	struct AbstractShaderModuleObject {
		size_t referenceCounter;  //< Reference counter. It must be on the beginning of this structure because of implementation of some functions in this class.
		vk::ShaderModule shaderModule;  //< Shader module handle. It must be on the second place in this structure because of implementation of some functions in this class.
		ShaderLibrary* shaderLibrary;  //< ShaderLibrary owning this ShaderModuleObject. It must be on the third place in this structure because of implementation of some functions in this class.
		vk::ShaderStageFlags owningMap;  //< It indicates the map that this structure is member of. It must be on the fourth place in this structure because of implementation of some functions in this class.
		vk::ShaderModuleIdentifierEXT identifier; //< Vulkan identifier
		std::atomic_flag compiling;
	};
	template<typename MapKey>
	struct ShaderModuleObject : AbstractShaderModuleObject {
		typename std::map<MapKey,ShaderModuleObject<MapKey>>::iterator eraseIt;  //< Iterator for removing this object from the map when the referenceCounter reaches zero.
	};

	template <typename MapKey>
	static void serializeMap(std::string_view prefix, const std::map<MapKey, ShaderModuleObject<MapKey>> &map, std::stringstream &output);

	using VertexShaderMapKey = VertexShaderState;
	using GeometryShaderMapKey = GeometryShaderState;
	using FragmentShaderMapKey = FragmentShaderState;

	std::map<VertexShaderMapKey, ShaderModuleObject<VertexShaderMapKey>> _vertexShaderMap;
	std::map<GeometryShaderMapKey, ShaderModuleObject<GeometryShaderMapKey>> _geometryShaderMap;
	std::map<FragmentShaderMapKey, ShaderModuleObject<FragmentShaderMapKey>> _fragmentShaderMap;

	vk::PipelineLayout _pipelineLayout;
	vk::DescriptorSetLayout _descriptorSetLayout;
	std::vector<vk::DescriptorSetLayout> _descriptorSetLayoutList;

	std::mutex mutex;
	bool _useShaderModuleIdentifier = false;

	static void refShaderModule(void* shaderModuleObject) noexcept;
	static void unrefShaderModule(void* shaderModuleObject) noexcept;
	static void destroyShaderModule(void* shaderModuleObject) noexcept;
	friend class SharedShaderModule;

public:

	// construction and destruction
	ShaderLibrary() noexcept = default;
	ShaderLibrary(CadR::VulkanDevice& device, uint32_t maxTextures = 250000);
	~ShaderLibrary() noexcept;
	void init(CadR::VulkanDevice& device, uint32_t maxTextures = 250000);
	void destroy() noexcept;

	// synchronous API to get and create shaders
	SharedShaderModule getOrCreateVertexShader(const ShaderState& state);
	SharedShaderModule getOrCreateGeometryShader(const ShaderState& state);
	SharedShaderModule getOrCreateFragmentShader(const ShaderState& state);
	SharedShaderModule getOrEmplaceVertexShader(const ShaderState& state);
	SharedShaderModule getOrEmplaceGeometryShader(const ShaderState& state);
	SharedShaderModule getOrEmplaceFragmentShader(const ShaderState& state);
	SharedShaderModule getVertexShader(const ShaderState& state);
	SharedShaderModule getGeometryShader(const ShaderState& state);
	SharedShaderModule getFragmentShader(const ShaderState& state);

	// asynchronous API to get and create shaders
	std::future<void> createVertexShaderAsync(const ShaderState& state, SharedShaderModule& shader);
	std::future<void> createGeometryShaderAsync(const ShaderState& state, SharedShaderModule& shader);
	std::future<void> createFragmentShaderAsync(const ShaderState& state, SharedShaderModule& shader);

	struct ShaderSet {
		SharedShaderModule vertex;
		SharedShaderModule geometry;
		SharedShaderModule fragment;
	};

	void createShaders(const ShaderState& state, SharedShaderModule &vertex, SharedShaderModule &geometry, SharedShaderModule &fragment);
	ShaderSet getOrCreateShaders(const ShaderState& state, bool compile = false);
	ShaderSet getShadersWithoutCompilation(const ShaderState& state);

	// getters
	CadR::VulkanDevice& device() const;
	vk::PipelineLayout pipelineLayout() const;
	vk::DescriptorSetLayout descriptorSetLayout() const;
	const std::vector<vk::DescriptorSetLayout>& descriptorSetLayoutList() const;

	size_t count() const noexcept;
	size_t identifierCount() const noexcept;
	size_t countVertex() const noexcept;
	size_t countGeometry() const noexcept;
	size_t countFragment() const noexcept;

	// VK_EXT_shader_module_identifier
	void setShaderModuleIdentifierEnabled(bool enabled, const CadR::VulkanDevice &device);
	bool useShaderModuleIdentifier() const noexcept;
	vk::ShaderModuleIdentifierEXT getShaderModuleIdentifier(vk::ShaderModule shaderModule);
	void saveIdentifierDatabase();
	void loadIdentifierDatabase();

private:
	void setIdentifierVertex(const std::string& serializedState, const vk::ShaderModuleIdentifierEXT &identifier);
	void setIdentifierFragment(const std::string& serializedState, const vk::ShaderModuleIdentifierEXT &identifier);
	void setIdentifierGeometry(const std::string& serializedState, const vk::ShaderModuleIdentifierEXT &identifier);

	template <typename MapKey>
	bool containsShaderModuleObjects(const std::map<MapKey, ShaderModuleObject<MapKey>> &map) const;

	template <typename MapKey>
	SharedShaderModule getOrCreateShader(const ShaderState& state, std::map<MapKey, ShaderModuleObject<MapKey>> &map, vk::ShaderModule (*create)(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName));

	template <typename MapKey>
	ShaderModuleObject<MapKey>* getOrEmplace(const MapKey& state, std::map<MapKey, ShaderModuleObject<MapKey>> &map);

	template <typename MapKey>
	void createAsync(const ShaderState &state, vk::ShaderModule (*create)(const ShaderState& state, CadR::VulkanDevice& device, const std::string&), SharedShaderModule* shaderModule);
};


// inline functions
inline SharedShaderModule::SharedShaderModule(void* shaderModuleObject) noexcept  : _smObject(shaderModuleObject) { ShaderLibrary::refShaderModule(shaderModuleObject); }
inline SharedShaderModule::~SharedShaderModule() noexcept  { if(_smObject) ShaderLibrary::unrefShaderModule(_smObject); }
inline SharedShaderModule::SharedShaderModule(SharedShaderModule&& other) noexcept  : _smObject(other._smObject) { other._smObject=nullptr; }
inline SharedShaderModule::SharedShaderModule(const SharedShaderModule& other) noexcept  : _smObject(other._smObject) { if(_smObject) ShaderLibrary::refShaderModule(_smObject); }
inline SharedShaderModule& SharedShaderModule::operator=(SharedShaderModule&& rhs) noexcept  { if(_smObject) ShaderLibrary::unrefShaderModule(_smObject); _smObject=rhs._smObject; rhs._smObject=nullptr; return *this; }
inline SharedShaderModule& SharedShaderModule::operator=(const SharedShaderModule& rhs) noexcept  { if(_smObject) ShaderLibrary::unrefShaderModule(_smObject); _smObject=rhs._smObject; if(_smObject) ShaderLibrary::refShaderModule(_smObject); return *this; }
inline vk::ShaderModule SharedShaderModule::get() const  { return static_cast<ShaderLibrary::AbstractShaderModuleObject*>(_smObject)->shaderModule; }
inline SharedShaderModule::operator vk::ShaderModule() const  { return static_cast<ShaderLibrary::AbstractShaderModuleObject*>(_smObject)->shaderModule; }
inline SharedShaderModule::operator bool() const  { return _smObject; }
inline void SharedShaderModule::reset() noexcept  { if(!_smObject) return; ShaderLibrary::unrefShaderModule(_smObject); _smObject=nullptr; }
inline vk::ShaderModuleIdentifierEXT* SharedShaderModule::getIdentifier() const { return _smObject? &static_cast<ShaderLibrary::AbstractShaderModuleObject*>(_smObject)->identifier : nullptr;}

inline void ShaderLibrary::refShaderModule(void* shaderModuleObject) noexcept  { auto* smObject=static_cast<ShaderLibrary::AbstractShaderModuleObject*>(shaderModuleObject); smObject->referenceCounter++; }
inline void ShaderLibrary::unrefShaderModule(void* shaderModuleObject) noexcept  { auto* smObject=static_cast<ShaderLibrary::AbstractShaderModuleObject*>(shaderModuleObject); if(smObject->referenceCounter==1) ShaderLibrary::destroyShaderModule(smObject); else smObject->referenceCounter--; }
inline SharedShaderModule ShaderLibrary::getVertexShader(const ShaderState& state)  { auto it=_vertexShaderMap.find(state); return (it!=_vertexShaderMap.end()) ? SharedShaderModule(&it->second) : SharedShaderModule(); }
inline SharedShaderModule ShaderLibrary::getGeometryShader(const ShaderState& state)  { auto it=_geometryShaderMap.find(state); return (it!=_geometryShaderMap.end()) ? SharedShaderModule(&it->second) : SharedShaderModule(); }
inline SharedShaderModule ShaderLibrary::getFragmentShader(const ShaderState& state)  { auto it=_fragmentShaderMap.find(state); return (it!=_fragmentShaderMap.end()) ? SharedShaderModule(&it->second) : SharedShaderModule(); }
inline CadR::VulkanDevice& ShaderLibrary::device() const  { return *_device; }
inline vk::PipelineLayout ShaderLibrary::pipelineLayout() const  { return _pipelineLayout; }
inline vk::DescriptorSetLayout ShaderLibrary::descriptorSetLayout() const  { return _descriptorSetLayout; }
inline const std::vector<vk::DescriptorSetLayout>& ShaderLibrary::descriptorSetLayoutList() const  { return _descriptorSetLayoutList; }

inline size_t ShaderLibrary::count() const noexcept { return countVertex() + countGeometry() + countFragment(); }
inline size_t ShaderLibrary::identifierCount() const noexcept { size_t count = 0; for (const auto &o : _vertexShaderMap) if (o.second.identifier.identifierSize > 0) ++count; for (const auto &o : _geometryShaderMap) if (o.second.identifier.identifierSize > 0) ++count; for (const auto &o : _fragmentShaderMap) if (o.second.identifier.identifierSize > 0) ++count; return count; }
inline size_t ShaderLibrary::countVertex() const noexcept { size_t count = 0; for (const auto &o : _vertexShaderMap) if (o.second.referenceCounter > 0) count++; return count; }
inline size_t ShaderLibrary::countGeometry() const noexcept { size_t count = 0; for (const auto &o : _geometryShaderMap) if (o.second.referenceCounter > 0) count++; return count; }
inline size_t ShaderLibrary::countFragment() const noexcept { size_t count = 0; for (const auto &o : _fragmentShaderMap) if (o.second.referenceCounter > 0) count++; return count; }

inline void ShaderLibrary::setShaderModuleIdentifierEnabled(bool enabled, const CadR::VulkanDevice &device) { _useShaderModuleIdentifier = enabled; if (enabled) { vkGetShaderModuleIdentifierEXT = (PFN_vkGetShaderModuleIdentifierEXT)device.getProcAddr("vkGetShaderModuleIdentifierEXT"); if (!vkGetShaderModuleIdentifierEXT) throw std::runtime_error("Unsupported Vulkan functionality"); } }
inline bool ShaderLibrary::useShaderModuleIdentifier() const noexcept { return _useShaderModuleIdentifier; }
}
