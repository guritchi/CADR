#include <CadPL/ShaderLibrary.h>
#include <CadPL/ShaderGenerator.h>
#include <CadR/VulkanDevice.h>

#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace CadPL;

// shader code in SPIR-V binary
static const uint32_t vertexUberShaderSpirv[]={
#include "shaders/UberShader.vert.spv"
};
static const uint32_t vertexIdBufferUberShaderSpirv[]={
#include "shaders/UberShader-idBuffer.vert.spv"
};
static const uint32_t geometryUberShaderSpirv[]={
#include "shaders/UberShader.geom.spv"
};
static const uint32_t geometryIdBufferUberShaderSpirv[]={
#include "shaders/UberShader-idBuffer.geom.spv"
};
static const uint32_t fragmentUberShaderSpirv[]={
#include "shaders/UberShader.frag.spv"
};
static const uint32_t fragmentIdBufferUberShaderSpirv[]={
#include "shaders/UberShader-idBuffer.frag.spv"
};



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
	assert(_vertexShaderMap.empty() && "ShaderLibrary::~ShaderLibrary(): All SharedShaderModules must be released before destroying ShaderLibrary.");
	assert(_geometryShaderMap.empty() && "ShaderLibrary::~ShaderLibrary(): All SharedShaderModules must be released before destroying ShaderLibrary.");
	assert(_fragmentShaderMap.empty() && "ShaderLibrary::~ShaderLibrary(): All SharedShaderModules must be released before destroying ShaderLibrary.");

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

	switch(smObject->owningMap) {
	case OwningMap::eVertex:   smObject->shaderLibrary->_vertexShaderMap.erase(static_cast<ShaderModuleObject<VertexShaderMapKey>*>(smObject)->eraseIt); break;
	case OwningMap::eGeometry: smObject->shaderLibrary->_geometryShaderMap.erase(static_cast<ShaderModuleObject<GeometryShaderMapKey>*>(smObject)->eraseIt); break;
	case OwningMap::eFragment: smObject->shaderLibrary->_fragmentShaderMap.erase(static_cast<ShaderModuleObject<FragmentShaderMapKey>*>(smObject)->eraseIt); break;
	default:
		assert(0 && "ShaderModuleObject::owningMap contains unknown value.");
	}
}


SharedShaderModule ShaderLibrary::getOrCreateVertexShader(const ShaderState& state)
{
	VertexShaderMapKey key(state);
	auto [it, newRecord] = _vertexShaderMap.try_emplace(key);
	if(newRecord) {
		try {
			it->second.shaderModule = ShaderGenerator::createVertexShader(state, *_device);
		} catch(...) {
			_vertexShaderMap.erase(it);
			throw;
		}
		it->second.referenceCounter = 0;
		it->second.shaderLibrary = this;
		it->second.owningMap = OwningMap::eVertex;
		it->second.eraseIt = it;
	}
	return SharedShaderModule(&it->second);
}


SharedShaderModule ShaderLibrary::getOrCreateGeometryShader(const ShaderState& state)
{
	GeometryShaderMapKey key(state);
	auto [it, newRecord] = _geometryShaderMap.try_emplace(key);
	if(newRecord) {
		try {
			it->second.shaderModule = ShaderGenerator::createGeometryShader(state, *_device);
            // if (!it->second.shaderModule) {
            //     _geometryShaderMap.erase(it);
            //     return {};
            // }
		} catch(...) {
			_geometryShaderMap.erase(it);
			throw;
		}
		it->second.referenceCounter = 0;
		it->second.shaderLibrary = this;
		it->second.owningMap = OwningMap::eGeometry;
		it->second.eraseIt = it;
	}
	return SharedShaderModule(&it->second);
}


SharedShaderModule ShaderLibrary::getOrCreateFragmentShader(const ShaderState& state)
{
	FragmentShaderMapKey key(state);
	auto [it, newRecord] = _fragmentShaderMap.try_emplace(key);
	if(newRecord) {
		try {
			it->second.shaderModule = ShaderGenerator::createFragmentShader(state, *_device);
		} catch(...) {
			_fragmentShaderMap.erase(it);
			throw;
		}
		it->second.referenceCounter = 0;
		it->second.shaderLibrary = this;
		it->second.owningMap = OwningMap::eFragment;
		it->second.eraseIt = it;
	}
	return SharedShaderModule(&it->second);
}

std::string ShaderState::serialize() const {
	std::stringstream ss;
	ss << std::hex;
	const uint8_t* data = reinterpret_cast<const uint8_t*>(this); // ShaderState must not contain pointers!
	for(size_t i = 0; i < sizeof(ShaderState); ++i) {
#if 0
		ss << std::setw(2) << std::setfill('0') << (int)data[i];
#else // runlength encode zeros
		uint8_t c = data[i];
		if (c != 0) {
			ss << std::setw(2) << std::setfill('0') << (int)c;
			continue;
		}
		uint8_t count = 1;
		while (i < sizeof(ShaderState) - 1 && c == data[i + 1]) {
			count++;
			i++;
		}
		if (count > 1) {
			ss << '_';
			ss << std::setw(2) << std::setfill('0') << (int)count;
			continue;
		}
		ss << '-';
#endif
	}
	return ss.str() ;
}

static const char* attributeTypeToString(int type) {
    switch (type) {
        case 0x01: return "float4, alignment 16";
        case 0x02: return "half4, alignment 8";
        case 0x03: return "half4, alignment 4";
        case 0x04: return "half4, alignment 4, reads the values with additional offset +2";
        case 0x05: return "uint4 normalized, alignment 16";
        case 0x06: return "uint4, alignment 16";
        case 0x07: return "int4 normalized, alignment 16";
        case 0x08: return "int4, alignment 16";
        case 0x09: return "ushort4 normalized, alignment 8";
        case 0x0a: return "ushort4, alignment 8";
        case 0x0b: return "ushort4 normalized, alignment 4";
        case 0x0c: return "ushort4, alignment 4";
        case 0x0d: return "ushort4 normalized, alignment 4, reads the values with additional offset +2";
        case 0x0e: return "ushort4, alignment 4, reads the values with additional offset +2";
        case 0x0f: return "short4 normalized, alignment 8";
        case 0x10: return "short4, alignment 8";
        case 0x11: return "short4 normalized, alignment 4";
        case 0x12: return "short4, alignment 4";
        case 0x13: return "short4 normalized, alignment 4, reads the values with additional offset +2";
        case 0x14: return "short4, alignment 4, reads the values with additional offset +2";
        case 0x15: return "ubyte4 normalize, alignment 4";
        case 0x16: return "ubyte4, alignment 4";
        case 0x17: return "ubyte4 normalize, alignment 4, reads the values with additional offset +2";
        case 0x18: return "ubyte4, alignment 4, reads the values with additional offset +2";
        case 0x19: return "byte4 normalize, alignment 4";
        case 0x1a: return "byte4, alignment 4";
        case 0x1b: return "byte4 normalize, alignment 4, reads the values with additional offset +2";
        case 0x1c: return "byte4, alignment 4, reads the values with additional offset +2";
        case 0x20: return "float3, alignment 16";
        case 0x21: return "float3, alignment 4";
        case 0x22: return "half3, alignment 4, on 8 bytes reads first six bytes";
        case 0x23: return "half3, alignment 4, on 8 bytes reads last six bytes";
        case 0x24: return "uint3, alignment 16, normalize";
        case 0x25: return "uint3, alignment 16";
        case 0x26: return "uint3, alignment 4, normalize";
        case 0x27: return "uint3, alignment 4";
        case 0x28: return "int3, alignment 16, normalize";
        case 0x29: return "int3, alignment 16";
        case 0x2a: return "int3, alignment 4, normalize";
        case 0x2b: return "int3, alignment 4";
        case 0x2c: return "ushort3, alignment 4, on 8 bytes reads first six bytes, normalize";
        case 0x2d: return "ushort3, alignment 4, on 8 bytes reads first six bytes";
        case 0x2e: return "ushort3, alignment 4, on 8 bytes reads last six bytes, normalize";
        case 0x2f: return "ushort3, alignment 4, on 8 bytes reads last six bytes";
        case 0x30: return "short3, alignment 4, on 8 bytes reads first six bytes, normalize";
        case 0x31: return "short3, alignment 4, on 8 bytes reads first six bytes";
        case 0x32: return "short3, alignment 4, on 8 bytes reads last six bytes, normalize";
        case 0x33: return "short3, alignment 4, on 8 bytes reads last six bytes";
        case 0x34: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes, normalize";
        case 0x35: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes";
        case 0x36: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes, normalize";
        case 0x37: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes";
        case 0x38: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize";
        case 0x39: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2";
        case 0x3a: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize";
        case 0x3b: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2";
        case 0x3c: return "byte3, alignment 4, on 4 bytes extracts first three bytes, normalize";
        case 0x3d: return "byte3, alignment 4, on 4 bytes extracts first three bytes";
        case 0x3e: return "byte3, alignment 4, on 4 bytes extracts last three bytes, normalize";
        case 0x3f: return "byte3, alignment 4, on 4 bytes extracts last three bytes";
        case 0x40: return "byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize";
        case 0x41: return "byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2";
        case 0x42: return "byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize";
        case 0x43: return "byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2";
        case 0x50: return "float2, alignment 8";
        case 0x51: return "float2, alignment 4";
        case 0x52: return "half2, alignment 4";
        case 0x53: return "half2, alignment 4, reads the values with additional offset +2";
        case 0x54: return "uint2, alignment 8, normalize";
        case 0x55: return "uint2, alignment 8";
        case 0x56: return "uint2, alignment 4, normalize";
        case 0x57: return "uint2, alignment 4";
        case 0x58: return "int2, alignment 8, normalize";
        case 0x59: return "int2, alignment 8";
        case 0x5a: return "int2, alignment 4, normalize";
        case 0x5b: return "int2, alignment 4";
        case 0x5c: return "ushort2, alignment 4, normalize";
        case 0x5d: return "ushort2, alignment 4";
        case 0x5e: return "ushort2, alignment 4, reads the values with additional offset +2, normalize";
        case 0x5f: return "ushort2, alignment 4, reads the values with additional offset +2";
        case 0x60: return "short2, alignment 4, normalize";
        case 0x61: return "short2, alignment 4";
        case 0x62: return "short2, alignment 4, reads the values with additional offset +2, normalize";
        case 0x63: return "short2, alignment 4, reads the values with additional offset +2";
        case 0x64: return "ubyte2, alignment 4, normalize";
        case 0x65: return "ubyte2, alignment 4";
        case 0x66: return "ubyte2, alignment 4, reads the values with additional offset +1, normalize";
        case 0x67: return "ubyte2, alignment 4, reads the values with additional offset +1";
        case 0x68: return "ubyte2, alignment 4, reads the values with additional offset +2, normalize";
        case 0x69: return "ubyte2, alignment 4, reads the values with additional offset +2";
        case 0x6a: return "ubyte2, alignment 4, reads the values with additional offset +3, normalize";
        case 0x6b: return "ubyte2, alignment 4, reads the values with additional offset +3";
        case 0x6c: return "byte2, alignment 4, normalize";
        case 0x6d: return "byte2, alignment 4";
        case 0x6e: return "byte2, alignment 4, reads the values with additional offset +1, normalize";
        case 0x6f: return "byte2, alignment 4, reads the values with additional offset +1";
        case 0x70: return "byte2, alignment 4, reads the values with additional offset +2, normalize";
        case 0x71: return "byte2, alignment 4, reads the values with additional offset +2";
        case 0x72: return "byte2, alignment 4, reads the values with additional offset +3, normalize";
        case 0x73: return "byte2, alignment 4, reads the values with additional offset +3";
        case 0x80: return "float, alignment 4";
        case 0x81: return "half, alignment 4";
        case 0x82: return "half, alignment 4, reads the values with additional offset +2";
        case 0x83: return "uint, alignment 4, normalize";
        case 0x84: return "uint, alignment 4";
        case 0x85: return "int2, alignment 4, normalize";
        case 0x86: return "int2, alignment 4";
        case 0x87: return "ushort, alignment 4, normalize";
        case 0x88: return "ushort, alignment 4";
        case 0x89: return "ushort, alignment 4, reads the values with additional offset +2, normalize";
        case 0x8a: return "ushort, alignment 4, reads the values with additional offset +2";
        case 0x8b: return "short, alignment 4, normalize";
        case 0x8c: return "short, alignment 4";
        case 0x8d: return "short, alignment 4, reads the values with additional offset +2, normalize";
        case 0x8e: return "short, alignment 4, reads the values with additional offset +2";
        case 0x8f: return "ubyte, alignment 4, normalize";
        case 0x90: return "ubyte, alignment 4";
        case 0x91: return "ubyte, alignment 4, reads the values with additional offset +1, normalize";
        case 0x92: return "ubyte, alignment 4, reads the values with additional offset +1";
        case 0x93: return "ubyte, alignment 4, reads the values with additional offset +2, normalize";
        case 0x94: return "ubyte, alignment 4, reads the values with additional offset +2";
        case 0x95: return "ubyte, alignment 4, reads the values with additional offset +3, normalize";
        case 0x96: return "ubyte, alignment 4, reads the values with additional offset +3";
        case 0x97: return "byte, alignment 4, normalize";
        case 0x98: return "byte, alignment 4";
        case 0x99: return "byte, alignment 4, reads the values with additional offset +1, normalize";
        case 0x9a: return "byte, alignment 4, reads the values with additional offset +1";
        case 0x9b: return "byte, alignment 4, reads the values with additional offset +2, normalize";
        case 0x9c: return "byte, alignment 4, reads the values with additional offset +2";
        case 0x9d: return "byte, alignment 4, reads the values with additional offset +3, normalize";
        case 0x9e: return "byte, alignment 4, reads the values with additional offset +3";
        default:
            return "unknown";
    }
}

std::string ShaderState::debugDump() const {
	std::stringstream ss;
	// bits 0..1: 0 - reserved, 1 - unlit, 2 - phong, 3 - metallicRoughness
    // bits 2..7: texture offset (0, 4, 8, 12, .....252)
    // bit 8: use color attribute for ambient and diffuse; material ambient and diffuse values are ignored
    // bit 9: use color attribute for diffuse; material diffuse value is ignored
    // bit 10: ignore color attribute alpha if color attribute is used (if bit 8 or 9 is set)
    // bit 11: ignore material alpha
    // bit 12: ignore base texture alpha if base texture is used
    const auto type = materialSetup & 0x03;
    const auto textureOffset = materialSetup & 0xFC;
    ss << "  opt: " << std::hex << optimizeFlags <<  std::dec << "  ";
    ss << "  material: " << std::hex << materialSetup << std::dec << "  ";
    if (type == 0) {
        ss << "unlit";
    }
    else if (type == 1) {
        ss << "phong";
    }
    else if (type == 2) {
        ss << "metallicRoughness";
    }
    ss << "\n";
    ss << "    textureOffset: " << textureOffset << "B\n";
    if (materialSetup & 0x100) {
        ss << "    use color attribute for ambient and diffuse\n";
    }
    if (materialSetup & 0x200) {
        ss << "    use color attribute for diffuse\n";
    }
    if (materialSetup & 0x400) {
        ss << "    ignore color attribute alpha if\n";
    }
    if (materialSetup & 0x800) {
        ss << "    ignore material alpha\n";
    }
    if (materialSetup & 0x1000) {
        ss << "    ignore base texture alpha\n";
    }

    ss << "    attributes: \n";
    for (int i = 0; i < attribAccessInfo.size(); ++i) {
        int type = attribAccessInfo[i] >> 8;
        int offset = attribAccessInfo[i] & 0xFF;
        if (type == 0) {
	        continue;
        }
        ss << "    [" << i << "]: " << type << "(" << attributeTypeToString(type) << "), " << offset << "B";
        if (i == 0) {
            ss << "  pos";
        }
        else if (i == 1) {
            ss << "  normal";
        }
        else if (i == 2) {
            ss << "  tangent";
        }
        else if (i == 3) {
            ss << "  color";
        }
        else {
            ss << "  texcoord";
        }
        ss << "\n";
    }

    ss << "  textures(" << (int)numTextures << "): \n";
    for (size_t i = 0; i < textureSetup.size(); ++i) {
        const auto texCoordIndex = textureSetup[i] & 0xFF;
        const auto type = (textureSetup[i] & 0xFF00) >> 8;
    	const auto size = textureSetup[i] >> 26;
    	if (type == 0) {
    		continue;
    	}
        ss << "    [" << i << "]: " << std::hex << textureSetup[i] << std::dec << " ";
        switch (type) {
            case 1:
            	ss << "normal ";
            	break;
            case 2:
            	ss << "occlusion ";
            	break;
            case 3:
            	ss << "emissive ";
            	break;
            case 4:
            	ss << "base ";
            	break;
        	case 5:
        		ss << "metallicRoughness";
        		break;
        }
        ss << "coordIndex: " << texCoordIndex << " size: " << size << "B" << "\n";
    }

	return ss.str() ;
}

bool ShaderState::operator<(const ShaderState& rhs) const
{
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(attribSetup < rhs.attribSetup)  return true;
	if(attribSetup > rhs.attribSetup)  return false;
	if(materialSetup < rhs.materialSetup)  return false;
	if(materialSetup > rhs.materialSetup)  return true;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	if(primitiveTopology < rhs.primitiveTopology)  return false;
	if(primitiveTopology > rhs.primitiveTopology)  return true;
	if(attribAccessInfo < rhs.attribAccessInfo)  return true;
	if(attribAccessInfo > rhs.attribAccessInfo)  return false;
	if(textureSetup < rhs.textureSetup)  return true;
	if(textureSetup > rhs.textureSetup)  return false;
	if(lightSetup < rhs.lightSetup)  return true;
	if(lightSetup > rhs.lightSetup)  return false;
	return projectionHandling < rhs.projectionHandling;
}

bool ShaderLibrary::VertexShaderMapKey::operator<(const ShaderLibrary::VertexShaderMapKey& rhs) const  {
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	if(primitiveTopology < rhs.primitiveTopology)  return false;
	if(primitiveTopology > rhs.primitiveTopology)  return true;
	if(attribAccessInfo < rhs.attribAccessInfo)  return true;
	if(attribAccessInfo > rhs.attribAccessInfo)  return false;
	return projectionHandling < rhs.projectionHandling;
}

bool ShaderLibrary::GeometryShaderMapKey::operator<(const ShaderLibrary::GeometryShaderMapKey& rhs) const  {
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	if(primitiveTopology < rhs.primitiveTopology)  return false;
	if(primitiveTopology > rhs.primitiveTopology)  return true;
	if(attribAccessInfo < rhs.attribAccessInfo)  return true;
	if(attribAccessInfo > rhs.attribAccessInfo)  return false;
	return projectionHandling < rhs.projectionHandling;
}

bool ShaderLibrary::FragmentShaderMapKey::operator<(const ShaderLibrary::FragmentShaderMapKey& rhs) const {
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(materialSetup < rhs.materialSetup)  return false;
	if(materialSetup > rhs.materialSetup)  return true;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	if(primitiveTopology < rhs.primitiveTopology)  return false;
	if(primitiveTopology > rhs.primitiveTopology)  return true;
	if(attribAccessInfo < rhs.attribAccessInfo)  return true;
	if(attribAccessInfo > rhs.attribAccessInfo)  return false;
	if(textureSetup < rhs.textureSetup)  return true;
	if(textureSetup > rhs.textureSetup)  return false;
	if(lightSetup < rhs.lightSetup)  return true;
	if(lightSetup > rhs.lightSetup)  return false;
	return false;
}
