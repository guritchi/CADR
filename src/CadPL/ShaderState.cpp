#include "ShaderState.h"

#include <sstream>
#include <iomanip>

using namespace CadPL;

VertexShaderState::VertexShaderState(const ShaderState& shaderState)
    : optimizeFlags(shaderState.optimizeFlags)
    , primitiveTopology(shaderState.primitiveTopology)
    , projectionHandling(shaderState.projectionHandling)
    , numAttributes(shaderState.numAttributes)
    , idBuffer(shaderState.idBuffer)
    , attribSetup(shaderState.attribSetup)
    , attribAccessInfo(shaderState.attribAccessInfo)
{}
VertexShaderState::VertexShaderState(const std::string& serializedState)
{
    deserialize(serializedState);
}

GeometryShaderState::GeometryShaderState(const ShaderState& shaderState)
    : optimizeFlags(shaderState.optimizeFlags)
    , primitiveTopology(shaderState.primitiveTopology)
    , projectionHandling(shaderState.projectionHandling)
    , numAttributes(shaderState.numAttributes)
    , idBuffer(shaderState.idBuffer)
    , attribSetup(shaderState.attribSetup)
    , attribAccessInfo(shaderState.attribAccessInfo)
{}
GeometryShaderState::GeometryShaderState(const std::string& serializedState)
{
    deserialize(serializedState);
}

FragmentShaderState::FragmentShaderState(const ShaderState& shaderState)
    : optimizeFlags(shaderState.optimizeFlags)
    , materialSetup(shaderState.materialSetup)
    , numAttributes(shaderState.numAttributes)
    , numTextures(shaderState.numTextures)
    , numLights(shaderState.numLights)
    , attribSetup(shaderState.attribSetup)
    , attribAccessInfo(shaderState.attribAccessInfo)
    , textureSetup(shaderState.textureSetup)
    , lightSetup(shaderState.lightSetup)
    , idBuffer(shaderState.idBuffer)
{}
FragmentShaderState::FragmentShaderState(const std::string& serializedState)
{
    deserialize(serializedState);
}

class Serializer {
	std::stringstream buffer;
	size_t zeros = 0;

	void finishZeros() {
		if (zeros == 1) {
			buffer << '-';
			zeros = 0;
		}
		else {
			while (zeros > 0) {
				auto count = std::min<size_t>(zeros, 26);
				buffer << char(63 + count);
				zeros -= count;
			}
		}
	}

public:
	void put(const uint8_t *src, size_t size) {
		for (size_t i = 0; i < size; ++i) {
			auto c = src[i];
			if (c == 0) {
				zeros++;
			}
			else {
				finishZeros();
				buffer << std::hex << std::setw(2) << std::setfill('0') << (int)c;
			}
		}
	}

	template<typename T>
	void put(const T &src) {
		put(reinterpret_cast<const uint8_t*>(&src), sizeof(src));
	}

	std::string str() {
		auto str = buffer.str();
		if (str.empty()) {
			return "0";
		}
		return str;
	}

};

struct Deserializer {
 	const uint8_t* ptr = {};
	const uint8_t* end = {};
	size_t zeros = 0;

	Deserializer(const uint8_t* data, size_t size) : ptr(data), end(data + size)
	{}

	void get(uint8_t *dst, size_t size) {
		const uint8_t *dstend = dst + size;
		uint8_t buf[4] = {};
		while (dst != dstend) {
			while (zeros > 0) {
				*dst = 0;
				dst++;
				zeros--;
				if (dst == dstend) {
					return;
				}
			}
			if (ptr == end) {
				while (dst != dstend) {
					*dst = 0;
					dst++;
				}
				break;
			}
			buf[0] = *ptr;
			ptr++;
			if (buf[0] == '-') {
				*dst = 0;
				dst++;
			}
			else if (buf[0] >= 65 && buf[0] <= 90) {
				zeros += buf[0] - 63;
			}
			else {
				if (ptr == end) {
					buf[1] = 0;
				}
				else {
					buf[1] = *ptr;
					ptr++;;
				}
				unsigned short byte;
				std::istringstream input(reinterpret_cast<const char*>(buf));
				input >> std::hex >> byte;
				*dst = byte % 0x100;
				dst++;
			}
		}
	}

 	template<typename T>
	void get(T &dst) {
		get(reinterpret_cast<uint8_t*>(&dst), sizeof(dst));
 	}

};

std::string ShaderState::serialize() const {
	Serializer s;
	s.put(optimizeFlags);
	s.put(projectionHandling);
	s.put(primitiveTopology);
	s.put(materialSetup);
	s.put(attribSetup);
	s.put(idBuffer);
	s.put(numAttributes);
	s.put(numTextures);
	s.put(numLights);
	s.put(reinterpret_cast<const uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	s.put(reinterpret_cast<const uint8_t*>(textureSetup.data()), numTextures * sizeof(textureSetup[0]));
	s.put(reinterpret_cast<const uint8_t*>(lightSetup.data()), numLights * sizeof(lightSetup[0]));
	return s.str();
}

void ShaderState::deserialize(const std::string &serializedState) {
	Deserializer d(reinterpret_cast<const uint8_t*>(serializedState.data()), serializedState.size());
	d.get(optimizeFlags);
	d.get(projectionHandling);
	d.get(primitiveTopology);
	d.get(materialSetup);
	d.get(attribSetup);
	d.get(idBuffer);
	d.get(numAttributes);
	d.get(numTextures);
	d.get(numLights);
	if (numAttributes > 0) {
		d.get(reinterpret_cast<uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	}
	if (numTextures > 0) {
		d.get(reinterpret_cast<uint8_t*>(textureSetup.data()), numTextures * sizeof(textureSetup[0]));
	}
	if (numLights > 0) {
		d.get(reinterpret_cast<uint8_t*>(lightSetup.data()), numLights * sizeof(lightSetup[0]));
	}
}

std::string VertexShaderState::serialize() const {
	Serializer s;
	s.put(optimizeFlags);
	s.put(projectionHandling);
	s.put(primitiveTopology);
	s.put(attribSetup);
	s.put(idBuffer);
	s.put(numAttributes);
	s.put(reinterpret_cast<const uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	return s.str();
}

void VertexShaderState::deserialize(const std::string &serializedState) {
	Deserializer d(reinterpret_cast<const uint8_t*>(serializedState.data()), serializedState.size());
	d.get(optimizeFlags);
	d.get(projectionHandling);
	d.get(primitiveTopology);
	d.get(attribSetup);
	d.get(idBuffer);
	d.get(numAttributes);
	if (numAttributes > 0) {
		d.get(reinterpret_cast<uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	}
}

std::string GeometryShaderState::serialize() const {
	Serializer s;
	s.put(optimizeFlags);
	s.put(projectionHandling);
	s.put(primitiveTopology);
	s.put(attribSetup);
	s.put(idBuffer);
	s.put(numAttributes);
	s.put(reinterpret_cast<const uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	return s.str();
}

void GeometryShaderState::deserialize(const std::string &serializedState) {
	Deserializer d(reinterpret_cast<const uint8_t*>(serializedState.data()), serializedState.size());
	d.get(optimizeFlags);
	d.get(projectionHandling);
	d.get(primitiveTopology);
	d.get(attribSetup);
	d.get(idBuffer);
	d.get(numAttributes);
	if (numAttributes > 0) {
		d.get(reinterpret_cast<uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	}
}

void FragmentShaderState::deserialize(const std::string &serializedState) {
	Deserializer d(reinterpret_cast<const uint8_t*>(serializedState.data()), serializedState.size());
	d.get(optimizeFlags);
	d.get(materialSetup);
	d.get(attribSetup);
	d.get(idBuffer);
	d.get(numAttributes);
	d.get(numTextures);
	d.get(numLights);
	if (numAttributes > 0) {
		d.get(reinterpret_cast<uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	}
	if (numTextures > 0) {
		d.get(reinterpret_cast<uint8_t*>(textureSetup.data()), numTextures * sizeof(textureSetup[0]));
	}
	if (numLights > 0) {
		d.get(reinterpret_cast<uint8_t*>(lightSetup.data()), numLights * sizeof(lightSetup[0]));
	}
}

std::string FragmentShaderState::serialize() const {
	Serializer s;
	s.put(optimizeFlags);
	s.put(materialSetup);
	s.put(attribSetup);
	s.put(idBuffer);
	s.put(numAttributes);
	s.put(numTextures);
	s.put(numLights);
	s.put(reinterpret_cast<const uint8_t*>(attribAccessInfo.data()), numAttributes * sizeof(attribAccessInfo[0]));
	s.put(reinterpret_cast<const uint8_t*>(textureSetup.data()), numTextures * sizeof(textureSetup[0]));
	s.put(reinterpret_cast<const uint8_t*>(lightSetup.data()), numLights * sizeof(lightSetup[0]));
	return s.str();
}

std::string ShaderState::optimizeFlagsToString(std::bitset<numOptimizeFlags> optimizeFlags) {
	auto flags = optimizeFlags.to_ulong();
	std::string out = "Optimize";
	if (flags == OptimizeNone.to_ulong()) {
		out += "None";
	}
	else if ((flags & OptimizeAll.to_ulong()) == OptimizeAll.to_ulong()) {
		out += "All";
	}
	else {
		out += "{";
		if (flags & OptimizeAttribs.to_ulong()) {
			out += "Attribs|";
		}
		if ((flags & OptimizeMaterial.to_ulong()) == OptimizeMaterial.to_ulong()) {
			out += "Material|";
		}
		else {
			if (flags & OptimizeMaterialModel.to_ulong()) out += "MaterialModel|";
			if (flags & OptimizeMaterialColorAttribute.to_ulong()) out += "MaterialColorAttribute|";
			if (flags & OptimizeMaterialAlpha.to_ulong()) out += "MaterialAlpha|";
		}
		if ((flags & OptimizeTextures.to_ulong()) == OptimizeTextures.to_ulong()) {
			out += "Textures|";
		}
		else {
			if (flags & OptimizeTextureFlags.to_ulong()) out += "TextureFlags|";
			if (flags & OptimizeTextureTypesAndTexCoordIndices.to_ulong()) out += "TextureTypesAndTexCoordIndices|";
		}
		if ((flags & OptimizeLights.to_ulong()) == OptimizeLights.to_ulong()) {
			out += "Lights|";
		}
		else {
			if (flags & OptimizeLightTypes.to_ulong()) out += "LightTypes|";
		}
		if (*out.rbegin() == '|') {
			out.erase(out.begin() + out.size() - 1);
		}
		out += "}";
	}
	return out;
}

static const char* attributeTypeToString(AttributeType type) {
    switch (type) {
    	case AttributeType::undefined: return "undefined";
        case AttributeType::vec4A16: return "float4, alignment 16";
        case AttributeType::half4A8: return "half4, alignment 8";
        case AttributeType::half4A4: return "half4, alignment 4";
        case AttributeType::half4A4Offset2: return "half4, alignment 4, reads the values with additional offset +2";
        case AttributeType::uint4A16Norm: return "uint4 normalized, alignment 16";
        case AttributeType::uint4A16: return "uint4, alignment 16";
        case AttributeType::int4A16Norm: return "int4 normalized, alignment 16";
        case AttributeType::int4A16: return "int4, alignment 16";
        case AttributeType::ushort4A8Norm: return "ushort4 normalized, alignment 8";
        case AttributeType::ushort4A8: return "ushort4, alignment 8";
        case AttributeType::ushort4A4Norm: return "ushort4 normalized, alignment 4";
        case AttributeType::ushort4A4: return "ushort4, alignment 4";
        case AttributeType::ushort4A4NormOffset2: return "ushort4 normalized, alignment 4, reads the values with additional offset +2";
        case AttributeType::ushort4A4Offset2: return "ushort4, alignment 4, reads the values with additional offset +2";
        case AttributeType::short4A8Norm: return "short4 normalized, alignment 8";
        case AttributeType::short4A8: return "short4, alignment 8";
        case AttributeType::short4A4Norm: return "short4 normalized, alignment 4";
        case AttributeType::short4A4: return "short4, alignment 4";
        case AttributeType::short4A4NormOffset2: return "short4 normalized, alignment 4, reads the values with additional offset +2";
        case AttributeType::short4A4Offset2: return "short4, alignment 4, reads the values with additional offset +2";
    	case AttributeType::ubyte4A4Norm: return "ubyte4 normalize, alignment 4";
        case AttributeType::ubyte4A4: return "ubyte4, alignment 4";
        case AttributeType::ubyte4A4NormOffset2: return "ubyte4 normalize, alignment 4, reads the values with additional offset +2";
        case AttributeType::ubyte4A4Offset2: return "ubyte4, alignment 4, reads the values with additional offset +2";
        case AttributeType::byte4A4Norm: return "byte4 normalize, alignment 4";
        case AttributeType::byte4A4: return "byte4, alignment 4";
        case AttributeType::byte4A4NormOffset2: return "byte4 normalize, alignment 4, reads the values with additional offset +2";
        case AttributeType::byte4A4Offset2: return "byte4, alignment 4, reads the values with additional offset +2";
        case AttributeType::vec3A16: return "float3, alignment 16";
        case AttributeType::vec3A4: return "float3, alignment 4";
        case AttributeType::half3A4: return "half3, alignment 4, on 8 bytes reads first six bytes";
        case AttributeType::half3A4Last6: return "half3, alignment 4, on 8 bytes reads last six bytes";
        case AttributeType::uint3A16Norm: return "uint3, alignment 16, normalize";
        case AttributeType::uint3A16: return "uint3, alignment 16";
        case AttributeType::uint3A4Norm: return "uint3, alignment 4, normalize";
        case AttributeType::uint3A4: return "uint3, alignment 4";
        case AttributeType::int3A16Norm: return "int3, alignment 16, normalize";
        case AttributeType::int3A16: return "int3, alignment 16";
        case AttributeType::int3A4Norm: return "int3, alignment 4, normalize";
        case AttributeType::int3A4: return "int3, alignment 4";
        case AttributeType::ushort3A4NormFirst6: return "ushort3, alignment 4, on 8 bytes reads first six bytes, normalize";
        case AttributeType::ushort3A4First6: return "ushort3, alignment 4, on 8 bytes reads first six bytes";
        case AttributeType::ushort3A4NormLast6: return "ushort3, alignment 4, on 8 bytes reads last six bytes, normalize";
        case AttributeType::ushort3A4Last6: return "ushort3, alignment 4, on 8 bytes reads last six bytes";
        case AttributeType::short3A4NormFirst6: return "short3, alignment 4, on 8 bytes reads first six bytes, normalize";
        case AttributeType::short3A4First6: return "short3, alignment 4, on 8 bytes reads first six bytes";
        case AttributeType::short3A4NormLast6: return "short3, alignment 4, on 8 bytes reads last six bytes, normalize";
        case AttributeType::short3A4Last6: return "short3, alignment 4, on 8 bytes reads last six bytes";
    	case AttributeType::ubyte3A4NormFirst3: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes, normalize";
        case AttributeType::ubyte3A4First3: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes";
        case AttributeType::ubyte3A4NormLast3: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes, normalize";
        case AttributeType::ubyte3A4Last3: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes";
        case AttributeType::ubyte3A4NormFirst3Offset2: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize";
        case AttributeType::ubyte3A4First3Offset2: return "ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2";
        case AttributeType::ubyte3A4NormLast3Offset2: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize";
        case AttributeType::ubyte3A4Last3Offset2: return "ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2";
        case AttributeType::byte3A4NormFirst3: return "byte3, alignment 4, on 4 bytes extracts first three bytes, normalize";
        case AttributeType::byte3A4First3: return "byte3, alignment 4, on 4 bytes extracts first three bytes";
        case AttributeType::byte3A4NormLast3: return "byte3, alignment 4, on 4 bytes extracts last three bytes, normalize";
        case AttributeType::byte3A4Last3: return "byte3, alignment 4, on 4 bytes extracts last three bytes";
        case AttributeType::byte3A4NormFirst3Offset2: return "byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize";
        case AttributeType::byte3A4First3Offset2: return "byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2";
        case AttributeType::byte3A4NormLast3Offset2: return "byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize";
        case AttributeType::byte3A4Last3Offset2: return "byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2";
        case AttributeType::vec2A8: return "float2, alignment 8";
        case AttributeType::vec2A4: return "float2, alignment 4";
        case AttributeType::half2A4: return "half2, alignment 4";
        case AttributeType::half2A4Offset2: return "half2, alignment 4, reads the values with additional offset +2";
        case AttributeType::uint2A8Norm: return "uint2, alignment 8, normalize";
        case AttributeType::uint2A8: return "uint2, alignment 8";
        case AttributeType::uint2A4Norm: return "uint2, alignment 4, normalize";
        case AttributeType::uint2A4: return "uint2, alignment 4";
        case AttributeType::int2A8Norm: return "int2, alignment 8, normalize";
        case AttributeType::int2A8: return "int2, alignment 8";
        case AttributeType::int2A4Norm: return "int2, alignment 4, normalize";
        case AttributeType::int2A4: return "int2, alignment 4";
        case AttributeType::ushort2A4Norm: return "ushort2, alignment 4, normalize";
        case AttributeType::ushort2A4: return "ushort2, alignment 4";
        case AttributeType::ushort2A4NormOffset2: return "ushort2, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::ushort2A4Offset2: return "ushort2, alignment 4, reads the values with additional offset +2";
        case AttributeType::short2A4Norm: return "short2, alignment 4, normalize";
        case AttributeType::short2A4: return "short2, alignment 4";
        case AttributeType::short2A4NormOffset2: return "short2, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::short2A4Offset2: return "short2, alignment 4, reads the values with additional offset +2";
        case AttributeType::ubyte2A4Norm: return "ubyte2, alignment 4, normalize";
        case AttributeType::ubyte2A4: return "ubyte2, alignment 4";
        case AttributeType::ubyte2A4NormOffset1: return "ubyte2, alignment 4, reads the values with additional offset +1, normalize";
        case AttributeType::ubyte2A4Offset1: return "ubyte2, alignment 4, reads the values with additional offset +1";
        case AttributeType::ubyte2A4NormOffset2: return "ubyte2, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::ubyte2A4Offset2: return "ubyte2, alignment 4, reads the values with additional offset +2";
        case AttributeType::ubyte2A4NormOffset3: return "ubyte2, alignment 4, reads the values with additional offset +3, normalize";
        case AttributeType::ubyte2A4Offset3: return "ubyte2, alignment 4, reads the values with additional offset +3";
        case AttributeType::byte2A4Norm: return "byte2, alignment 4, normalize";
        case AttributeType::byte2A4: return "byte2, alignment 4";
        case AttributeType::byte2A4NormOffset1: return "byte2, alignment 4, reads the values with additional offset +1, normalize";
        case AttributeType::byte2A4Offset1: return "byte2, alignment 4, reads the values with additional offset +1";
        case AttributeType::byte2A4NormOffset2: return "byte2, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::byte2A4Offset2: return "byte2, alignment 4, reads the values with additional offset +2";
        case AttributeType::byte2A4NormOffset3: return "byte2, alignment 4, reads the values with additional offset +3, normalize";
        case AttributeType::byte2A4Offset3: return "byte2, alignment 4, reads the values with additional offset +3";
        case AttributeType::floatA8: return "float, alignment 4";
        case AttributeType::halfA4: return "half, alignment 4";
        case AttributeType::halfA4Offset2: return "half, alignment 4, reads the values with additional offset +2";
        case AttributeType::uintA4Norm: return "uint, alignment 4, normalize";
        case AttributeType::uintA4: return "uint, alignment 4";
        case AttributeType::intA4Norm: return "int, alignment 4, normalize";
        case AttributeType::intA4: return "int, alignment 4";
        case AttributeType::ushortA4Norm: return "ushort, alignment 4, normalize";
        case AttributeType::ushortA4: return "ushort, alignment 4";
        case AttributeType::ushortA4NormOffset2: return "ushort, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::ushortA4Offset2: return "ushort, alignment 4, reads the values with additional offset +2";
        case AttributeType::shortA4Norm: return "short, alignment 4, normalize";
        case AttributeType::shortA4: return "short, alignment 4";
        case AttributeType::shortA4NormOffset2: return "short, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::shortA4Offset2: return "short, alignment 4, reads the values with additional offset +2";
        case AttributeType::ubyteA4Norm: return "ubyte, alignment 4, normalize";
        case AttributeType::ubyteA4: return "ubyte, alignment 4";
        case AttributeType::ubyteA4NormOffset1: return "ubyte, alignment 4, reads the values with additional offset +1, normalize";
        case AttributeType::ubyteA4Offset1: return "ubyte, alignment 4, reads the values with additional offset +1";
        case AttributeType::ubyteA4NormOffset2: return "ubyte, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::ubyteA4Offset2: return "ubyte, alignment 4, reads the values with additional offset +2";
        case AttributeType::ubyteA4NormOffset3: return "ubyte, alignment 4, reads the values with additional offset +3, normalize";
        case AttributeType::ubyteA4Offset3: return "ubyte, alignment 4, reads the values with additional offset +3";
        case AttributeType::byteA4Norm: return "byte, alignment 4, normalize";
        case AttributeType::byteA4: return "byte, alignment 4";
        case AttributeType::byteA4NormOffset1: return "byte, alignment 4, reads the values with additional offset +1, normalize";
        case AttributeType::byteA4Offset1: return "byte, alignment 4, reads the values with additional offset +1";
        case AttributeType::byteA4NormOffset2: return "byte, alignment 4, reads the values with additional offset +2, normalize";
        case AttributeType::byteA4Offset2: return "byte, alignment 4, reads the values with additional offset +2";
        case AttributeType::byteA4NormOffset3: return "byte, alignment 4, reads the values with additional offset +3, normalize";
        case AttributeType::byteA4Offset3: return "byte, alignment 4, reads the values with additional offset +3";
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

    ss << "    attributes(" << (int)numAttributes << "): \n";
    for (int i = 0; i < attribAccessInfo.size(); ++i) {
        auto type = static_cast<AttributeType>(attribAccessInfo[i] >> 8);
        int offset = attribAccessInfo[i] & 0xFF;
        if (type == AttributeType::undefined) {
	        continue;
        }
        ss << "    [" << i << "]: 0x" << std::hex << static_cast<uint32_t>(type) << std::dec << " (" << attributeTypeToString(type) << "), " << offset << "B";
        if (i == PositionAttributeIndex) {
            ss << "  pos";
        }
        else if (i == NormalAttributeIndex) {
            ss << "  normal";
        }
        else if (i == TangentAttributeIndex) {
            ss << "  tangent";
        }
        else if (i == ColorAttributeIndex) {
            ss << "  color";
        }
        else {
            ss << "  texcoord";
        }
        ss << "\n";
    }
	ss << "  vertexSize: " << (int)(attribSetup & 0x01fc) << "B\n";

    ss << "  textures(" << (int)numTextures << "): \n";
    for (size_t i = 0; i < textureSetup.size(); ++i) {
        const auto type = static_cast<TextureType>((textureSetup[i] & 0xFF00) >> 8);
    	if (type == TextureType::none) {
    		continue;
    	}
    	const auto texCoordIndex = textureSetup[i] & 0xFF;
    	const auto size = textureSetup[i] >> 26;
        ss << "    [" << i << "]: " << std::hex << textureSetup[i] << std::dec << " ";
        switch (type) {
            case TextureType::normal:
            	ss << "normal ";
            	break;
            case TextureType::occlusion:
            	ss << "occlusion ";
            	break;
            case TextureType::emissive:
            	ss << "emissive ";
            	break;
            case TextureType::base:
            	ss << "base ";
            	break;
        	case TextureType::metallicRoughness:
        		ss << "metallicRoughness ";
        		break;
        	case TextureType::sheen:
        		ss << "sheen ";
        		break;
        	case TextureType::sheenRoughness:
        		ss << "sheenRougness ";
        		break;
        	default:
        		ss << "unknown";
        		break;
        }
        ss << " coordIndex: " << texCoordIndex << " size: " << size << "B" << "\n";
    }

	return ss.str() ;
}

void ShaderState::set(const ShaderState &state, std::bitset<numOptimizeFlags> optimizeFlags) {
	idBuffer = state.idBuffer;
	primitiveTopology = state.primitiveTopology;
	projectionHandling = state.projectionHandling;
	if (optimizeFlags.to_ulong() & OptimizeMaterial.to_ulong()) {
		materialSetup = state.materialSetup;
	}
	// else {
	// 	materialSetup = {};
	// }
	if (optimizeFlags.to_ulong() & OptimizeAttribs.to_ulong()) {
		attribAccessInfo = state.attribAccessInfo;
		numAttributes = state.numAttributes;
		attribSetup = state.attribSetup;
	}
	// else{
	// 	attribAccessInfo = {};
	// 	attribSetup = {};
	// }
	if (optimizeFlags.to_ulong() & OptimizeLights.to_ulong()) {
		lightSetup = state.lightSetup;
		numLights = state.numLights;
	}
	// else {
	// 	lightSetup = {};
	// 	numLights = {};
	// }
	if (optimizeFlags.to_ulong() & OptimizeTextures.to_ulong()) {
		textureSetup = state.textureSetup;
		numTextures = state.numTextures;
	}
	// else {
	// 	textureSetup = {};
	// 	numTextures = {};
	// }
	this->optimizeFlags = optimizeFlags;
}

bool ShaderState::operator<(const ShaderState& rhs) const
{
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(materialSetup < rhs.materialSetup)  return false;
	if(materialSetup > rhs.materialSetup)  return true;
	if(primitiveTopology < rhs.primitiveTopology)  return false;
	if(primitiveTopology > rhs.primitiveTopology)  return true;
	if(numAttributes < rhs.numAttributes)  return true;
	if(numAttributes > rhs.numAttributes)  return false;
	if(numTextures < rhs.numTextures)  return true;
	if(numTextures > rhs.numTextures)  return false;
	if(numLights < rhs.numLights)  return true;
	if(numLights > rhs.numLights)  return false;
	if(attribSetup < rhs.attribSetup)  return true;
	if(attribSetup > rhs.attribSetup)  return false;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	for (size_t i = 0; i < numAttributes; ++i) {
		if(attribAccessInfo[i] < rhs.attribAccessInfo[i])  return true;
		if(attribAccessInfo[i] > rhs.attribAccessInfo[i])  return false;
	}
	for (size_t i = 0; i < numTextures; ++i) {
		if(textureSetup[i] < rhs.textureSetup[i])  return true;
		if(textureSetup[i] > rhs.textureSetup[i])  return false;
	}
	for (size_t i = 0; i < numLights; ++i) {
		if(lightSetup[i] < rhs.lightSetup[i])  return true;
		if(lightSetup[i] > rhs.lightSetup[i])  return false;
	}
	return projectionHandling < rhs.projectionHandling;
}

bool VertexShaderState::operator<(const VertexShaderState& rhs) const  {
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(primitiveTopology < rhs.primitiveTopology)  return false;
	if(primitiveTopology > rhs.primitiveTopology)  return true;
	if(numAttributes < rhs.numAttributes)  return true;
	if(numAttributes > rhs.numAttributes)  return false;
	if(attribSetup < rhs.attribSetup)  return true;
	if(attribSetup > rhs.attribSetup)  return false;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	for (size_t i = 0; i < numAttributes; ++i) {
		if(attribAccessInfo[i] < rhs.attribAccessInfo[i])  return true;
		if(attribAccessInfo[i] > rhs.attribAccessInfo[i])  return false;
	}
	return projectionHandling < rhs.projectionHandling;
}

bool GeometryShaderState::operator<(const GeometryShaderState& rhs) const  {
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(primitiveTopology < rhs.primitiveTopology)  return false;
	if(primitiveTopology > rhs.primitiveTopology)  return true;
	if(numAttributes < rhs.numAttributes)  return true;
	if(numAttributes > rhs.numAttributes)  return false;
	if(attribSetup < rhs.attribSetup)  return true;
	if(attribSetup > rhs.attribSetup)  return false;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	for (size_t i = 0; i < numAttributes; ++i) {
		if(attribAccessInfo[i] < rhs.attribAccessInfo[i])  return true;
		if(attribAccessInfo[i] > rhs.attribAccessInfo[i])  return false;
	}
	return projectionHandling < rhs.projectionHandling;
}

bool FragmentShaderState::operator<(const FragmentShaderState& rhs) const {
	if(optimizeFlags.to_ulong() < rhs.optimizeFlags.to_ulong())  return false;
	if(optimizeFlags.to_ulong() > rhs.optimizeFlags.to_ulong())  return true;
	if(materialSetup < rhs.materialSetup)  return false;
	if(materialSetup > rhs.materialSetup)  return true;
	if(numAttributes < rhs.numAttributes)  return true;
	if(numAttributes > rhs.numAttributes)  return false;
	if(numTextures < rhs.numTextures)  return true;
	if(numTextures > rhs.numTextures)  return false;
	if(numLights < rhs.numLights)  return true;
	if(numLights > rhs.numLights)  return false;
	if(attribSetup < rhs.attribSetup)  return true;
	if(attribSetup > rhs.attribSetup)  return false;
	if(idBuffer < rhs.idBuffer)  return false;
	if(idBuffer > rhs.idBuffer)  return true;
	for (size_t i = 0; i < numAttributes; ++i) {
		if(attribAccessInfo[i] < rhs.attribAccessInfo[i])  return true;
		if(attribAccessInfo[i] > rhs.attribAccessInfo[i])  return false;
	}
	for (size_t i = 0; i < numTextures; ++i) {
		if(textureSetup[i] < rhs.textureSetup[i])  return true;
		if(textureSetup[i] > rhs.textureSetup[i])  return false;
	}
	for (size_t i = 0; i < numLights; ++i) {
		if(lightSetup[i] < rhs.lightSetup[i])  return true;
		if(lightSetup[i] > rhs.lightSetup[i])  return false;
	}
	return false;
}