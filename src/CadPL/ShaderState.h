#pragma once

#include <array>
#include <bitset>

#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp> // shaderc_shader_kind

namespace CadPL {
	enum class AttributeType : uint16_t
	{
		undefined					= 0,
		vec4A16						/*= 0x01*/, // float4, alignment 16
		half4A8						/*= 0x02*/, // half4, alignment 8
		half4A4						/*= 0x03*/, // half4, alignment 4
		half4A4Offset2				/*= 0x04*/, // half4, alignment 4, reads the values with additional offset +2
		uint4A16Norm				/*= 0x05*/, // uint4 normalized, alignment 16
		uint4A16					/*= 0x06*/, // uint4, alignment 16
		int4A16Norm					/*= 0x07*/, // int4 normalized, alignment 16
		int4A16						/*= 0x08*/, // int4, alignment 16
		ushort4A8Norm				/*= 0x09*/, // ushort4 normalized, alignment 8
		ushort4A8				    /*= 0x0a*/, // ushort4, alignment 8
		ushort4A4Norm				/*= 0x0b*/, // ushort4 normalized, alignment 4
		ushort4A4				    /*= 0x0c*/, // ushort4, alignment 4
		ushort4A4NormOffset2		/*= 0x0d*/, // ushort4 normalized, alignment 4, reads the values with additional offset +2
		ushort4A4Offset2			/*= 0x0e*/, // ushort4, alignment 4, reads the values with additional offset +2
		short4A8Norm				/*= 0x0f*/, // short4 normalized, alignment 8
		short4A8					/*= 0x10*/, // short4, alignment 8
		short4A4Norm				/*= 0x11*/, // short4 normalized, alignment 4
		short4A4				    /*= 0x12*/, // short4, alignment 4
		short4A4NormOffset2			/*= 0x13*/, // short4 normalized, alignment 4, reads the values with additional offset +2
		short4A4Offset2		   	    /*= 0x14*/, // short4, alignment 4, reads the values with additional offset +2
		ubyte4A4Norm			    /*= 0x15*/, // "ubyte4 normalize, alignment 4"
		ubyte4A4				    /*= 0x16*/, // ubyte4, alignment 4
		ubyte4A4NormOffset2			/*= 0x17*/, // ubyte4 normalize, alignment 4, reads the values with additional offset +2
		ubyte4A4Offset2				/*= 0x18*/, // ubyte4, alignment 4, reads the values with additional offset +2
		byte4A4Norm					/*= 0x19*/, // byte4 normalize, alignment 4
		byte4A4						/*= 0x1a*/, // byte4, alignment 4
		byte4A4NormOffset2			/*= 0x1b*/, // byte4 normalize, alignment 4, reads the values with additional offset +2
		byte4A4Offset2				/*= 0x1c*/, // byte4, alignment 4, reads the values with additional offset +2
		vec3A16						/*= 0x20*/, // float3, alignment 16
		vec3A4						/*= 0x21*/, // float3, alignment 4
		half3A4					    /*= 0x22*/, // half3, alignment 4, on 8 bytes reads first six bytes
		half3A4Last6				/*= 0x23*/, // half3, alignment 4, on 8 bytes reads last six bytes
		uint3A16Norm	   		    /*= 0x24*/, // uint3, alignment 16, normalize
		uint3A16					/*= 0x25*/, // uint3, alignment 16
		uint3A4Norm					/*= 0x26*/, // uint3, alignment 4, normalize
		uint3A4						/*= 0x27*/, // uint3, alignment 4
		int3A16Norm					/*= 0x28*/, // int3, alignment 16, normalize
		int3A16						/*= 0x29*/, // int3, alignment 16
		int3A4Norm					/*= 0x2a*/, // int3, alignment 4, normalize
		int3A4						/*= 0x2b*/, // int3, alignment 4
		ushort3A4NormFirst6			/*= 0x2c*/, // ushort3, alignment 4, on 8 bytes reads first six bytes, normalize
		ushort3A4First6				/*= 0x2d*/, // ushort3, alignment 4, on 8 bytes reads first six bytes
		ushort3A4NormLast6			/*= 0x2e*/, // ushort3, alignment 4, on 8 bytes reads last six bytes, normalize
		ushort3A4Last6				/*= 0x2f*/, // ushort3, alignment 4, on 8 bytes reads last six bytes
		short3A4NormFirst6			/*= 0x30*/, // short3, alignment 4, on 8 bytes reads first six bytes, normalize
		short3A4First6				/*= 0x31*/, // short3, alignment 4, on 8 bytes reads first six bytes
		short3A4NormLast6			/*= 0x32*/, // short3, alignment 4, on 8 bytes reads last six bytes, normalize
		short3A4Last6				/*= 0x33*/, // short3, alignment 4, on 8 bytes reads last six bytes
		ubyte3A4NormFirst3			/*= 0x34*/, // ubyte3, alignment 4, on 4 bytes extracts first three bytes, normalize
		ubyte3A4First3				/*= 0x35*/, // ubyte3, alignment 4, on 4 bytes extracts first three bytes
		ubyte3A4NormLast3			/*= 0x36*/, // ubyte3, alignment 4, on 4 bytes extracts last three bytes, normalize
		ubyte3A4Last3				/*= 0x37*/, // ubyte3, alignment 4, on 4 bytes extracts last three bytes
		ubyte3A4NormFirst3Offset2	/*= 0x38*/, // ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize
		ubyte3A4First3Offset2		/*= 0x39*/, // ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2
		ubyte3A4NormLast3Offset2	/*= 0x3a*/, // ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize
		ubyte3A4Last3Offset2		/*= 0x3b*/, // ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2
		byte3A4NormFirst3			/*= 0x3c*/, // byte3, alignment 4, on 4 bytes extracts first three bytes, normalize
		byte3A4First3				/*= 0x3d*/, // byte3, alignment 4, on 4 bytes extracts first three bytes
		byte3A4NormLast3			/*= 0x3e*/, // byte3, alignment 4, on 4 bytes extracts last three bytes, normalize
		byte3A4Last3				/*= 0x3f*/, // byte3, alignment 4, on 4 bytes extracts last three bytes
		byte3A4NormFirst3Offset2	/*= 0x40*/, // byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize
		byte3A4First3Offset2		/*= 0x41*/, // byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2
		byte3A4NormLast3Offset2		/*= 0x42*/, // byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize
		byte3A4Last3Offset2			/*= 0x43*/, // byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2
		vec2A8						/*= 0x50*/, // float2, alignment 8
		vec2A4						/*= 0x51*/, // float2, alignment 4
		half2A4						/*= 0x52*/, // half2, alignment 4
		half2A4Offset2				/*= 0x53*/, // half2, alignment 4, reads the values with additional offset +2
		uint2A8Norm				 	/*= 0x54*/, // uint2, alignment 8, normalize
		uint2A8						/*= 0x55*/, // uint2, alignment 8
		uint2A4Norm					/*= 0x56*/, // uint2, alignment 4, normalize
		uint2A4						/*= 0x57*/, // uint2, alignment 4
		int2A8Norm					/*= 0x58*/, // int2, alignment 8, normalize
		int2A8						/*= 0x59*/, // int2, alignment 8
		int2A4Norm					/*= 0x5a*/, // int2, alignment 4, normalize
		int2A4						/*= 0x5b*/, // int2, alignment 4
		ushort2A4Norm				/*= 0x5c*/, // ushort2, alignment 4, normalize
		ushort2A4					/*= 0x5d*/, // ushort2, alignment 4
		ushort2A4NormOffset2		/*= 0x5e*/, // ushort2, alignment 4, reads the values with additional offset +2, normalize
		ushort2A4Offset2			/*= 0x5f*/, // ushort2, alignment 4, reads the values with additional offset +2
		short2A4Norm				/*= 0x60*/, // short2, alignment 4, normalize
		short2A4					/*= 0x61*/, // short2, alignment 4
		short2A4NormOffset2			/*= 0x62*/, // short2, alignment 4, reads the values with additional offset +2, normalize
		short2A4Offset2				/*= 0x63*/, // short2, alignment 4, reads the values with additional offset +2
		ubyte2A4Norm				/*= 0x64*/, // ubyte2, alignment 4, normalize
		ubyte2A4					/*= 0x65*/, // ubyte2, alignment 4
		ubyte2A4NormOffset1			/*= 0x66*/, // ubyte2, alignment 4, reads the values with additional offset +1, normalize
		ubyte2A4Offset1				/*= 0x67*/, // ubyte2, alignment 4, reads the values with additional offset +1
		ubyte2A4NormOffset2			/*= 0x68*/, // ubyte2, alignment 4, reads the values with additional offset +2, normalize
		ubyte2A4Offset2				/*= 0x69*/, // ubyte2, alignment 4, reads the values with additional offset +2
		ubyte2A4NormOffset3			/*= 0x6a*/, // ubyte2, alignment 4, reads the values with additional offset +3, normalize
		ubyte2A4Offset3				/*= 0x6b*/, // ubyte2, alignment 4, reads the values with additional offset +3
		byte2A4Norm					/*= 0x6c*/, // byte2, alignment 4, normalize
		byte2A4						/*= 0x6d*/, // byte2, alignment 4
		byte2A4NormOffset1			/*= 0x6e*/, // byte2, alignment 4, reads the values with additional offset +1, normalize
		byte2A4Offset1				/*= 0x6f*/, // byte2, alignment 4, reads the values with additional offset +1
		byte2A4NormOffset2			/*= 0x70*/, // byte2, alignment 4, reads the values with additional offset +2, normalize
		byte2A4Offset2				/*= 0x71*/, // byte2, alignment 4, reads the values with additional offset +2
		byte2A4NormOffset3			/*= 0x72*/, // byte2, alignment 4, reads the values with additional offset +3, normalize
		byte2A4Offset3				/*= 0x73*/, // byte2, alignment 4, reads the values with additional offset +3
		floatA8						/*= 0x80*/,  // float, alignment 4
		halfA4						/*= 0x81*/, // half, alignment 4
		halfA4Offset2				/*= 0x82*/, // half, alignment 4, reads the values with additional offset +2
		uintA4Norm					/*= 0x83*/, // uint, alignment 4, normalize
		uintA4						/*= 0x84*/, // uint, alignment 4
		intA4Norm					/*= 0x85*/, // int, alignment 4, normalize
		intA4						/*= 0x86*/, // int2, alignment 4
		ushortA4Norm				/*= 0x87*/, // ushort, alignment 4, normalize
		ushortA4					/*= 0x88*/, // ushort, alignment 4
		ushortA4NormOffset2			/*= 0x89*/, // ushort, alignment 4, reads the values with additional offset +2, normalize
		ushortA4Offset2				/*= 0x8a*/, // ushort, alignment 4, reads the values with additional offset +2
		shortA4Norm					/*= 0x8b*/, // short, alignment 4, normalize
		shortA4						/*= 0x8c*/, // short, alignment 4
		shortA4NormOffset2			/*= 0x8d*/, // short, alignment 4, reads the values with additional offset +2, normalize
		shortA4Offset2				/*= 0x8e*/, // short, alignment 4, reads the values with additional offset +2
		ubyteA4Norm					/*= 0x8f*/, // ubyte, alignment 4, normalize
		ubyteA4						/*= 0x90*/, // ubyte, alignment 4
		ubyteA4NormOffset1			/*= 0x91*/, // ubyte, alignment 4, reads the values with additional offset +1, normalize
		ubyteA4Offset1				/*= 0x92*/, // ubyte, alignment 4, reads the values with additional offset +1
		ubyteA4NormOffset2			/*= 0x93*/, // ubyte, alignment 4, reads the values with additional offset +2, normalize
		ubyteA4Offset2				/*= 0x94*/, // ubyte, alignment 4, reads the values with additional offset +2
		ubyteA4NormOffset3			/*= 0x95*/, // ubyte, alignment 4, reads the values with additional offset +3, normalize
		ubyteA4Offset3				/*= 0x96*/, // ubyte, alignment 4, reads the values with additional offset +3
		byteA4Norm					/*= 0x97*/, // byte, alignment 4, normalize
		byteA4						/*= 0x98*/, // byte, alignment 4
		byteA4NormOffset1			/*= 0x99*/, // byte, alignment 4, reads the values with additional offset +1, normalize
		byteA4Offset1				/*= 0x9a*/, // byte, alignment 4, reads the values with additional offset +1
		byteA4NormOffset2			/*= 0x9b*/, // byte, alignment 4, reads the values with additional offset +2, normalize
		byteA4Offset2				/*= 0x9c*/, // byte, alignment 4, reads the values with additional offset +2
		byteA4NormOffset3			/*= 0x9d*/, // byte, alignment 4, reads the values with additional offset +3, normalize
		byteA4Offset3				/*= 0x9e*/, // byte, alignment 4, reads the values with additional offset +3
	};

	enum class TextureType
	{
		none = 0,
		normal,
		occlusion,
		base,
		emissive,
		metallicRoughness,
		sheen,
		sheenRoughness
	};
	static constexpr auto MaxTextures = 8;

	// fixed position attributes
	static constexpr auto PositionAttributeIndex = 0;
	static constexpr auto NormalAttributeIndex = 1;
	static constexpr auto TangentAttributeIndex = 2;
	static constexpr auto ColorAttributeIndex = 3;

	struct CADPL_EXPORT ShaderState {

		static constexpr const unsigned numOptimizeFlags = 7;
		static constexpr const std::bitset<numOptimizeFlags> OptimizeNone = 0x00;  //< No optimizations. Uber-shader will be used.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeAttribs = 0x01;  //< Optimize attribute access. Number of attributes, their indices, their type and data offset are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeMaterialModel = 0x02;  //< Optimize material model (unlit, phong, metallic-roughness,...). The material model is fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeMaterialColorAttribute = 0x04;  //< Optimize phong color attribute settings (color to diffuse, color to ambient and diffuse). The settings are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeMaterialAlpha = 0x08;  //< Optimize alpha computation (ignore texture alpha, ignore material alpha, ignore color attribute alpha). The alpha flags are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeMaterial = 0x0e;  //< Optimize all material related settings. The settings are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeTextureTypesAndTexCoordIndices = 0x10;  //< Optimize texture types and attribute indices from which texture coordinates are sourced. Number of textures, their types (normal texture, occlusion texture, emissive texture, base texture,...) and attribute indices for sourcing texture coordinates are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeTextureFlags = 0x20;  //< Optimize texture flags (apply strength, apply texture coordinate transformation, blend color included, phong's texture environment, first component index). The settings are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeTextures = 0x30;  //< Optimize all texture related settings. The settings are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeLightTypes = 0x40;  //< Optimize light types. Number of lights and their types are fixed and hardcoded into the shader code.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeLights = 0x40;  //< Optimize all light related settings.
		static constexpr const std::bitset<numOptimizeFlags> OptimizeAll = 0x7f;  //< Make all available optimizations.

		std::bitset<numOptimizeFlags> optimizeFlags = OptimizeNone; //  size: 8
		enum class ProjectionHandling { SceneMatrix, PerspectivePushAndSpecializationConstants };
		ProjectionHandling projectionHandling = ProjectionHandling::SceneMatrix;
		vk::PrimitiveTopology primitiveTopology;
		uint32_t materialSetup;
		uint8_t numAttributes;
		uint8_t numTextures;
		uint16_t numLights;
		static constexpr const unsigned maxNumAttribs = 16;
		std::array<uint16_t,maxNumAttribs> attribAccessInfo;
		std::array<uint32_t,MaxTextures> textureSetup;
		std::array<uint16_t, 4> lightSetup;
		uint16_t attribSetup;
		bool idBuffer;

		bool operator<(const ShaderState& rhs) const;

		void set(const ShaderState &state, std::bitset<numOptimizeFlags> optimizeFlags);

		std::string serialize() const;
		void deserialize(const std::string &serializedState);

		std::string debugDump() const;

		static std::string optimizeFlagsToString(std::bitset<numOptimizeFlags> optimizeFlags);
	};

	// modifications to ShaderState must propagate here
	struct VertexShaderState {
		static constexpr vk::ShaderStageFlagBits ShaderStage = vk::ShaderStageFlagBits::eVertex;
		static constexpr shaderc_shader_kind ShaderKind = shaderc_vertex_shader;

		std::bitset<ShaderState::numOptimizeFlags> optimizeFlags;
		vk::PrimitiveTopology primitiveTopology;
		ShaderState::ProjectionHandling projectionHandling;
		uint8_t numAttributes;
		bool idBuffer;
		uint16_t attribSetup;
		std::array<uint16_t, ShaderState::maxNumAttribs> attribAccessInfo;

		VertexShaderState(const ShaderState& shaderState);
		VertexShaderState(const std::string& serializedState);
		bool operator<(const VertexShaderState& rhs) const;
		std::string serialize() const;
		void deserialize(const std::string &serializedState);
	};
	struct GeometryShaderState {
		static constexpr vk::ShaderStageFlagBits ShaderStage = vk::ShaderStageFlagBits::eGeometry;
		static constexpr shaderc_shader_kind ShaderKind = shaderc_geometry_shader;

		std::bitset<ShaderState::numOptimizeFlags> optimizeFlags;
		vk::PrimitiveTopology primitiveTopology;
		ShaderState::ProjectionHandling projectionHandling;
		uint8_t numAttributes;
		bool idBuffer;
		uint16_t attribSetup;
		std::array<uint16_t, ShaderState::maxNumAttribs> attribAccessInfo;

		GeometryShaderState(const ShaderState& shaderState);
		GeometryShaderState(const std::string &serializedState);
		bool operator<(const GeometryShaderState& rhs) const;
		std::string serialize() const;
		void deserialize(const std::string &serializedState);
	};
	struct FragmentShaderState {
		static constexpr vk::ShaderStageFlagBits ShaderStage = vk::ShaderStageFlagBits::eFragment;
		static constexpr shaderc_shader_kind ShaderKind = shaderc_fragment_shader;

		std::bitset<ShaderState::numOptimizeFlags> optimizeFlags;
		uint32_t materialSetup;
		uint8_t numAttributes;
		uint8_t numTextures;
		uint16_t numLights;


		static constexpr const unsigned maxNumAttribs = 16;
		uint16_t attribSetup;
		std::array<uint16_t,maxNumAttribs> attribAccessInfo;
		std::array<uint32_t,MaxTextures> textureSetup;
		std::array<uint16_t, 4> lightSetup;
		bool idBuffer;

		FragmentShaderState(const ShaderState& shaderState);
		FragmentShaderState(const std::string& serializedState);
		bool operator<(const FragmentShaderState& rhs) const;
		std::string serialize() const;
		void deserialize(const std::string &serializedState);
	};

}