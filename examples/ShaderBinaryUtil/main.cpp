#include <filesystem>
#include <span>
#include <iostream>
#include <fstream>
#include <variant>

#include <shaderc/shaderc.hpp>

#include <CadR/VulkanDevice.h>
#include <CadR/VulkanInstance.h>
#include <CadR/VulkanLibrary.h>
#include <CadR/Renderer.h>


typedef VkResult (VKAPI_PTR *PFN_vkCreateShadersEXT)(VkDevice device, uint32_t createInfoCount, const VkShaderCreateInfoEXT* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkShaderEXT* pShaders);
typedef void (VKAPI_PTR *PFN_vkDestroyShaderEXT)(VkDevice device, VkShaderEXT shader, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkGetShaderBinaryDataEXT)(VkDevice device, VkShaderEXT shader, size_t* pDataSize, void* pData);

PFN_vkCreateShadersEXT createShadersExt = nullptr;
PFN_vkDestroyShaderEXT destroyShaderEXT = nullptr;
PFN_vkGetShaderBinaryDataEXT getShaderBinaryDataEXT = nullptr;

CadR::VulkanLibrary vulkanLib;
CadR::VulkanInstance vulkanInstance;
vk::PhysicalDevice physicalDevice;
CadR::VulkanDevice device;

std::vector<unsigned char> readFile(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Can't open file " + std::string(filename));
    }
    file.unsetf(std::ios::skipws);

    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<unsigned char> vec;
    vec.reserve(fileSize);

    vec.insert(vec.begin(),
               std::istream_iterator<unsigned char>(file),
               std::istream_iterator<unsigned char>());

    return vec;
}

std::string readFileAsString(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Can't open file " + std::string(filename));
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


// Compiles a shader to SPIR-V assembly. Returns the assembly text
// as a string.
std::string compileFileToAssembly(const std::string& name,
                                     shaderc_shader_kind kind,
                                     const std::string& source,
                                     bool optimize = false) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::AssemblyCompilationResult result = compiler.CompileGlslToSpvAssembly(
            source, kind, name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << result.GetErrorMessage();
        throw std::runtime_error("Can't compile file: " + std::string(name));
    }

    return {result.cbegin(), result.cend()};
}

struct CompileOutput {
    std::vector<uint32_t> binary;
    std::string assembly;
};

CompileOutput compileFile(const std::string& name,
                          shaderc_shader_kind kind,
                          const std::string& source,
                          bool optimize,
                          bool assembly,
                          bool printTime)
{

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    CompileOutput output;

    // Like -DMY_DEFINE=1
    // options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto start = std::chrono::system_clock::now();
    shaderc::SpvCompilationResult module =
            compiler.CompileGlslToSpv(source, kind, name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << module.GetErrorMessage();
        throw std::runtime_error("Can't compile file: " + std::string(name));
        // return std::vector<uint32_t>();
    }
    if (printTime) {
        auto end = std::chrono::system_clock::now();
        auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << name << " took " << elapsed.count() << "ms\n";
    }
    if (assembly) {
        output.assembly = compileFileToAssembly(name, kind, source);
    }
    output.binary = {module.cbegin(), module.cend()};

    return output;
}

std::pair<CompileOutput, vk::ShaderStageFlagBits> compileFile(const std::filesystem::path& file,
                                                              const std::string& name,
                                                              bool optimize,
                                                              bool assembly,
                                                              bool printTime) {
    const auto &ext = file.extension();
    shaderc_shader_kind kind;
    vk::ShaderStageFlagBits shaderStage;
    if (ext == ".frag") {
        kind = shaderc_fragment_shader;
        shaderStage = vk::ShaderStageFlagBits::eFragment;
    }
    else if (ext == ".vert") {
        kind = shaderc_vertex_shader;
        shaderStage = vk::ShaderStageFlagBits::eVertex;
    }
    else {
        std::cerr << "Unknown extension: " << ext << "(" << file << ")\n";
        throw std::runtime_error("Can't compile file");
    }

    const auto source = readFileAsString(file.string().c_str());
    return std::make_pair(compileFile(name, kind, source, optimize, assembly, printTime), shaderStage);
}

void dumpSpirvShader(const vk::ShaderCreateInfoEXT &info, const std::string& outputFile) {
    VkShaderEXT shader;
    auto result = createShadersExt(device.get(), 1, reinterpret_cast<const VkShaderCreateInfoEXT*>(&info), nullptr, &shader);
    if (result != VK_SUCCESS) {
        std::cerr << vk::to_string((vk::Result) result) << "\n";
        throw std::runtime_error("createShadersExt failed");
    }

    size_t dataSize;
    result = getShaderBinaryDataEXT(device.get(), shader, &dataSize, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("getShaderBinaryDataEXT failed");
    }
    std::vector<unsigned char> data;
    data.resize(dataSize);
    result = getShaderBinaryDataEXT(device.get(), shader, &dataSize, data.data());
    if (result != VK_SUCCESS) {
        throw std::runtime_error("getShaderBinaryDataEXT failed");
    }
    std::cout << outputFile << ", source spirv: " << info.codeSize << "B, shader: " << data.size() << "B\n";

    std::ofstream fs(outputFile, std::ios::out | std::ios::binary | std::ios::app);
    fs.write((const char*)data.data(), data.size());
    fs.close();

    destroyShaderEXT(device.get(), shader, nullptr);
}

struct ShaderInfo {
    std::string sourceFile;
    std::string code;
    shaderc_shader_kind kind = {};
    const vk::SpecializationInfo *specializationInfo = {};
    bool optimizedVariant = true;
    bool nonoptimizedVariant = false;
    bool printTime = true;
    bool printAssembly = false;
    bool saveAssembly = true;
    bool saveSpirv = true;
};

vk::ShaderStageFlagBits shaderKindToStage(shaderc_shader_kind kind) {
    switch (kind) {
        case shaderc_fragment_shader:
            return vk::ShaderStageFlagBits::eFragment;
        case shaderc_vertex_shader:
            return vk::ShaderStageFlagBits::eVertex;
        default:
            return {};
    }
}

void createShader(const ShaderInfo &info, const std::filesystem::path& outputDir, bool optimize) {
    const auto file = std::filesystem::path(info.sourceFile);
    const auto name = file.filename().string();
    const bool assembly = info.printAssembly || info.saveAssembly;

    CompileOutput spirv;
    vk::ShaderStageFlagBits stage;
    if (info.code.empty()) {
        std::tie(spirv, stage) = compileFile(file, name, optimize, assembly, info.printTime);
    }
    else {
        spirv = compileFile(name, info.kind, info.code, optimize, assembly, info.printTime);
        stage = shaderKindToStage(info.kind);
    }

    auto outFile = (outputDir / name).string();
    if (optimize) {
        outFile += "_opt";
    }
    if (info.saveAssembly) {
        // std::cout << "SPIRV: \n" << assembly << "\n";}
        std::ofstream fs(outFile + ".txt", std::ios::out | std::ios::binary | std::ios::app);
        fs.write((const char*)spirv.assembly.data(), spirv.assembly.size());
        fs.close();
    }
    if (info.saveSpirv) {
        std::ofstream fs(outFile + ".spirv", std::ios::out | std::ios::binary | std::ios::app);
        fs.write((const char*)spirv.binary.data(), spirv.binary.size() * 4);
        fs.close();
    }

    vk::ShaderCreateInfoEXT createInfo = {};
    createInfo.stage = stage;
    createInfo.codeType = vk::ShaderCodeTypeEXT::eSpirv;
    createInfo.codeSize = spirv.binary.size() * 4;
    createInfo.pCode = spirv.binary.data();
    createInfo.pName = "main";
    createInfo.pSpecializationInfo = info.specializationInfo;

    dumpSpirvShader(createInfo, outFile + ".bin");
}

void createShader(const ShaderInfo &info, const std::filesystem::path& outputDir) {
    if (info.nonoptimizedVariant) {
        createShader(info, outputDir, false);
    }
    if (info.optimizedVariant) {
        createShader(info, outputDir, true);
    }
}

void createShaders(const std::span<const char*> sources, const std::filesystem::path outputDir) {
    std::filesystem::create_directory(outputDir);

    for (const auto &s : sources) {
        ShaderInfo info;
        info.sourceFile = s;
        createShader(info, outputDir);
    }
}

void createShaders(const std::span<ShaderInfo> sources, const std::filesystem::path outputDir) {
    std::filesystem::create_directory(outputDir);
    for (const auto &s : sources) {
        createShader(s, outputDir);
    }
}

int main(int argc, const char* argv[]) {

    if (argc < 1) {
        std::cout << "no sources specified\n";
        return 0;
    }

    const std::filesystem::path outputDir = "output";

    try {

        vulkanLib.load(CadR::VulkanLibrary::defaultName());
        vulkanInstance.create(vulkanLib, "shader util", 0, "CADR", 0, VK_API_VERSION_1_2, {}, {});
        // init device and renderer
        std::tuple<vk::PhysicalDevice, uint32_t, uint32_t> deviceAndQueueFamilies =
                vulkanInstance.chooseDevice(
                        vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,  // queueOperations
                        VK_NULL_HANDLE, // window.surface(),  // presentationSurface
                        [](CadR::VulkanInstance& instance, vk::PhysicalDevice pd) -> bool {  // filterCallback
                            if(instance.getPhysicalDeviceProperties(pd).apiVersion < VK_API_VERSION_1_2)
                                return false;
                            auto features =
                                    instance.getPhysicalDeviceFeatures2<
                                            vk::PhysicalDeviceFeatures2,
                                            vk::PhysicalDeviceVulkan11Features,
                                            vk::PhysicalDeviceVulkan12Features>(pd);
                            return
                                    features.get<vk::PhysicalDeviceFeatures2>().features.multiDrawIndirect &&
                                    features.get<vk::PhysicalDeviceFeatures2>().features.shaderInt64 &&
                                    features.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                    features.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress;
                        });

        physicalDevice = std::get<0>(deviceAndQueueFamilies);
        if(!physicalDevice)
            throw std::runtime_error("No compatible Vulkan device found.");

        device.create(
                vulkanInstance, deviceAndQueueFamilies,
#if 1 // enable or disable validation extensions
                // (0 enables validation extensions and features for debugging purposes)
                {"VK_KHR_swapchain", "VK_EXT_shader_object", "VK_KHR_dynamic_rendering"},
                []() {
                    CadR::Renderer::RequiredFeaturesStructChain f = CadR::Renderer::requiredFeaturesStructChain();
                    f.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy = true;
                    return f;
                }().get<vk::PhysicalDeviceFeatures2>()
#else
                {"VK_KHR_swapchain", "VK_KHR_shader_non_semantic_info"},
		[]() {
			CadR::Renderer::RequiredFeaturesStructChain f = CadR::Renderer::requiredFeaturesStructChain();
			f.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy = true;
			f.get<vk::PhysicalDeviceVulkan12Features>().uniformAndStorageBuffer8BitAccess = true;
			return f;
		}().get<vk::PhysicalDeviceFeatures2>()
#endif
        );

        ::createShadersExt = (PFN_vkCreateShadersEXT) device.getProcAddr("vkCreateShadersEXT");
        if (!createShadersExt) {
            throw std::runtime_error("Unsupported VkDevice functionality");
        }
        ::destroyShaderEXT = (PFN_vkDestroyShaderEXT) device.getProcAddr("vkDestroyShaderEXT");
        if (!destroyShaderEXT) {
            throw std::runtime_error("Unsupported VkDevice functionality");
        }
        ::getShaderBinaryDataEXT = (PFN_vkGetShaderBinaryDataEXT) device.getProcAddr("vkGetShaderBinaryDataEXT");
        if (!getShaderBinaryDataEXT) {
            throw std::runtime_error("Unsupported VkDevice functionality");
        }

        // createShaders(std::span<const char *>(argv + 1, argc - 1), outputDir);

        const std::array<uint32_t,1> specializationConstants = {
            0x2000
        };
        const std::array specializationMap {
            vk::SpecializationMapEntry{0,0,4},  // constantID, offset, size
        };
        const vk::SpecializationInfo spec1(  // pSpecializationInfo
            specializationMap.size(),  // mapEntryCount
            specializationMap.data(),  // pMapEntries
            specializationConstants.size() * sizeof(int),  // dataSize
            specializationConstants.data()  // pData
        );


        std::string interface = std::string(R"(
#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

//
//  buffer references
//

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
AlignedVec4Ref {
	vec4 value;
};

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
AlignedUVec4Ref {
	uvec4 value;
};

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
AlignedIVec4Ref {
	ivec4 value;
};

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
AlignedVec3Ref {
	vec3 value;
};

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
AlignedUVec3Ref {
	uvec3 value;
};

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
AlignedIVec3Ref {
	ivec3 value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
UnalignedVec3Ref {
	vec3 value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
UnalignedUVec3Ref {
	uvec3 value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
UnalignedIVec3Ref {
	ivec3 value;
};

layout(buffer_reference, std430, buffer_reference_align=8) restrict readonly buffer
AlignedVec2Ref {
	vec2 value;
};

layout(buffer_reference, std430, buffer_reference_align=8) restrict readonly buffer
AlignedUVec2Ref {
	uvec2 value;
};

layout(buffer_reference, std430, buffer_reference_align=8) restrict readonly buffer
AlignedIVec2Ref {
	ivec2 value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
UnalignedVec2Ref {
	vec2 value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
UnalignedUVec2Ref {
	uvec2 value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
UnalignedIVec2Ref {
	ivec2 value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
AlignedFloatRef {
	float value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
AlignedUIntRef {
	uint value;
};

layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
AlignedIntRef {
	int value;
};

)") + std::string(R"(

//
//  read float
//
float readFloat(uint64_t vertexDataPtr, uint settings)
{
	// settings:
	// bits 0..7 - offset (0..255)
	// bits 8..15 - type
	//
	// type values:
	//   - 0x80 - float, alignment 4
	//   - 0x81 - half, alignment 4
	//   - 0x82 - half, alignment 4, reads the values with additional offset +2
	//   - 0x83 - uint, alignment 4, normalize
	//   - 0x84 - uint, alignment 4
	//   - 0x85 - int2, alignment 4, normalize
	//   - 0x86 - int2, alignment 4
	//   - 0x87 - ushort, alignment 4, normalize
	//   - 0x88 - ushort, alignment 4
	//   - 0x89 - ushort, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x8a - ushort, alignment 4, reads the values with additional offset +2
	//   - 0x8b - short, alignment 4, normalize
	//   - 0x8c - short, alignment 4
	//   - 0x8d - short, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x8e - short, alignment 4, reads the values with additional offset +2
	//   - 0x8f - ubyte, alignment 4, normalize
	//   - 0x90 - ubyte, alignment 4
	//   - 0x91 - ubyte, alignment 4, reads the values with additional offset +1, normalize
	//   - 0x92 - ubyte, alignment 4, reads the values with additional offset +1
	//   - 0x93 - ubyte, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x94 - ubyte, alignment 4, reads the values with additional offset +2
	//   - 0x95 - ubyte, alignment 4, reads the values with additional offset +3, normalize
	//   - 0x96 - ubyte, alignment 4, reads the values with additional offset +3
	//   - 0x97 - byte, alignment 4, normalize
	//   - 0x98 - byte, alignment 4
	//   - 0x99 - byte, alignment 4, reads the values with additional offset +1, normalize
	//   - 0x9a - byte, alignment 4, reads the values with additional offset +1
	//   - 0x9b - byte, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x9c - byte, alignment 4, reads the values with additional offset +2
	//   - 0x9d - byte, alignment 4, reads the values with additional offset +3, normalize
	//   - 0x9e - byte, alignment 4, reads the values with additional offset +3

	uint offset = settings & 0x00ff;  // max offset is 255
	uint64_t addr = vertexDataPtr + offset;
	uint type = settings & 0xff00;

	// float
	if(type == 0x8000)
		return AlignedFloatRef(addr).value;

	// half
	if(type == 0x8100) {
		uint v = AlignedUIntRef(addr).value;
		return unpackHalf2x16(v).x;
	}
	if(type == 0x8200) {
		uint v = AlignedUIntRef(addr).value;
		return unpackHalf2x16(v).y;
	}

	// uint
	if(type == 0x8300) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return float(v) / 0xffffffff;
	}
	if(type == 0x8400)
		// alignment 4, do not normalize
		return AlignedUIntRef(addr).value;

	// int
	if(type == 0x8500) {
		// alignment 4, normalize
		int v = AlignedIntRef(addr).value;
		return max(float(v) / 0x7fffffff, -1.);
	}
	if(type == 0x8600)
		// alignment 4, do not normalize
		return AlignedIntRef(addr).value;

	// ushort
	if(type == 0x8700) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm2x16(v).x;
	}
	if(type == 0x8800) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return float(v & 0xffff);
	}
	if(type == 0x8900) {
		// alignment 4, offset +2, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm2x16(v).y;
	}
	if(type == 0x8a00) {
		// alignment 4, offset +2, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return float(v >> 16);
	}

	// short
	if(type == 0x8b00) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm2x16(v).x;
	}
	if(type == 0x8c00) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		int r = int(v & 0xffff);
		r |= 0xffff0000 * (r >> 15);
		return r;
	}
	if(type == 0x8d00) {
		// alignment 4, offset +2, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm2x16(v).y;
	}
	if(type == 0x8e00) {
		// alignment 4, offset +2, do not normalize
		uint v = AlignedUIntRef(addr).value;
		int r = int(v >> 16);
		r |= 0xffff0000 * (r >> 15);
		return r;
	}

	// ubyte
	if(type == 0x8f00) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).x;
	}
	if(type == 0x9000) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return float(v & 0xff);
	}
	if(type == 0x9100) {
		// alignment 4, offset +1, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).y;
	}
	if(type == 0x9200) {
		// alignment 4, offset +1 do not normalize
		uint v = AlignedUIntRef(addr).value;
		return float((v >> 8) & 0xff);
	}
	if(type == 0x9300) {
		// alignment 4, offset +2, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).z;
	}
	if(type == 0x9400) {
		// alignment 4, offset +2 do not normalize
		uint v = AlignedUIntRef(addr).value;
		return float((v >> 16) & 0xff);
	}
	if(type == 0x9500) {
		// alignment 4, offset +3, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).w;
	}
	if(type == 0x9600) {
		// alignment 4, offset +3, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return float(v >> 24);
	}

	// byte
	if(type == 0x9700) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm4x8(v).x;
	}
	if(type == 0x9800) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		int r = int(v & 0xff);
		r |= 0xffffff00 * (r >> 7);
		return float(r);
	}
	if(type == 0x9900) {
		// alignment 4, offset +1, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).y;
	}
	if(type == 0x9a00) {
		// alignment 4, offset +1 do not normalize
		uint v = AlignedUIntRef(addr).value;
		int r = int((v >> 8) & 0xff);
		r |= 0xffffff00 * (r >> 7);
		return float(r);
	}
	if(type == 0x9b00) {
		// alignment 4, offset +2, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).z;
	}
	if(type == 0x9c00) {
		// alignment 4, offset +2 do not normalize
		uint v = AlignedUIntRef(addr).value;
		int r = int((v >> 16) & 0xff);
		r |= 0xffffff00 * (r >> 7);
		return float(r);
	}
	if(type == 0x9d00) {
		// alignment 4, offset +3, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).w;
	}
	if(type == 0x9e00) {
		// alignment 4, offset +3, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return float(v >> 24);
	}

	// return NaN
	return float(0/0);
}

)") + std::string(R"(

//
//  read vec2
//
vec2 readVec2(uint64_t vertexDataPtr, uint settings)
{
	// settings:
	// bits 0..7 - offset (0..255)
	// bits 8..15 - type
	//
	// type values:
	//   - 0x50 - float2, alignment 8
	//   - 0x51 - float2, alignment 4
	//   - 0x52 - half2, alignment 4
	//   - 0x53 - half2, alignment 4, reads the values with additional offset +2
	//   - 0x54 - uint2, alignment 8, normalize
	//   - 0x55 - uint2, alignment 8
	//   - 0x56 - uint2, alignment 4, normalize
	//   - 0x57 - uint2, alignment 4
	//   - 0x58 - int2, alignment 8, normalize
	//   - 0x58 - int2, alignment 8
	//   - 0x5a - int2, alignment 4, normalize
	//   - 0x5b - int2, alignment 4
	//   - 0x5c - ushort2, alignment 4, normalize
	//   - 0x5d - ushort2, alignment 4
	//   - 0x5e - ushort2, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x5f - ushort2, alignment 4, reads the values with additional offset +2
	//   - 0x60 - short2, alignment 4, normalize
	//   - 0x61 - short2, alignment 4
	//   - 0x62 - short2, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x63 - short2, alignment 4, reads the values with additional offset +2
	//   - 0x64 - ubyte2, alignment 4, normalize
	//   - 0x65 - ubyte2, alignment 4
	//   - 0x66 - ubyte2, alignment 4, reads the values with additional offset +1, normalize
	//   - 0x67 - ubyte2, alignment 4, reads the values with additional offset +1
	//   - 0x68 - ubyte2, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x69 - ubyte2, alignment 4, reads the values with additional offset +2
	//   - 0x6a - ubyte2, alignment 4, reads the values with additional offset +3, normalize
	//   - 0x6b - ubyte2, alignment 4, reads the values with additional offset +3
	//   - 0x6c - byte2, alignment 4, normalize
	//   - 0x6d - byte2, alignment 4
	//   - 0x6e - byte2, alignment 4, reads the values with additional offset +1, normalize
	//   - 0x6f - byte2, alignment 4, reads the values with additional offset +1
	//   - 0x70 - byte2, alignment 4, reads the values with additional offset +2, normalize
	//   - 0x71 - byte2, alignment 4, reads the values with additional offset +2
	//   - 0x72 - byte2, alignment 4, reads the values with additional offset +3, normalize
	//   - 0x73 - byte2, alignment 4, reads the values with additional offset +3

	uint offset = settings & 0x00ff;  // max offset is 255
	uint64_t addr = vertexDataPtr + offset;
	uint type = settings & 0xff00;

	// float2
	if(type == 0x5000)
		return AlignedVec2Ref(addr).value;
	if(type == 0x5100)
		return UnalignedVec2Ref(addr).value;

	// half2
	if(type == 0x5200) {
		uint v = AlignedUIntRef(addr).value;
		return unpackHalf2x16(v);
	}
	if(type == 0x5300) {
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackHalf2x16(v[0]);
	}

	// uint2
	if(type == 0x5400) {
		// alignment 8, normalize
		uvec2 v = AlignedUVec2Ref(addr).value;
		return vec2(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff);
	}
	if(type == 0x5500)
		// alignment 8, do not normalize
		return AlignedUVec2Ref(addr).value;
	if(type == 0x5600) {
		// alignment 4, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec2(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff);
	}
	if(type == 0x5700)
		// alignment 4, do not normalize
		return UnalignedUVec2Ref(addr).value;

	// int2
	if(type == 0x5800) {
		// alignment 8, normalize
		ivec2 v = AlignedIVec2Ref(addr).value;
		return max(vec2(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff),
		           -1.);
	}
	if(type == 0x5900)
		// alignment 8, do not normalize
		return AlignedIVec2Ref(addr).value;
	if(type == 0x5a00) {
		// alignment 4, normalize
		ivec2 v = UnalignedIVec2Ref(addr).value;
		return max(vec2(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff),
		           -1.);
	}
	if(type == 0x5b00)
		// alignment 4, do not normalize
		return UnalignedIVec2Ref(addr).value;

	// ushort2
	if(type == 0x5c00) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm2x16(v);
	}
	if(type == 0x5d00) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return vec2(v & 0xffff, v >> 16);
	}
	if(type == 0x5e00) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackUnorm2x16(v[0]);
	}
	if(type == 0x5f00) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec2(v[0] >> 16, v[1] & 0xffff);
	}

	// short2
	if(type == 0x6000) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm2x16(v);
	}
	if(type == 0x6100) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		ivec2 r = ivec2(int(v & 0xffff), int(v >> 16));
		r |= 0xffff0000 * (r >> 15);
		return vec2(r);
	}
	if(type == 0x6200) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return unpackSnorm2x16(v[0]);
	}
	if(type == 0x6300) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		ivec2 r = ivec2(int(v[0] >> 16), int(v[1] & 0xffff));
		r |= 0xffff0000 * (r >> 15);
		return vec2(r);
	}

	// ubyte
	if(type == 0x6400) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).xy;
	}
	if(type == 0x6500) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return vec2(v & 0xff, (v >> 8) & 0xff);
	}
	if(type == 0x6600) {
		// alignment 4, offset +1, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).yz;
	}
	if(type == 0x6700) {
		// alignment 4, offset +1 do not normalize
		uint v = AlignedUIntRef(addr).value;
		return vec2((v >> 8) & 0xff, (v >> 16) & 0xff);
	}
	if(type == 0x6800) {
		// alignment 4, offset +2, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).zw;
	}
	if(type == 0x6900) {
		// alignment 4, offset +2 do not normalize
		uint v = AlignedUIntRef(addr).value;
		return vec2((v >> 16) & 0xff, (v >> 24) & 0xff);
	}
	if(type == 0x6a00) {
		// alignment 4, offset +3, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 24) | ((v[1] & 0xff) << 8);
		return unpackUnorm4x8(v[0]).xy;
	}
	if(type == 0x6b00) {
		// alignment 4, offset +3, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec2(v[0] >> 24, v[1] & 0xff);
	}

	// byte
	if(type == 0x6c00) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm4x8(v).xy;
	}
	if(type == 0x6d00) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		ivec2 r = ivec2(int(v & 0xff), int((v >> 8) & 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec2(r);
	}
	if(type == 0x6e00) {
		// alignment 4, offset +1, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).yz;
	}
	if(type == 0x6f00) {
		// alignment 4, offset +1 do not normalize
		uint v = AlignedUIntRef(addr).value;
		ivec2 r = ivec2(int((v >> 8) & 0xff), int((v >> 16) & 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec2(r);
	}
	if(type == 0x7000) {
		// alignment 4, offset +2, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).zw;
	}
	if(type == 0x7100) {
		// alignment 4, offset +2 do not normalize
		uint v = AlignedUIntRef(addr).value;
		ivec2 r = ivec2(int((v >> 16) & 0xff), int((v >> 24) & 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec2(r);
	}
	if(type == 0x7200) {
		// alignment 4, offset +3, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 24) | ((v[1] & 0xff) << 8);
		return unpackUnorm4x8(v[0]).xy;
	}
	if(type == 0x7300) {
		// alignment 4, offset +3, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec2(v[0] >> 24, (v[1] & 0xff) << 8);
	}

	// return NaN
	return vec2(0/0);
}

)") + std::string(R"(

vec3 readAlignedVec3(uint64_t address) {
    return AlignedVec3Ref(address).value;
}

//
//  read vec3
//
vec3 readVec3(uint64_t vertexDataPtr, uint settings)
{
	// settings:
	// bits 0..7 - offset (0..255)
	// bits 8..15 - type
	//
	// type values:
	//   - 0x20 - float3, alignment 16
	//   - 0x21 - float3, alignment 4
	//   - 0x22 - half3, alignment 4, on 8 bytes reads first six bytes
	//   - 0x23 - half3, alignment 4, on 8 bytes reads last six bytes
	//   - 0x24 - uint3, alignment 16, normalize
	//   - 0x25 - uint3, alignment 16
	//   - 0x26 - uint3, alignment 4, normalize
	//   - 0x27 - uint3, alignment 4
	//   - 0x28 - int3, alignment 16, normalize
	//   - 0x29 - int3, alignment 16
	//   - 0x2a - int3, alignment 4, normalize
	//   - 0x2b - int3, alignment 4
	//   - 0x2c - ushort3, alignment 4, on 8 bytes reads first six bytes, normalize
	//   - 0x2d - ushort3, alignment 4, on 8 bytes reads first six bytes
	//   - 0x2e - ushort3, alignment 4, on 8 bytes reads last six bytes, normalize
	//   - 0x2f - ushort3, alignment 4, on 8 bytes reads last six bytes
	//   - 0x30 - short3, alignment 4, on 8 bytes reads first six bytes, normalize
	//   - 0x31 - short3, alignment 4, on 8 bytes reads first six bytes
	//   - 0x32 - short3, alignment 4, on 8 bytes reads last six bytes, normalize
	//   - 0x33 - short3, alignment 4, on 8 bytes reads last six bytes
	//   - 0x34 - ubyte3, alignment 4, on 4 bytes extracts first three bytes, normalize
	//   - 0x35 - ubyte3, alignment 4, on 4 bytes extracts first three bytes
	//   - 0x36 - ubyte3, alignment 4, on 4 bytes extracts last three bytes, normalize
	//   - 0x37 - ubyte3, alignment 4, on 4 bytes extracts last three bytes
	//   - 0x38 - ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize
	//   - 0x39 - ubyte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2
	//   - 0x3a - ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize
	//   - 0x3b - ubyte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2
	//   - 0x3c - byte3, alignment 4, on 4 bytes extracts first three bytes, normalize
	//   - 0x3d - byte3, alignment 4, on 4 bytes extracts first three bytes
	//   - 0x3e - byte3, alignment 4, on 4 bytes extracts last three bytes, normalize
	//   - 0x3f - byte3, alignment 4, on 4 bytes extracts last three bytes
	//   - 0x40 - byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2, normalize
	//   - 0x41 - byte3, alignment 4, on 4 bytes extracts first three bytes, reads the values with additional offset +2
	//   - 0x42 - byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2, normalize
	//   - 0x43 - byte3, alignment 4, on 4 bytes extracts last three bytes, reads the values with additional offset +2

	const uint offset = settings & 0x00ff;  // max offset is 255
	const uint64_t addr = vertexDataPtr + offset;
	const uint type = settings & 0xff00;

	// float3
	if(type == 0x2000)
		return AlignedVec3Ref(addr).value;
	if(type == 0x2100)
		return UnalignedVec3Ref(addr).value;

	// half3
	if(type == 0x2200) {
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]).x);
	}
	if(type == 0x2300) {
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(unpackHalf2x16(v[0]).y, unpackHalf2x16(v[1]));
	}

	// uint3
	if(type == 0x2400) {
		// alignment 16, normalize
		uvec3 v = AlignedUVec3Ref(addr).value;
		return vec3(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff,
		            float(v.z) / 0xffffffff);
	}
	if(type == 0x2500)
		// alignment 16, do not normalize
		return AlignedUVec3Ref(addr).value;
	if(type == 0x2600) {
		// alignment 4, normalize
		uvec3 v = UnalignedUVec3Ref(addr).value;
		return vec3(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff,
		            float(v.z) / 0xffffffff);
	}
	if(type == 0x2700)
		// alignment 16, do not normalize
		return UnalignedUVec3Ref(addr).value;

	// int3
	if(type == 0x2800) {
		// alignment 16, normalize
		ivec3 v = AlignedIVec3Ref(addr).value;
		return max(vec3(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff,
		                float(v.z) / 0x7fffffff),
		           -1.);
	}
	if(type == 0x2900)
		// alignment 16, do not normalize
		return AlignedIVec3Ref(addr).value;
	if(type == 0x2a00) {
		// alignment 4, normalize
		ivec3 v = UnalignedIVec3Ref(addr).value;
		return max(vec3(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff,
		                float(v.z) / 0x7fffffff),
		           -1.);
	}
	if(type == 0x2b00)
		// alignment 4, do not normalize
		return UnalignedIVec3Ref(addr).value;

	// ushort3
	if(type == 0x2c00) {
		// alignment 4, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]).x);
	}
	if(type == 0x2d00) {
		// alignment 4, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff);
	}
	if(type == 0x2e00) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(unpackUnorm2x16(v[0]).y, unpackUnorm2x16(v[1]));
	}
	if(type == 0x2f00) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
	}

	// short3
	if(type == 0x3000) {
		// alignment 4, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]).x);
	}
	if(type == 0x3100) {
		// alignment 4, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		ivec3 r = ivec3(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff));
		r |= 0xffff0000 * (r >> 15);
		return vec3(r);
	}
	if(type == 0x3200) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec3(unpackSnorm2x16(v[0]).y, unpackSnorm2x16(v[1]));
	}
	if(type == 0x3300) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		ivec3 r = ivec3(int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
		r |= 0xffff0000 * (r >> 15);
		return vec3(r);
	}

	// ubyte
	if(type == 0x3400) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).xyz;
	}
	if(type == 0x3500) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return vec3(v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff);
	}
	if(type == 0x3600) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v).yzw;
	}
	if(type == 0x3700) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return vec3((v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff);
	}
	if(type == 0x3800) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackUnorm4x8(v[0]).xyz;
	}
	if(type == 0x3900) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return vec3(v[0] & 0xff, (v[0] >> 8) & 0xff, (v[0] >> 16) & 0xff);
	}
	if(type == 0x3a00) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackUnorm4x8(v[0]).yzw;
	}
	if(type == 0x3b00) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return vec3((v[0] >> 8) & 0xff, (v[0] >> 16) & 0xff, (v[0] >> 24) & 0xff);
	}

	// byte
	if(type == 0x3c00) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm4x8(v).xyz;
	}
	if(type == 0x3d00) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		ivec3 r = ivec3(int(v & 0xff), int((v >> 8) & 0xff), int((v >> 16) & 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec3(r);
	}
	if(type == 0x3e00) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm4x8(v).yzw;
	}
	if(type == 0x3f00) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		ivec3 r = ivec3(int((v >> 8) & 0xff), int((v >> 16) & 0xff), int((v >> 24) & 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec3(r);
	}
	if(type == 0x4000) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackSnorm4x8(v[0]).xyz;
	}
	if(type == 0x4100) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		ivec3 r = ivec3(int(v[0] & 0xff), int((v[0] >> 8) & 0xff), int((v[0] >> 16) & 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec3(r);
	}
	if(type == 0x4200) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackSnorm4x8(v[0]).yzw;
	}
	if(type == 0x4300) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		ivec3 r = ivec3(int((v[0] >> 8) & 0xff), int((v[0] >> 16) & 0xff), int((v[0] >> 24) >> 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec3(r);
	}

	// return NaN
	return vec3(0/0);
}

)") + std::string(R"(

//
//  read vec4
//
vec4 readVec4(uint64_t vertexDataPtr, uint settings)
{
	// settings:
	// bits 0..7 - offset (0..255)
	// bits 8..15 - type
	//
	// type values:
	//   - 0x00 - not used attribute
	//   - 0x01 - float4, alignment 16
	//   - 0x02 - half4, alignment 8
	//   - 0x03 - half4, alignment 4
	//   - 0x04 - half4, alignment 4, reads the values with additional offset +2
	//   - 0x05 - uint4 normalized, alignment 16
	//   - 0x06 - uint4, alignment 16
	//   - 0x07 - int4 normalized, alignment 16
	//   - 0x08 - int4, alignment 16
	//   - 0x09 - ushort4 normalized, alignment 8
	//   - 0x0a - ushort4, alignment 8
	//   - 0x0b - ushort4 normalized, alignment 4
	//   - 0x0c - ushort4, alignment 4
	//   - 0x0d - ushort4 normalized, alignment 4, reads the values with additional offset +2
	//   - 0x0e - ushort4, alignment 4, reads the values with additional offset +2
	//   - 0x0f - short4 normalized, alignment 8
	//   - 0x10 - short4, alignment 8
	//   - 0x11 - short4 normalized, alignment 4
	//   - 0x12 - short4, alignment 4
	//   - 0x13 - short4 normalized, alignment 4, reads the values with additional offset +2
	//   - 0x14 - short4, alignment 4, reads the values with additional offset +2
	//   - 0x15 - ubyte4 normalize, alignment 4
	//   - 0x16 - ubyte4, alignment 4
	//   - 0x17 - ubyte4 normalize, alignment 4, reads the values with additional offset +2
	//   - 0x18 - ubyte4, alignment 4, reads the values with additional offset +2
	//   - 0x19 - byte4 normalize, alignment 4
	//   - 0x1a - byte4, alignment 4
	//   - 0x1b - byte4 normalize, alignment 4, reads the values with additional offset +2
	//   - 0x1c - byte4, alignment 4, reads the values with additional offset +2

	uint offset = settings & 0x00ff;  // max offset is 255
	uint64_t addr = vertexDataPtr + offset;
	uint type = settings & 0xff00;

	// float4
	if(type == 0x0100)
		return AlignedVec4Ref(addr).value;

	// half4
	if(type == 0x0200) {
		uvec2 v = AlignedUVec2Ref(addr).value;
		return vec4(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]));
	}
	if(type == 0x0300) {
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec4(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]));
	}
	if(type == 0x0400) {
		uvec3 v = UnalignedUVec3Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		v[1] = (v[1] >> 16) | (v[2] << 16);
		return vec4(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]));
	}

	// uint4
	if(type == 0x0500) {
		// alignment 16, normalize
		uvec4 v = AlignedUVec4Ref(addr).value;
		return vec4(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff,
		            float(v.z) / 0xffffffff, float(v.w) / 0xffffffff);
	}
	if(type == 0x0600)
		// alignment 16, do not normalize
		return AlignedUVec4Ref(addr).value;

	// int4
	if(type == 0x0700) {
		// alignment 16, normalize
		ivec4 v = AlignedIVec4Ref(addr).value;
		return max(vec4(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff,
		                float(v.z) / 0x7fffffff, float(v.w) / 0x7fffffff),
		           -1.);
	}
	if(type == 0x0800)
		// alignment 16, do not normalize
		return AlignedIVec4Ref(addr).value;

	// ushort4
	if(type == 0x0900) {
		// alignment 8, normalize
		uvec2 v = AlignedUVec2Ref(addr).value;
		return vec4(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]));
	}
	if(type == 0x0a00) {
		// alignment 8, do not normalize
		uvec2 v = AlignedUVec2Ref(addr).value;
		return vec4(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
	}
	if(type == 0x0b00) {
		// alignment 4, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec4(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]));
	}
	if(type == 0x0c00) {
		// alignment 4, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec4(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
	}
	if(type == 0x0d00) {
		// alignment 4, offset +2, normalize
		uvec3 v = UnalignedUVec3Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		v[1] = (v[1] >> 16) | (v[2] << 16);
		return vec4(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]));
	}
	if(type == 0x0e00) {
		// alignment 4, offset +2, do not normalize
		uvec3 v = UnalignedUVec3Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		v[1] = (v[1] >> 16) | (v[2] << 16);
		return vec4(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
	}

	// short4
	if(type == 0x0f00) {
		// alignment 8, normalize
		uvec2 v = AlignedUVec2Ref(addr).value;
		return vec4(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]));
	}
	if(type == 0x1000) {
		// alignment 8, do not normalize
		uvec2 v = AlignedUVec2Ref(addr).value;
		ivec4 r = ivec4(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
		r |= 0xffff0000 * (r >> 15);
		return vec4(r);
	}
	if(type == 0x1100) {
		// alignment 4, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		return vec4(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]));
	}
	if(type == 0x1200) {
		// alignment 4, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		ivec4 r = ivec4(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
		r |= 0xffff0000 * (r >> 15);
		return vec4(r);
	}
	if(type == 0x1300) {
		// alignment 4, offset +2, normalize
		uvec3 v = UnalignedUVec3Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		v[1] = (v[1] >> 16) | (v[2] << 16);
		return vec4(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]));
	}
	if(type == 0x1400) {
		// alignment 4, offset +2, do not normalize
		uvec3 v = UnalignedUVec3Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		v[1] = (v[1] >> 16) | (v[2] << 16);
		ivec4 r = ivec4(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
		r |= 0xffff0000 * (r >> 15);
		return vec4(r);
	}

	// ubyte
	if(type == 0x1500) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackUnorm4x8(v);
	}
	if(type == 0x1600) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		return vec4(v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff);
	}
	if(type == 0x1700) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackUnorm4x8(v[0]);
	}
	if(type == 0x1800) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return vec4(v[0] & 0xff, (v[0] >> 8) & 0xff, (v[0] >> 16) & 0xff, (v[0] >> 24) & 0xff);
	}

	// byte
	if(type == 0x1900) {
		// alignment 4, normalize
		uint v = AlignedUIntRef(addr).value;
		return unpackSnorm4x8(v);
	}
	if(type == 0x1a00) {
		// alignment 4, do not normalize
		uint v = AlignedUIntRef(addr).value;
		ivec4 r = ivec4(int(v & 0xff), int((v >> 8) & 0xff), int((v >> 16) & 0xff), int((v >> 24) >> 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec4(r);
	}
	if(type == 0x1b00) {
		// alignment 4, offset +2, normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		return unpackSnorm4x8(v[0]);
	}
	if(type == 0x1c00) {
		// alignment 4, offset +2, do not normalize
		uvec2 v = UnalignedUVec2Ref(addr).value;
		v[0] = (v[0] >> 16) | (v[1] << 16);
		ivec4 r = ivec4(int(v[0] & 0xff), int((v[0] >> 8) & 0xff), int((v[0] >> 16) & 0xff), int((v[0] >> 24) >> 0xff));
		r |= 0xffffff00 * (r >> 7);
		return vec4(r);
	}

	// try readVec3
	return vec4(readVec3(vertexDataPtr, settings), 1);
}
)") + std::string(R"(

layout(push_constant) uniform pushConstants {
	layout(offset=0) uint64_t drawablePointersBufferPtr;  // one buffer for the whole scene
	layout(offset=8) uint64_t sceneDataPtr;  // one buffer for the whole scene
	layout(offset=16) uint attribAccessInfoList[8];  // per-stateSet attribAccessInfo for 16 attribs
	layout(offset=48) uint materialSetup;
};

uint getColorAccessInfo()  {
    return attribAccessInfoList[1] >> 16;
}

uint getMaterialModel(uint materialSetup)  { return materialSetup & 0x03; }
uint getMaterialFirstTextureOffset(uint materialSetup)  { return materialSetup & 0xfc; }
bool getMaterialUseColorAttribute(uint materialSetup)  { return (materialSetup & 0x0300) != 0; }
bool getMaterialUseColorAttributeForAmbientAndDiffuse(uint materialSetup)  { return (materialSetup & 0x0100) != 0; }
bool getMaterialUseColorAttributeForDiffuseOnly(uint materialSetup)  { return (materialSetup & 0x0200) != 0; }
bool getMaterialIgnoreColorAttributeAlpha(uint materialSetup)  { return (materialSetup & 0x0400) != 0; }
bool getMaterialIgnoreMaterialAlpha(uint materialSetup)  { return (materialSetup & 0x0800) != 0; }
bool getMaterialIgnoreBaseTextureAlpha(uint materialSetup)  { return (materialSetup & 0x1000) != 0; }

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
BaseMaterialRef {
    layout(offset=0) uint settings;
};

)");


        auto s = std::array{
            ShaderInfo{.sourceFile = "empty_main",
                       .code = R"(
#version 460
void main(void) {}
)",
                        .kind = shaderc_fragment_shader,
            },
            ShaderInfo{.sourceFile = "simple",
                       .code = R"(
#version 460

layout(location = 0) out vec4 outColor;

void main(void) {
    outColor = vec4(1);
}
)",
                        .kind = shaderc_fragment_shader,
            },
            ShaderInfo{.sourceFile = "simple_with_include",
                       .code = interface + R"(

layout(location = 0) out vec4 outColor;

void main(void) {
    outColor = vec4(1);
}
)",
                       .kind = shaderc_fragment_shader,
            },
            ShaderInfo{.sourceFile = "uber_variant",
                       .code = interface + R"(

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    uint colorAccessInfo = getColorAccessInfo();

    vec3 color = readVec3(vertex0DataPtr, colorAccessInfo);

    outColor = vec4(color, 1);
}
)",
                       .kind = shaderc_fragment_shader,
            },
            ShaderInfo{.sourceFile = "uber_variant_with_specialization",
                       .code = interface + R"(

layout(constant_id = 0) const uint s1 = 0;

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    // uint colorAccessInfo = s1;
    vec3 color = readVec3(vertex0DataPtr, s1);

    outColor = vec4(color, 1);
}
)",
                       .kind = shaderc_fragment_shader,
                       .specializationInfo = &spec1
            },
            ShaderInfo{.sourceFile = "const_variant",
                       .code = interface + R"(

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    const uint colorAccessInfo = 0x2000;

    vec3 color = readVec3(vertex0DataPtr, colorAccessInfo);

    outColor = vec4(color, 1);
}
)",
                       .kind = shaderc_fragment_shader
            },
            ShaderInfo{.sourceFile = "specialized_variant",
                       .code =
                               // interface +
                               R"(
#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

//
//  buffer references
//
layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
AlignedVec3Ref {
	vec3 value;
};

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    vec3 color = AlignedVec3Ref(vertex0DataPtr).value;

    outColor = vec4(color, 1);
}
)",
                       .kind = shaderc_fragment_shader
            },
            ShaderInfo{.sourceFile = "specialized_variant2",
                    .code = interface + R"(

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    vec3 color = AlignedVec3Ref(vertex0DataPtr + 0).value;

    outColor = vec4(color, 1);
}
)",
                    .kind = shaderc_fragment_shader
            },
            ShaderInfo{.sourceFile = "ifs",
                    .code = interface + R"(

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    uint settings = BaseMaterialRef( inVertexAndDrawableDataPtr.w).settings;

    if(getMaterialUseColorAttribute(settings)) {
        const uint colorAccessInfo = 0x2000;
        vec3 color = readVec3(vertex0DataPtr, colorAccessInfo);
        outColor = vec4(color, 1);
    }
    else {
        outColor = vec4(1, 1, 1, 1);
    }
}
)",
                    .kind = shaderc_fragment_shader
            },
            ShaderInfo{.sourceFile = "ifs_material_attrib",
                    .code = interface + R"(

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    const uint settings = 0x0300;

    if(getMaterialUseColorAttribute(settings)) {
        const uint colorAccessInfo = 0x2000;
        vec3 color = readVec3(vertex0DataPtr, colorAccessInfo);
        outColor = vec4(color, 1);
    }
    else {
        outColor = vec4(1, 1, 1, 1);
    }
}
)",
                    .kind = shaderc_fragment_shader
            },
            ShaderInfo{.sourceFile = "ifs_no_material_attrib",
                    .code = interface + R"(

layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3

layout(location = 0) out vec4 outColor;

void main(void) {

    uint64_t vertex0DataPtr = inVertexAndDrawableDataPtr.x;
	uint64_t vertex1DataPtr = inVertexAndDrawableDataPtr.y;
	uint64_t vertex2DataPtr = inVertexAndDrawableDataPtr.z;

    const uint settings = 0;

    if(getMaterialUseColorAttribute(settings)) {
        const uint colorAccessInfo = 0x2000;
        vec3 color = readVec3(vertex0DataPtr, colorAccessInfo);
        outColor = vec4(color, 1);
    }
    else {
        outColor = vec4(1, 1, 1, 1);
    }
}
)",
                    .kind = shaderc_fragment_shader
            }
        };

        createShaders(s, outputDir);
    }
    catch(std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    device.destroy();
    vulkanInstance.destroy();

    return 0;
}
