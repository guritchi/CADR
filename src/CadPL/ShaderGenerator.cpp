#include <CadPL/ShaderGenerator.h>
#include <CadPL/ShaderLibrary.h>
#include <CadR/VulkanDevice.h>

#include <ShaderGeneratorHash.hpp>

#include <shaderc/shaderc.hpp>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

using namespace std;
using namespace CadPL;


static constexpr bool optimizeSpirV = true;
static constexpr uint32_t SpirvMagicNumber = 0x07230203;
static constexpr size_t CodeStringReservation = 128000; // 128kB


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


std::string compileToAssembly(const std::string& name,
                              shaderc_shader_kind kind,
                              const std::string& source,
                              bool optimize)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    if (optimize) {
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
    }

    shaderc::AssemblyCompilationResult result = compiler.CompileGlslToSpvAssembly(
            source, kind, name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << result.GetErrorMessage();
        throw std::runtime_error("Can't compile file: " + std::string(name));
    }

    return {result.cbegin(), result.cend()};
}

shaderc::SpvCompilationResult compileToSpirV(const std::string& name,
                                             shaderc_shader_kind kind,
                                             const std::string& source,
                                             bool optimize)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    if (optimize) {
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
    }

    shaderc::SpvCompilationResult module =
            compiler.CompileGlslToSpv(source, kind, name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << module.GetErrorMessage() << std::endl;
        throw std::runtime_error("Can't compile shader: " + std::string(name));
    }
    return module;
}


static std::string cacheDirectory() {
    std::string name;
#ifdef _WIN32
    name += "\\\\?\\"; // long path
#endif
    name += std::filesystem::current_path().string();
    name += "\\ShaderCache";
    return name;
}

static std::string createCacheName(shaderc_shader_kind kind, const ShaderState& state) {
    const auto &encoded = state.serialize();
    std::string name = cacheDirectory();
    name += "\\";
    if (!optimizeSpirV) {
        name += "d";
    }
    name += std::to_string(kind);
    name += encoded;
    return name;
}

void ShaderGenerator::initializeCache() {
    const auto &dir = cacheDirectory();
    std::string hash;
    std::ifstream hashfile(dir + "\\hash.txt");
    if (hashfile.is_open()) {
        std::getline(hashfile, hash);
        hashfile.close();
    }
    else {
        std::filesystem::create_directory(dir);
    }
    if (hash != GEN_HASH) {
        for (const auto &entry: std::filesystem::directory_iterator(dir)) {
            const auto &ext = entry.path().extension();
            if (ext == ".spv" || ext == ".glsl" || ext == ".txt") {
                if (!std::filesystem::remove(entry.path())) {
                    std::cerr << "Failed to remove file\n";
                }
            }
        }
        std::ofstream hashfile(dir + "\\hash.txt");
        hashfile << GEN_HASH;
        hashfile.close();
    }
}


static std::uint32_t swapEndianness(std::uint32_t word) noexcept {
#ifdef __cpp_lib_byteswap
    return std::byteswap(word);
#elif defined(__GNUC__)
    return __builtin_bswap32(word);
#elif defined(_MSC_VER)
    return _byteswap_ulong(word);
#else
    return ((word & 0xff) << 24)
         | ((word & 0xff00) << 8)
         | ((word & 0xff0000) >> 8)
         | ((word & 0xff000000) >> 24);
#endif
}

static std::vector<uint32_t> readSpirvFromFile(const std::string &fileName) {
    std::ifstream file(fileName, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    file.unsetf(std::ios::skipws);

    file.seekg(0, std::ios::end);
    auto fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize == 0 || fileSize % sizeof(std::uint32_t) != 0) {
        return {};
    }
    std::vector<uint32_t> spirv(fileSize / 4);
    if (!file.read(reinterpret_cast<char*>(spirv.data()), fileSize))
        throw std::runtime_error("Failed to read file: " + fileName);

    if (spirv[0] == swapEndianness(SpirvMagicNumber)) {
        for (auto &word: spirv) {
            word = swapEndianness(word);
        }
    }
    else if (spirv[0] != SpirvMagicNumber) {
        return {};
    }

    return spirv;
}

static void writeFile(const std::string &name, const char *data, size_t size) {
    std::ofstream file(name, std::ios::out | std::ios::binary);
    if (file.is_open()) {
        file.unsetf(std::ios::skipws);
        file.write(data, size);
        file.close();
    }
    else {
        std::cerr << std::strerror(errno) << std::endl;
    }
}

static void writeFile(const std::string &name, const std::string &content) {
    writeFile(name, reinterpret_cast<const char *>(content.data()), content.size());
}

static vk::ShaderModule createShader(const std::string &code, const std::string &name, const std::string &cacheFileName, shaderc_shader_kind kind, CadR::VulkanDevice& device) {

    const auto start = std::chrono::system_clock::now();

    vk::ShaderModuleCreateInfo info;
    info.flags = vk::ShaderModuleCreateFlags();

    std::vector<uint32_t> spirv;
    shaderc::SpvCompilationResult compilation;
    spirv = readSpirvFromFile(cacheFileName + ".spv");

    if (spirv.empty()) {
#ifndef NDEBUG
        // debug files
        writeFile(cacheFileName + ".glsl", code);
        // writeFile(cacheFileName + ".txt", compileToAssembly(name, kind, code, optimizeSpirV));
#endif

        compilation = compileToSpirV(name, kind, code, optimizeSpirV);
        const auto size = (compilation.cend() - compilation.cbegin()) * sizeof(uint32_t);

        writeFile(cacheFileName + ".spv", reinterpret_cast<const char *>(compilation.cbegin()), size);
        info.codeSize = size;
        info.pCode = compilation.cbegin();
    }
    else {
        info.codeSize = spirv.size() * sizeof(uint32_t);
        info.pCode = spirv.data();
    }

    const auto spirvTime = std::chrono::system_clock::now();

    const auto module = device.createShaderModule(info);

    const auto createTime = std::chrono::system_clock::now();
    std::cout << name << " compile: " << std::chrono::duration_cast<std::chrono::milliseconds>(spirvTime - start).count()
              << "ms, createShaderModule(): " << std::chrono::duration_cast<std::chrono::milliseconds>(createTime - spirvTime).count() << "ms\n";

    return module;
}


vk::ShaderModule ShaderGenerator::createVertexShader(const ShaderState& state, CadR::VulkanDevice& device)
{
	const uint32_t* code;
	size_t size;
	if(state.idBuffer) {
		code = vertexIdBufferUberShaderSpirv;
		size = sizeof(vertexIdBufferUberShaderSpirv);
	}
	else {
		code = vertexUberShaderSpirv;
		size = sizeof(vertexUberShaderSpirv);
	}

	return
		device.createShaderModule(
			vk::ShaderModuleCreateInfo(
				vk::ShaderModuleCreateFlags(),  // flags
				size,  // codeSize
				code   // pCode
			)
		);
}


vk::ShaderModule ShaderGenerator::createGeometryShader(const ShaderState& state, CadR::VulkanDevice& device)
{
	const uint32_t* code;
	size_t size;
	if(state.idBuffer) {
		code = geometryIdBufferUberShaderSpirv;
		size = sizeof(geometryIdBufferUberShaderSpirv);
	}
	else {
		code = geometryUberShaderSpirv;
		size = sizeof(geometryUberShaderSpirv);
	}

	return
		device.createShaderModule(
			vk::ShaderModuleCreateInfo(
				vk::ShaderModuleCreateFlags(),  // flags
				size,  // codeSize
				code   // pCode
			)
		);
}


vk::ShaderModule ShaderGenerator::createFragmentShader(const ShaderState& state, CadR::VulkanDevice& device)
{
	const uint32_t* code;
	size_t size;
	if(state.idBuffer) {
		code = fragmentIdBufferUberShaderSpirv;
		size = sizeof(fragmentIdBufferUberShaderSpirv);
	}
	else {
		code = fragmentUberShaderSpirv;
		size = sizeof(fragmentUberShaderSpirv);
	}

	return
		device.createShaderModule(
			vk::ShaderModuleCreateInfo(
				vk::ShaderModuleCreateFlags(),  // flags
				size,  // codeSize
				code   // pCode
			)
		);
}
