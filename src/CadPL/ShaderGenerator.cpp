#include <CadPL/ShaderGenerator.h>
#include <CadPL/ShaderState.h>
#include <CadR/VulkanDevice.h>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <set>
#include <variant>
#include "glm/detail/setup.hpp"
#include <shaderc/shaderc.hpp>

#include "DebugUtils.h"
#include <ShaderGeneratorHash.hpp>
#ifdef CADPL_USE_PREGEN
#include "shaders/PregeneratedShaders.hpp"
#endif

using namespace std;
using namespace CadPL;


static constexpr bool ShaderValidation = true;
static constexpr bool OptimizeSpirV = true;
static constexpr uint32_t SpirvMagicNumber = 0x07230203;
static constexpr size_t CodeStringReservation = 128000; // 128kB


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
        std::cerr << "shader: " << name << "\n" << module.GetErrorMessage() << std::endl;
        throw std::runtime_error("Can't compile shader: " + std::string(name));
    }
    return module;
}


static std::string cacheDirectory;
static bool cacheSpirV = false;

std::string ShaderGenerator::defaultCacheDirectory()
{
    std::string name;
#ifdef _WIN32
    name += R"(\\?\)"; // long path
#endif
    name += std::filesystem::current_path().string();
    name += "\\ShaderCache";
    return name;
}

std::string& ShaderGenerator::getCacheDirectory()
{
    return cacheDirectory;
}

std::string ShaderGenerator::createCacheName(const std::string_view directory, shaderc_shader_kind kind, const std::string& state)
{
    std::string name = std::string(directory);
    if (!directory.empty()) {
        name += "\\";
    }
    if (!OptimizeSpirV) {
        name += "d";
    }
    name += std::to_string(kind);
    name += state;
    return name;
}

void ShaderGenerator::initialize()
{
#ifdef CADPL_USE_PREGEN
    initializePregeneratedShaders();
#endif
}

void ShaderGenerator::initializeCache(const std::string_view directory, bool wipe)
{
    cacheDirectory = directory;
    if (!cacheDirectory.empty()) {
        std::string hashFilename = cacheDirectory + "\\hash.txt";
        std::string hash;
        {
            std::ifstream hashfile(hashFilename);
            if (hashfile.is_open()) {
                std::getline(hashfile, hash);
                hashfile.close();
            } else {
                std::filesystem::create_directory(cacheDirectory);
            }
        }
        if (hash != GEN_HASH || wipe) {
            for (const auto &entry: std::filesystem::directory_iterator(cacheDirectory)) {
                const auto &ext = entry.path().extension();
                if (ext == ".spv" || ext == ".glsl" || ext == ".txt" || ext == ".bin") {
                    if (!std::filesystem::remove(entry.path())) {
                        std::cerr << "Failed to remove file\n";
                    }
                }
            }
            std::ofstream hashfile(hashFilename);
            hashfile << GEN_HASH;
            hashfile.close();
        }
    }
}

void ShaderGenerator::enableSpirVCache(bool enabled)
{
    cacheSpirV = enabled;
}

bool ShaderGenerator::useSpirVCache() {
    return cacheSpirV;
}


struct PipelineCacheHeader {
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t driverVersion;
    vk::ArrayWrapper1D<uint8_t, VK_UUID_SIZE> pipelineCacheUUID;

    void setProperties(const vk::PhysicalDeviceProperties &properties) {
        vendorID = properties.vendorID;
        deviceID = properties.deviceID;
        driverVersion = properties.driverVersion;
        pipelineCacheUUID = properties.pipelineCacheUUID;
    }

    bool valid(const vk::PhysicalDeviceProperties &properties) const {
        return vendorID == properties.vendorID &&
               deviceID == properties.deviceID &&
               driverVersion == properties.driverVersion &&
               pipelineCacheUUID == properties.pipelineCacheUUID;
    }

};

void ShaderGenerator::savePipelineCache(const CadR::VulkanDevice& device, vk::PipelineCache cache, const vk::PhysicalDeviceProperties &properties, const std::string_view filename)
{
    auto func = (PFN_vkGetPipelineCacheData)device.getProcAddr("vkGetPipelineCacheData");
    if (!func) {
        std::cerr << "Failed to get vkGetPipelineCacheData function\n";
        return;
    }
    size_t size;
    std::vector<uint8_t> data;
    VkResult result;
    do {
        result = func(device.handle(), cache, &size, nullptr);
        if ( result == VK_SUCCESS && size) {
            data.resize(size);
            result = func(device.handle(), cache, &size, data.data());
        }
    } while (result == VK_INCOMPLETE);
    if (result != VK_SUCCESS) {
        std::cerr << "Failed to get vkGetPipelineCacheData\n";
        return;
    }
    if (size == 0) {
        return;
    }
    if (size < data.size()) {
        data.resize(size);
    }

    PipelineCacheHeader header;
    header.setProperties(properties);

    std::ofstream file(filename.data(), std::ios::out | std::ios::binary);
    if (file.is_open()) {
        file.unsetf(std::ios::skipws);
        file.write(reinterpret_cast<const char *>(&header), static_cast<std::streamsize>(sizeof(PipelineCacheHeader)));
        file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
        file.close();
    }
    else {
        std::cerr << "Error writing file: " << std::strerror(errno) << '\n';
    }
}

void ShaderGenerator::savePipelineCache(const CadR::VulkanDevice& device, vk::PipelineCache cache, const vk::PhysicalDeviceProperties &properties)
{
    if (!cacheDirectory.empty()) {
        savePipelineCache(device, cache, properties, cacheDirectory + "\\cache.bin");
    }
}

std::vector<uint8_t> ShaderGenerator::loadPipelineCacheData(const vk::PhysicalDeviceProperties &properties, const std::string_view filename)
{
    std::ifstream file(filename.data(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        return {};
    }
    file.unsetf(std::ios::skipws);

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    PipelineCacheHeader header;
    if (fileSize <= sizeof(PipelineCacheHeader)) {
        return {};
    }
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(PipelineCacheHeader))) {
        std::cerr << "Failed to read file: " << filename << '\n';
        return {};
    }
    if (!header.valid(properties)) {
        std::cerr << "PipelineCacheHeader does not match\n";
        return {};
    }
    std::vector<uint8_t> data(fileSize - sizeof(PipelineCacheHeader));
    if (!file.read(reinterpret_cast<char*>(data.data()), data.size())) {
        std::cerr << "Failed to read file: " << filename << '\n';
        return {};
    }
    return data;
}

vk::PipelineCache ShaderGenerator::loadPipelineCache(const CadR::VulkanDevice &device, const vk::PhysicalDeviceProperties &properties, vk::PipelineCacheCreateFlags flags, const std::string_view filename, size_t *loadedCacheSize)
{
    std::vector<uint8_t> data;
    if (!filename.empty()) {
        data = loadPipelineCacheData(properties, filename);
        if (loadedCacheSize) {
            *loadedCacheSize = data.size();
        }
    }
    auto result = device.createPipelineCache(
        vk::PipelineCacheCreateInfo(
            flags,
            data.size(),  // initialDataSize
            data.data()   // pInitialData
        )
    );
    return result;
}

vk::PipelineCache ShaderGenerator::loadPipelineCache(const CadR::VulkanDevice &device, const vk::PhysicalDeviceProperties &properties, vk::PipelineCacheCreateFlags flags, size_t *loadedCacheSize)
{
    return loadPipelineCache(device, properties, flags, cacheDirectory.empty()? "" : (cacheDirectory + "\\cache.bin"), loadedCacheSize);
}

static std::uint32_t swapEndianness(std::uint32_t word) noexcept
{
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

static std::vector<uint32_t> readSpirvFromFile(const std::string &fileName)
{
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

static void writeFile(const std::string &name, const char *data, size_t size)
{
    std::ofstream file(name, std::ios::out | std::ios::binary);
    if (file.is_open()) {
        file.unsetf(std::ios::skipws);
        file.write(data, static_cast<std::streamsize>(size));
        file.close();
    }
    else {
        std::cerr << std::strerror(errno) << std::endl;
    }
}

static void writeFile(const std::string &name, const std::string &content)
{
    writeFile(name, reinterpret_cast<const char *>(content.data()), content.size());
}

static std::string toHex(size_t value)
{
    std::stringstream stream;
    stream << std::hex << value;
    return stream.str();
}


class OutputLine {
    std::string &_buffer;

public:
    explicit OutputLine(std::string &buffer) noexcept
        : _buffer(buffer)
    {
    }

    template<typename T>
    OutputLine& operator<<(const T &data) {
        if constexpr (std::is_arithmetic_v<T>) {
            _buffer += std::to_string(data);
        }
        else {
            _buffer += data;
        }
        return *this;
    }

};


class OutputStream {

    std::string _buffer;
#ifndef NDEBUG
    int _indent = 0;

    void putIndent()
    {
        _buffer.append(_indent, ' ');
    }
#endif

public:

    explicit OutputStream() = default;

    OutputStream(const OutputStream&) = delete;
    OutputStream& operator=(const OutputStream&) = delete;

    std::string& string() {
        return _buffer;
    }

    const std::string& string() const
    {
        return _buffer;
    }

    void reserve(size_t size)
    {
        _buffer.reserve(size);
    }
    void addIndent(int indent) noexcept
    {
#ifndef NDEBUG
        _indent += indent;
        if (_indent < 0) {
            _indent = 0;
        }
#else
        (void)indent;
#endif
    }

    OutputLine operator()()
    {
#ifndef NDEBUG
        putIndent();
#endif
        return OutputLine{_buffer};
    }

    template<typename T>
    OutputStream& operator<<(const T &data)
    {
        (this->operator()() << data);
        return *this;
    }

    void push() noexcept
    {
#ifndef NDEBUG
        addIndent(4);
#endif
    }

    void pop() noexcept
    {
#ifndef NDEBUG
        addIndent(-4);
#endif
    }

};


// convenience class for passing simple strings or complex lamdas to generators
class Expression
{
    std::variant<const char*, std::string, std::function<void(OutputStream&)>> _value;

public:

    template<typename T>
    Expression(const T &value) : _value(value)
    {}

    operator bool() const {
        if (std::holds_alternative<const char*>(_value)) {
            return std::get<const char*>(_value) != nullptr;
        }
        if (std::holds_alternative<std::function<void(OutputStream&)>>(_value)) {
            return std::get<std::function<void(OutputStream&)>>(_value) != nullptr;
        }
        if (std::holds_alternative<std::string>(_value)) {
            return !std::get<std::string>(_value).empty();
        }
        return true;
    }

    void generate(OutputStream& output) const
    {
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>; // Get the underlying type of 'arg'
            if constexpr (std::is_same_v<T, std::function<void(OutputStream&)>>) {
                arg(output);
            }
            else {
                output << arg;
            }
        }, _value);
    }

};


struct AttributeInfo
{

    AttributeInfo(uint16_t accessInfo) noexcept
      : type(accessInfo >> 8)
      , offset(accessInfo & 0xFF)
    {
    }

    uint32_t type;
    uint32_t offset;

};

static constexpr const char* ReadBufferReferences = R"(
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
)";

struct FunctionLiteral {
    const char* type = "";
    const char* name = "";
    const char* code = "";
};

struct ReadFunction {
    AttributeType type = {};
    FunctionLiteral read;
};

static constexpr auto ReadFunctions = std::array{
    ReadFunction{ AttributeType::undefined, {"", ""}},
    // float4
    ReadFunction{ AttributeType::vec4A16, {"vec4", "readVec4FromVec4A16", R"(
(uint64_t addr) {
    return AlignedVec4Ref(addr).value;
}
)"}
    },
    // half4
	ReadFunction{ AttributeType::half4A8, {"vec4", "readVec4FromHalf4A8", R"(
(uint64_t addr) {
    uvec2 v = AlignedUVec2Ref(addr).value;
    return vec4(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::half4A4, {"vec4", "readVec4FromHalf4A4", R"(
(uint64_t addr) {
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec4(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::half4A4Offset2, {"vec4", "readVec4FromHalf4A4Offset2", R"(
(uint64_t addr) {
    uvec3 v = UnalignedUVec3Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    v[1] = (v[1] >> 16) | (v[2] << 16);
    return vec4(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]));
}
)"}
	},
    // uint4
	ReadFunction{ AttributeType::uint4A16Norm, {"vec4", "readVec4FromUint4A16Norm", R"(
(uint64_t addr) {
    // alignment 16, normalize
    uvec4 v = AlignedUVec4Ref(addr).value;
    return vec4(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff,
                float(v.z) / 0xffffffff, float(v.w) / 0xffffffff);
}
)"}
	},
	ReadFunction{ AttributeType::uint4A16, {"vec4", "readVec4FromUint4A16", R"(
(uint64_t addr) {
    // alignment 16, do not normalize
    return AlignedUVec4Ref(addr).value;
}
)"}
	},
    // int4
	ReadFunction{ AttributeType::int4A16Norm, {"vec4", "readVec4FromInt4A16Norm", R"(
(uint64_t addr) {
    // alignment 16, normalize
    ivec4 v = AlignedIVec4Ref(addr).value;
    return max(vec4(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff,
                    float(v.z) / 0x7fffffff, float(v.w) / 0x7fffffff),
               -1.);
}
)"}
	},
	ReadFunction{ AttributeType::int4A16, {"vec4", "readVec4FromInt4A16", R"(
(uint64_t addr) {
    // alignment 16, do not normalize
    return AlignedIVec4Ref(addr).value;
})"}
	},
    // ushort4
	ReadFunction{ AttributeType::ushort4A8Norm, {"vec4", "readVec4FromUshort4A8Norm", R"(
(uint64_t addr) {
    // alignment 8, normalize
    uvec2 v = AlignedUVec2Ref(addr).value;
    return vec4(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::ushort4A8, {"vec4", "readVec4FromUshort4A8", R"(
(uint64_t addr) {
    // alignment 8, do not normalize
    uvec2 v = AlignedUVec2Ref(addr).value;
    return vec4(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
}
)"}
	},
	ReadFunction{ AttributeType::ushort4A4Norm, {"vec4", "readVec4FromUshort4A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec4(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::ushort4A4, {"vec4", "readVec4FromUshort4A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec4(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
}
)"}
	},
	ReadFunction{ AttributeType::ushort4A4NormOffset2, {"vec4", "readVec4FromUshort4A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec3 v = UnalignedUVec3Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    v[1] = (v[1] >> 16) | (v[2] << 16);
    return vec4(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::ushort4A4Offset2, {"vec4", "readVec4FromUshort4A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec3 v = UnalignedUVec3Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    v[1] = (v[1] >> 16) | (v[2] << 16);
    return vec4(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
}
)"}
	},
    // short4
	ReadFunction{ AttributeType::short4A8Norm, {"vec4", "readVec4FromShort4A8Norm", R"(
(uint64_t addr) {
    // alignment 8, normalize
    uvec2 v = AlignedUVec2Ref(addr).value;
    return vec4(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::short4A8, {"vec4", "readVec4FromShort4A8", R"(
(uint64_t addr) {
    // alignment 8, do not normalize
    uvec2 v = AlignedUVec2Ref(addr).value;
    ivec4 r = ivec4(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
    r |= 0xffff0000 * (r >> 15);
    return vec4(r);
}
)"}
	},
	ReadFunction{ AttributeType::short4A4Norm, {"vec4", "readVec4FromShort4A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec4(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::short4A4, {"vec4", "readVec4FromShort4A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    ivec4 r = ivec4(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
    r |= 0xffff0000 * (r >> 15);
    return vec4(r);
})"}
	},
	ReadFunction{ AttributeType::short4A4NormOffset2, {"vec4", "readVec4FromShort4A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec3 v = UnalignedUVec3Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    v[1] = (v[1] >> 16) | (v[2] << 16);
    return vec4(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]));
}
)"}
	},
	ReadFunction{ AttributeType::short4A4Offset2, {"vec4", "readVec4FromShort4A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec3 v = UnalignedUVec3Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    v[1] = (v[1] >> 16) | (v[2] << 16);
    ivec4 r = ivec4(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
    r |= 0xffff0000 * (r >> 15);
    return vec4(r);
}
)"}
	},
    // ubyte
    ReadFunction{ AttributeType::ubyte4A4Norm, {"vec4", "readVec4FromUbyte4A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v);
}
)"}
    },
	ReadFunction{ AttributeType::ubyte4A4, {"vec4", "readVec4FromUbyte4A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return vec4(v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff);
}
)"}
	},
	ReadFunction{ AttributeType::ubyte4A4NormOffset2, {"vec4", "readVec4FromUbyte4A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackUnorm4x8(v[0]);
}
)"}
	},
	ReadFunction{ AttributeType::ubyte4A4Offset2, {"vec4", "readVec4FromUbyte4A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return vec4(v[0] & 0xff, (v[0] >> 8) & 0xff, (v[0] >> 16) & 0xff, (v[0] >> 24) & 0xff);
}
)"}
	},
    // byte
	ReadFunction{ AttributeType::byte4A4Norm, {"vec4", "readVec4FromByte4A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm4x8(v);
}
)"}
	},
	ReadFunction{ AttributeType::byte4A4, {"vec4", "readVec4FromByte4A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    ivec4 r = ivec4(int(v & 0xff), int((v >> 8) & 0xff), int((v >> 16) & 0xff), int((v >> 24) >> 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec4(r);
}
)"}
	},
	ReadFunction{ AttributeType::byte4A4NormOffset2, {"vec4", "readVec4FromByte4A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackSnorm4x8(v[0]);
}
)"}
	},
	ReadFunction{ AttributeType::byte4A4Offset2, {"vec4", "readVec4FromByte4A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    ivec4 r = ivec4(int(v[0] & 0xff), int((v[0] >> 8) & 0xff), int((v[0] >> 16) & 0xff), int((v[0] >> 24) >> 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec4(r);
}
)"}
	},
    // float3
    ReadFunction{ AttributeType::vec3A16, {"vec3", "readVec3FromVec3A16", R"(
(uint64_t addr) {
    return AlignedVec3Ref(addr).value;
}
)"}
    },
    ReadFunction{ AttributeType::vec3A4, {"vec3", "readVec3FromVec3A4", R"(
(uint64_t addr) {
    return UnalignedVec3Ref(addr).value;
}
)"}
    },
    // half3
    ReadFunction{ AttributeType::half3A4, {"vec3", "readVec3FromHalf3A4", R"(
(uint64_t addr) {
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(unpackHalf2x16(v[0]), unpackHalf2x16(v[1]).x);
}
)"}
	},
    ReadFunction{ AttributeType::half3A4Last6, {"vec3", "readVec3FromHalf3A4Last6", R"(
(uint64_t addr) {
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(unpackHalf2x16(v[0]).y, unpackHalf2x16(v[1]));
}
)"}
	},
    // uint3
    ReadFunction{ AttributeType::uint3A16Norm, {"vec3", "readVec3FromUint3A16Norm", R"(
(uint64_t addr) {
    // alignment 16, normalize
    uvec3 v = AlignedUVec3Ref(addr).value;
    return vec3(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff,
                    float(v.z) / 0xffffffff);
}
)"}
	},
    ReadFunction{ AttributeType::uint3A16, {"vec3", "readVec3FromUint3A16", R"(
(uint64_t addr) {
    // alignment 16, do not normalize
    return AlignedUVec3Ref(addr).value;
}
)"}
	},
    ReadFunction{ AttributeType::uint3A4Norm, {"vec3", "readVec3FromUint3A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uvec3 v = UnalignedUVec3Ref(addr).value;
    return vec3(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff,
                    float(v.z) / 0xffffffff);
}
)"}
	},
    ReadFunction{ AttributeType::uint3A4, {"vec3", "readVec3FromUint3A4", R"(
(uint64_t addr) {
    // alignment 16, do not normalize
    return UnalignedUVec3Ref(addr).value;
}
)"}
	},
    // int3
    ReadFunction{ AttributeType::int3A16Norm, {"vec3", "readVec3FromInt3A16Norm", R"(
(uint64_t addr) {
    // alignment 16, normalize
    ivec3 v = AlignedIVec3Ref(addr).value;
    return max(vec3(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff,
                        float(v.z) / 0x7fffffff),
                   -1.);
}
)"}
	},
    ReadFunction{ AttributeType::int3A16, {"vec3", "readVec3FromInt3A16", R"(
(uint64_t addr) {
    // alignment 16, do not normalize
    return AlignedIVec3Ref(addr).value;
}
)"}
	},
    ReadFunction{ AttributeType::int3A4Norm, {"vec3", "readVec3FromInt3A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    ivec3 v = UnalignedIVec3Ref(addr).value;
    return max(vec3(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff,
                        float(v.z) / 0x7fffffff),
                   -1.);
}
)"}
	},
    ReadFunction{ AttributeType::int3A4, {"vec3", "readVec3FromInt3A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    return UnalignedIVec3Ref(addr).value;
}
)"}
	},
    // ushort3
    ReadFunction{ AttributeType::ushort3A4NormFirst6, {"vec3", "readVec3FromUshort3A4NormFirst6", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(unpackUnorm2x16(v[0]), unpackUnorm2x16(v[1]).x);
}
)"}
	},
    ReadFunction{ AttributeType::ushort3A4First6, {"vec3", "readVec3FromUshort3A4First6", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(v[0] & 0xffff, v[0] >> 16, v[1] & 0xffff);
}
)"}
	},
    ReadFunction{ AttributeType::ushort3A4NormLast6, {"vec3", "readVec3FromUshort3A4NormLast6", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(unpackUnorm2x16(v[0]).y, unpackUnorm2x16(v[1]));
}
)"}
	},
    ReadFunction{ AttributeType::ushort3A4Last6, {"vec3", "readVec3FromUshort3A4Last6", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(v[0] >> 16, v[1] & 0xffff, v[1] >> 16);
}
)"}
	},
    ReadFunction{ AttributeType::short3A4NormFirst6, {"vec3", "readVec3FromShort3A4NormFirst6", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(unpackSnorm2x16(v[0]), unpackSnorm2x16(v[1]).x);
}
)"}
	},
    ReadFunction{ AttributeType::short3A4First6, {"vec3", "readVec3FromShort3A4First6", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    ivec3 r = ivec3(int(v[0] & 0xffff), int(v[0] >> 16), int(v[1] & 0xffff));
    r |= 0xffff0000 * (r >> 15);
    return vec3(r);
}
)"}
	},
    ReadFunction{ AttributeType::short3A4NormLast6, {"vec3", "readVec3FromShort3A4NormLast6", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec3(unpackSnorm2x16(v[0]).y, unpackSnorm2x16(v[1]));
}
)"}
	},
    ReadFunction{ AttributeType::short3A4Last6, {"vec3", "readVec3FromShort3A4Last6", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    ivec3 r = ivec3(int(v[0] >> 16), int(v[1] & 0xffff), int(v[1] >> 16));
    r |= 0xffff0000 * (r >> 15);
    return vec3(r);
}
)"}
	},
    // ubyte
    ReadFunction{ AttributeType::ubyte3A4NormFirst3, {"vec3", "readVec3FromUbyte3A4NormFirst3", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).xyz;
}
)"}
	},
    ReadFunction{ AttributeType::ubyte3A4First3, {"vec3", "readVec3FromUbyte3A4First3", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return vec3(v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff);
}
)"}
	},
    ReadFunction{ AttributeType::ubyte3A4NormLast3, {"vec3", "readVec3FromUbyte3A4NormLast3", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).yzw;
}
)"}
	},
    ReadFunction{ AttributeType::ubyte3A4Last3, {"vec3", "readVec3FromUbyte3A4Last3", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return vec3((v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff);
}
)"}
	},
    ReadFunction{ AttributeType::ubyte3A4NormFirst3Offset2, {"vec3", "readVec3FromUbyte3A4NormFirst3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackUnorm4x8(v[0]).xyz;
}
)"}
	},
    ReadFunction{ AttributeType::ubyte3A4First3Offset2, {"vec3", "readVec3FromUbyte3A4First3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return vec3(v[0] & 0xff, (v[0] >> 8) & 0xff, (v[0] >> 16) & 0xff);
}
)"}
	},
    ReadFunction{ AttributeType::ubyte3A4NormLast3Offset2, {"vec3", "readVec3FromUbyte3A4NormLast3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackUnorm4x8(v[0]).yzw;
}
)"}
	},
    ReadFunction{ AttributeType::ubyte3A4Last3Offset2, {"vec3", "readVec3FromUbyte3A4Last3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return vec3((v[0] >> 8) & 0xff, (v[0] >> 16) & 0xff, (v[0] >> 24) & 0xff);
}
)"}
	},
    // byte
    ReadFunction{ AttributeType::byte3A4NormFirst3, {"vec3", "readVec3FromByte3A4NormFirst3", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm4x8(v).xyz;
}
)"}
	},
    ReadFunction{ AttributeType::byte3A4First3, {"vec3", "readVec3FromByte3A4First3", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    ivec3 r = ivec3(int(v & 0xff), int((v >> 8) & 0xff), int((v >> 16) & 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec3(r);
}
)"}
	},
    ReadFunction{ AttributeType::byte3A4NormLast3, {"vec3", "readVec3FromByte3A4NormLast3", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm4x8(v).yzw;
}
)"}
	},
    ReadFunction{ AttributeType::byte3A4Last3, {"vec3", "readVec3FromByte3A4Last3", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    ivec3 r = ivec3(int((v >> 8) & 0xff), int((v >> 16) & 0xff), int((v >> 24) & 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec3(r);
}
)"}
	},
    ReadFunction{ AttributeType::byte3A4NormFirst3Offset2, {"vec3", "readVec3FromByte3A4NormFirst3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackSnorm4x8(v[0]).xyz;
}
)"}
	},
    ReadFunction{ AttributeType::byte3A4First3Offset2, {"vec3", "readVec3FromByte3A4First3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    ivec3 r = ivec3(int(v[0] & 0xff), int((v[0] >> 8) & 0xff), int((v[0] >> 16) & 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec3(r);
}
)"}
	},
    ReadFunction{ AttributeType::byte3A4NormLast3Offset2, {"vec3", "readVec3FromByte3A4NormLast3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackSnorm4x8(v[0]).yzw;
}
)"}
	},
    ReadFunction{ AttributeType::byte3A4Last3Offset2, {"vec3", "readVec3FromByte3A4Last3Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    ivec3 r = ivec3(int((v[0] >> 8) & 0xff), int((v[0] >> 16) & 0xff), int((v[0] >> 24) >> 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec3(r);
}
)"}
	},
    // float2
    ReadFunction{ AttributeType::vec2A8, {"vec2", "readVec2FromVec2A8", R"(
(uint64_t addr) {
    return AlignedVec2Ref(addr).value;
}
)"}
    },
    ReadFunction{ AttributeType::vec2A4, {"vec2", "readVec2FromVec2A4", R"(
(uint64_t addr) {
    return UnalignedVec2Ref(addr).value;
}
)"}
    },
    ReadFunction{ AttributeType::half2A4, {"vec2", "readVec2FromHalf2A4", R"(
(uint64_t addr) {
    uint v = AlignedUIntRef(addr).value;
    return unpackHalf2x16(v);
}
)"}
    },
    ReadFunction{ AttributeType::half2A4Offset2, {"vec2", "readVec2FromHalf2A4Offset2", R"(
(uint64_t addr) {
   uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackHalf2x16(v[0]);
}
)"}
    },
    // uint2
    ReadFunction{ AttributeType::uint2A8Norm, {"vec2", "readVec2FromUint2A8Norm", R"(
(uint64_t addr) {
    // alignment 8, normalize
    uvec2 v = AlignedUVec2Ref(addr).value;
    return vec2(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff);
}
)"}
    },
    ReadFunction{ AttributeType::uint2A8, {"vec2", "readVec2FromUint2A8", R"(
(uint64_t addr) {
    // alignment 8, do not normalize
    return AlignedUVec2Ref(addr).value;
}
)"}
    },
    ReadFunction{ AttributeType::uint2A4Norm, {"vec2", "readVec2FromUint2A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec2(float(v.x) / 0xffffffff, float(v.y) / 0xffffffff);
}
)"}
    },
    ReadFunction{ AttributeType::uint2A4, {"vec2", "readVec2FromUint2A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    return UnalignedUVec2Ref(addr).value;
}
)"}
    },
    // int2
    ReadFunction{ AttributeType::int2A8Norm, {"vec2", "readVec2FromInt2A8Norm", R"(
(uint64_t addr) {
    // alignment 8, normalize
    ivec2 v = AlignedIVec2Ref(addr).value;
    return max(vec2(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff),
                   -1.);
}
)"}
    },
    ReadFunction{ AttributeType::int2A8, {"vec2", "readVec2FromInt2A8", R"(
(uint64_t addr) {
    // alignment 8, do not normalize
    return AlignedIVec2Ref(addr).value;
}
)"}
    },
    ReadFunction{ AttributeType::int2A4Norm, {"vec2", "readVec2FromInt2A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    ivec2 v = UnalignedIVec2Ref(addr).value;
    return max(vec2(float(v.x) / 0x7fffffff, float(v.y) / 0x7fffffff),
                   -1.);
}
)"}
    },
    ReadFunction{ AttributeType::int2A4, {"vec2", "readVec2FromInt2A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    return UnalignedIVec2Ref(addr).value;
}
)"}
    },
    // ushort2
    ReadFunction{ AttributeType::ushort2A4Norm, {"vec2", "readVec2FromUshort2A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm2x16(v);
}
)"}
    },
    ReadFunction{ AttributeType::ushort2A4, {"vec2", "readVec2FromUshort2A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return vec2(v & 0xffff, v >> 16);
}
)"}
    },
    ReadFunction{ AttributeType::ushort2A4NormOffset2, {"vec2", "readVec2FromUshort2A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 16) | (v[1] << 16);
    return unpackUnorm2x16(v[0]);
}
)"}
    },
    ReadFunction{ AttributeType::ushort2A4Offset2, {"vec2", "readVec2FromUshort2A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec2(v[0] >> 16, v[1] & 0xffff);
}
)"}
    },
    // short2
    ReadFunction{ AttributeType::short2A4Norm, {"vec2", "readVec2FromShort2A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm2x16(v);
}
)"}
    },
    ReadFunction{ AttributeType::short2A4, {"vec2", "readVec2FromShort2A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    ivec2 r = ivec2(int(v & 0xffff), int(v >> 16));
    r |= 0xffff0000 * (r >> 15);
    return vec2(r);
}
)"}
    },
    ReadFunction{ AttributeType::short2A4NormOffset2, {"vec2", "readVec2FromShort2A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return unpackSnorm2x16(v[0]);
}
)"}
    },
    ReadFunction{ AttributeType::short2A4Offset2, {"vec2", "readVec2FromShort2A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    ivec2 r = ivec2(int(v[0] >> 16), int(v[1] & 0xffff));
    r |= 0xffff0000 * (r >> 15);
    return vec2(r);
}
)"}
    },
    // ubyte
    ReadFunction{ AttributeType::ubyte2A4Norm, {"vec2", "readVec2FromUbyte2A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).xy;
}
)"}
    },
    ReadFunction{ AttributeType::ubyte2A4, {"vec2", "readVec2FromUbyte2A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return vec2(v & 0xff, (v >> 8) & 0xff);
}
)"}
    },
    ReadFunction{ AttributeType::ubyte2A4NormOffset1, {"vec2", "readVec2FromUbyte2A4NormOffset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).yz;
}
)"}
    },
    ReadFunction{ AttributeType::ubyte2A4Offset1, {"vec2", "readVec2FromUbyte2A4Offset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1 do not normalize
    uint v = AlignedUIntRef(addr).value;
    return vec2((v >> 8) & 0xff, (v >> 16) & 0xff);
}
)"}
    },
    ReadFunction{ AttributeType::ubyte2A4NormOffset2, {"vec2", "readVec2FromUbyte2A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).zw;
}
)"}
    },
    ReadFunction{ AttributeType::ubyte2A4Offset2, {"vec2", "readVec2FromUbyte2A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2 do not normalize
    uint v = AlignedUIntRef(addr).value;
    return vec2((v >> 16) & 0xff, (v >> 24) & 0xff);
}
)"}
    },
    ReadFunction{ AttributeType::ubyte2A4NormOffset3, {"vec2", "readVec2FromUbyte2A4NormOffset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 24) | ((v[1] & 0xff) << 8);
    return unpackUnorm4x8(v[0]).xy;
}
)"}
    },
    ReadFunction{ AttributeType::ubyte2A4Offset3, {"vec2", "readVec2FromUbyte2A4Offset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec2(v[0] >> 24, v[1] & 0xff);
}
)"}
    },
    // byte
    ReadFunction{ AttributeType::byte2A4Norm, {"vec2", "readVec2FromByte2A4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm4x8(v).xy;
}
)"}
    },
    ReadFunction{ AttributeType::byte2A4, {"vec2", "readVec2FromByte2A4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    ivec2 r = ivec2(int(v & 0xff), int((v >> 8) & 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec2(r);
}
)"}
    },
    ReadFunction{ AttributeType::byte2A4NormOffset1, {"vec2", "readVec2FromByte2A4NormOffset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).yz;
}
)"}
    },
    ReadFunction{ AttributeType::byte2A4Offset1, {"vec2", "readVec2FromByte2A4Offset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1 do not normalize
    uint v = AlignedUIntRef(addr).value;
    ivec2 r = ivec2(int((v >> 8) & 0xff), int((v >> 16) & 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec2(r);
}
)"}
    },
    ReadFunction{ AttributeType::byte2A4NormOffset2, {"vec2", "readVec2FromByte2A4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).zw;
}
)"}
    },
    ReadFunction{ AttributeType::byte2A4Offset2, {"vec2", "readVec2FromByte2A4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2 do not normalize
    uint v = AlignedUIntRef(addr).value;
    ivec2 r = ivec2(int((v >> 16) & 0xff), int((v >> 24) & 0xff));
    r |= 0xffffff00 * (r >> 7);
    return vec2(r);
}
)"}
    },
    ReadFunction{ AttributeType::byte2A4NormOffset3, {"vec2", "readVec2FromByte2A4NormOffset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    v[0] = (v[0] >> 24) | ((v[1] & 0xff) << 8);
    return unpackUnorm4x8(v[0]).xy;
}
)"}
    },
    ReadFunction{ AttributeType::byte2A4Offset3, {"vec2", "readVec2FromByte2A4Offset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, do not normalize
    uvec2 v = UnalignedUVec2Ref(addr).value;
    return vec2(v[0] >> 24, (v[1] & 0xff) << 8);
}
)"}
    },
    // float
    ReadFunction{ AttributeType::floatA8, {"float", "readFloatFromFloatA8", R"(
(uint64_t addr) {
    return AlignedFloatRef(addr).value;
}
)"}
	},
    // half
    ReadFunction{ AttributeType::halfA4, {"float", "readFloatFromHalfA4", R"(
(uint64_t addr) {
    uint v = AlignedUIntRef(addr).value;
    return unpackHalf2x16(v).x;
}
)"}
    },
    ReadFunction{ AttributeType::halfA4Offset2, {"float", "readFloatFromHalfA4Offset2", R"(
(uint64_t addr) {
    uint v = AlignedUIntRef(addr).value;
    return unpackHalf2x16(v).y;
}
)"}
    },
    // uint
    ReadFunction{ AttributeType::uintA4Norm, {"float", "readFloatFromUintA4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return float(v) / 0xffffffff;
}
)"}
    },
    ReadFunction{ AttributeType::uintA4, {"float", "readFloatFromUintA4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    return AlignedUIntRef(addr).value;
}
)"}
    },
    // int
    ReadFunction{ AttributeType::intA4Norm, {"float", "readFloatFromIntA4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    int v = AlignedIntRef(addr).value;
    return max(float(v) / 0x7fffffff, -1.);
}
)"}
    },
    ReadFunction{ AttributeType::intA4, {"float", "readFloatFromIntA4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    return AlignedIntRef(addr).value;
}
)"}
    },
    // ushort
    ReadFunction{ AttributeType::ushortA4Norm, {"float", "readFloatFromUshortA4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm2x16(v).x;
}
)"}
    },
    ReadFunction{ AttributeType::ushortA4, {"float", "readFloatFromUshortA4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return float(v & 0xffff);
}
)"}
    },
    ReadFunction{ AttributeType::ushortA4NormOffset2, {"float", "readFloatFromUshortA4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm2x16(v).y;
}
)"}
    },
    ReadFunction{ AttributeType::ushortA4Offset2, {"float", "readFloatFromUshortA4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return float(v >> 16);
}
)"}
    },
    // short
    ReadFunction{ AttributeType::shortA4Norm, {"float", "readFloatFromShortA4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm2x16(v).x;
}
)"}
    },
    ReadFunction{ AttributeType::shortA4, {"float", "readFloatFromShortA4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    int r = int(v & 0xffff);
    r |= 0xffff0000 * (r >> 15);
    return r;
}
)"}
    },
    ReadFunction{ AttributeType::shortA4NormOffset2, {"float", "readFloatFromShortA4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm2x16(v).y;
}
)"}
    },
    ReadFunction{ AttributeType::shortA4Offset2, {"float", "readFloatFromShortA4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, do not normalize
    uint v = AlignedUIntRef(addr).value;
    int r = int(v >> 16);
    r |= 0xffff0000 * (r >> 15);
    return r;
}
)"}
    },
    // ubyte
    ReadFunction{ AttributeType::ubyteA4Norm, {"float", "readFloatFromUbyteA4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).x;
}
)"}
    },
    ReadFunction{ AttributeType::ubyteA4, {"float", "readFloatFromUbyteA4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return float(v & 0xff);
}
)"}
    },
    ReadFunction{ AttributeType::ubyteA4NormOffset1, {"float", "readFloatFromUbyteA4NormOffset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).y;
}
)"}
    },
    ReadFunction{ AttributeType::ubyteA4Offset1, {"float", "readFloatFromUbyteA4Offset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1 do not normalize
    uint v = AlignedUIntRef(addr).value;
    return float((v >> 8) & 0xff);
}
)"}
    },
    ReadFunction{ AttributeType::ubyteA4NormOffset2, {"float", "readFloatFromUbyteA4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).z;
}
)"}
    },
    ReadFunction{ AttributeType::ubyteA4Offset2, {"float", "readFloatFromUbyteA4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2 do not normalize
    uint v = AlignedUIntRef(addr).value;
    return float((v >> 16) & 0xff);
}
)"}
    },
    ReadFunction{ AttributeType::ubyteA4NormOffset3, {"float", "readFloatFromUbyteA4NormOffset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).w;
}
)"}
    },
    ReadFunction{ AttributeType::ubyteA4Offset3, {"float", "readFloatFromUbyteA4Offset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return float(v >> 24);
}
)"}
    },
    // byte
    ReadFunction{ AttributeType::byteA4Norm, {"float", "readFloatFromByteA4Norm", R"(
(uint64_t addr) {
    // alignment 4, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackSnorm4x8(v).x;
}
)"}
    },
    ReadFunction{ AttributeType::byteA4, {"float", "readFloatFromByteA4", R"(
(uint64_t addr) {
    // alignment 4, do not normalize
    uint v = AlignedUIntRef(addr).value;
    int r = int(v & 0xff);
    r |= 0xffffff00 * (r >> 7);
    return float(r);
}
)"}
    },
    ReadFunction{ AttributeType::byteA4NormOffset1, {"float", "readFloatFromByteA4NormOffset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).y;
}
)"}
    },
    ReadFunction{ AttributeType::byteA4Offset1, {"float", "readFloatFromByteA4Offset1", R"(
(uint64_t addr) {
    // alignment 4, offset +1 do not normalize
    uint v = AlignedUIntRef(addr).value;
    int r = int((v >> 8) & 0xff);
    r |= 0xffffff00 * (r >> 7);
    return float(r);
}
)"}
    },
    ReadFunction{ AttributeType::byteA4NormOffset2, {"float", "readFloatFromByteA4NormOffset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).z;
}
)"}
    },
    ReadFunction{ AttributeType::byteA4Offset2, {"float", "readFloatFromByteA4Offset2", R"(
(uint64_t addr) {
    // alignment 4, offset +2 do not normalize
    uint v = AlignedUIntRef(addr).value;
    int r = int((v >> 16) & 0xff);
    r |= 0xffffff00 * (r >> 7);
    return float(r);
}
)"}
    },
    ReadFunction{ AttributeType::byteA4NormOffset3, {"float", "readFloatFromByteA4NormOffset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, normalize
    uint v = AlignedUIntRef(addr).value;
    return unpackUnorm4x8(v).w;
}
)"}
    },
    ReadFunction{ AttributeType::byteA4Offset3, {"float", "readFloatFromByteA4Offset3", R"(
(uint64_t addr) {
    // alignment 4, offset +3, do not normalize
    uint v = AlignedUIntRef(addr).value;
    return float(v >> 24);
}
)"}
    },
};

static constexpr bool AssertReadFunctions() {
    for (size_t i = 0; i < ReadFunctions.size(); ++i) {
        if (i != static_cast<size_t>(ReadFunctions[i].type)) {
            return false;
        }
    }
    return true;
}
static_assert(AssertReadFunctions());

static void generateReadFuncGroup(OutputStream &output, const std::string_view type) {
    for (const auto &r : ReadFunctions) {
        if (r.read.type == type) {
            std::stringstream value;
            value << "0x" << std::hex << std::setw(4) << std::setfill('0') << (static_cast<int>(r.type) << 8);

            output() << "    if(type == " << value.str() << ")\n";
            output() << "        return " << r.read.name << "(addr);\n";
        }
    }
}


static void generateReadFunctions(OutputStream &output, const ShaderState &state) {
    std::set<uint32_t> types;
    for (size_t i = 0; i < state.numAttributes; ++i) {
        const auto &type = AttributeInfo(state.attribAccessInfo[i]).type;
        if (type != 0) {
            types.insert(AttributeInfo(state.attribAccessInfo[i]).type);
        }
    }

    output << ReadBufferReferences;
    for (const auto &i : types) {
        assert(i < ReadFunctions.size());
        const auto &func = ReadFunctions[i].read;
        output() << func.type << " " << func.name << func.code;
    }
}

static void generateOptimizedRead(OutputStream &output, const AttributeInfo &attrib, const std::string_view address, const std::string_view defaultValue)
{
    if (attrib.type >= ReadFunctions.size()) {
        output << defaultValue;
        return;
    }
    const auto &func = ReadFunctions[attrib.type].read;
    output() << func.name << "(" << address << " + " << std::to_string(attrib.offset) << ")";
}

static void generateOptimizedIf(OutputStream &output, bool optimized, bool evaluation, const std::string_view condition, const Expression &trueBody, const Expression &falseBody = nullptr)
{
    if (!optimized) {
        output() << "if(" << condition << ") {\n";
        output.push();
    }
    if (!optimized || evaluation) {
        trueBody.generate(output);
    }
    if (!optimized) {
        output.pop();
        output << "}\n";
        if (falseBody) {
            output << "else {\n";
            output.push();
        }
    }
    if (falseBody && (!optimized || !evaluation)) {
        falseBody.generate(output);
    }
    if (!optimized && falseBody) {
        output.pop();
        output << "}\n";
    }
}


static void generateUberShaderReadFuncs(OutputStream &output)
{
    output << ReadBufferReferences;

    for (size_t i = 1; i < ReadFunctions.size(); ++i) {
        const auto &func = ReadFunctions[i].read;
        output() << func.type << " " << func.name << func.code;
    }

    output << R"(
//
//  read float
//
float readFloat(uint64_t vertexDataPtr, uint settings)
{
    uint offset = settings & 0x00ff;  // max offset is 255
    uint64_t addr = vertexDataPtr + offset;
    uint type = settings & 0xff00;
)";
    generateReadFuncGroup(output, "float");
    output << R"(
    // return NaN
    return float(0/0);
}
)";

    output << R"(
//
//  read vec2
//
vec2 readVec2(uint64_t vertexDataPtr, uint settings)
{
    uint offset = settings & 0x00ff;  // max offset is 255
    uint64_t addr = vertexDataPtr + offset;
    uint type = settings & 0xff00;
)";
    generateReadFuncGroup(output, "vec2");
    output << R"(
    // return NaN
    return vec2(0/0);
}
)";

    output << R"(
//
//  read vec3
//
vec3 readVec3(uint64_t vertexDataPtr, uint settings)
{
    uint offset = settings & 0x00ff;  // max offset is 255
    uint64_t addr = vertexDataPtr + offset;
    uint type = settings & 0xff00;
)";
    generateReadFuncGroup(output, "vec3");
    output << R"(
    // return NaN
    return vec3(0/0);
}
)";

    output << R"(
//
//  read vec4
//
vec4 readVec4(uint64_t vertexDataPtr, uint settings)
{
    uint offset = settings & 0x00ff;  // max offset is 255
    uint64_t addr = vertexDataPtr + offset;
    uint type = settings & 0xff00;
)";
    generateReadFuncGroup(output, "vec4");
    output << R"(
    // try readVec3
    return vec4(readVec3(vertexDataPtr, settings), 1);
}
)";

}

static void generateProjectionConstants(OutputStream &output)
{
    output << R"(
// projection matrix specialization constants
// (projectionMatrix members that do not depend on zNear and zFar clipping planes)
layout(constant_id = 0) const float p31 = 0.;
layout(constant_id = 1) const float p32 = 0.;
layout(constant_id = 2) const float p34 = 0.;
layout(constant_id = 3) const float p41 = 0.;
layout(constant_id = 4) const float p42 = 0.;
layout(constant_id = 5) const float p44 = 1.;
)";
}

static void generateSceneDataInterface(OutputStream &output)
{
    output << R"(
//
// scene data
//

layout(buffer_reference, std430, buffer_reference_align=64) restrict readonly buffer
SceneDataRef {
	mat4 viewMatrix;        // current camera view matrix
	mat4 projectionMatrix;  // current camera projection matrix
	float p11,p22,p33,p43;  // alternative specification of projectionMatrix - only members that depend on zNear
	                        // and zFar clipping planes; remaining members are passed in as specialization constants
	vec3 ambientLight;      // scene ambient light
    uint numLights;
    vec3 cameraPos;
	layout(offset=192) uint lightData[];  // array of OpenGLLight and GltfLight structures is stored here
};
uint getLightDataOffset()  { return 192; }
)";
}

static void generatePushConstants(OutputStream &output, bool optimizeAttribs, bool idBuffer)
{
    output << "// push constants\n"
           << "layout(push_constant) uniform pushConstants {\n";
    output.push();
    output << "layout(offset=0) uint64_t sceneDataPtr;  // pointer to SceneDataRef; usually updated per scene render pass or once per scene rendering\n"
           << "layout(offset=8) uint64_t drawablePointersBufferPtr;  // pointer to DrawablePointersRef array; there is one DrawablePointersRef array for each StateSet, so the pointer is updated before each StateSet rendering\n";
    // if (!optimizeAttribs) {
        output << "layout(offset=16) uint attribAccessInfoList[8];  // per-stateSet attribAccessInfo for 16 attribs\n"
               << "layout(offset=48) uint attribSetup;  // doc is provided bellow with getVertexDataSize()\n"
               << "layout(offset=52) uint materialSetup;  // doc is provided with UnlitMaterialRef, PhongMaterialRef and MetallicRoughnessMaterialRef\n";
    // }
    if (idBuffer) {
        output << "layout(offset=56) uint stateSetID;  // ID of the current StateSet\n";
    }
    output.pop();
    output << "};";
}

static void includeVertexDrawableInterface(OutputStream &output)
{
    output << R"(
//
// drawable data
//

layout(buffer_reference, std430, buffer_reference_align=64) restrict readonly buffer
MatrixListRef {
    mat4 matrices[];
};

layout(buffer_reference, std430, buffer_reference_align=64) restrict readonly buffer
MatrixRef {
    mat4 matrix;
};

// indices
layout(buffer_reference, std430, buffer_reference_align=4) restrict readonly buffer
IndexDataRef {
    uint indices[];
};

const uint ModelMatrixOffset = 64;
mat4 getDrawableMatrix(uint64_t matrixListPtr, uint index) {
    return MatrixRef(matrixListPtr + ModelMatrixOffset + index * 64).matrix;
}

// drawable data pointers
layout(buffer_reference, std430, buffer_reference_align=8) restrict readonly buffer DrawablePointersRef {
 	uint64_t vertexDataPtr;
	uint64_t indexDataPtr;
	uint64_t matrixListPtr;
	uint64_t drawableDataPtr;
};
const uint DrawablePointersSize = 32;
)";
}

static void includeUberShaderInterface(OutputStream &output, bool optimizeMaterial = false, bool optimizeTextures = false, bool optimizeAttribs = false, bool idBuffer = false)
{
    generateSceneDataInterface(output);
    generatePushConstants(output, optimizeAttribs, idBuffer);

    if (!optimizeAttribs) {
        output << R"(
// pushConstants.attribAccessInfoList
uint getPositionAccessInfo()  { return attribAccessInfoList[0] & 0xffff; }
uint getNormalAccessInfo()  { return attribAccessInfoList[0] >> 16; }
uint getTangentAccessInfo()  { return attribAccessInfoList[1] & 0xffff; }
uint getColorAccessInfo()  { return attribAccessInfoList[1] >> 16; }
uint getTexCoordAccessInfo(uint attribIndex) { uint texCoordAccessInfo = attribAccessInfoList[attribIndex>>1]; if((attribIndex & 0x1) == 0) texCoordAccessInfo &= 0x0000ffff; else texCoordAccessInfo >>= 16; return texCoordAccessInfo; }

// pushConstants.attribSetup
// bit 2..8: vertex data size (0, 4, 8,..., 508)
uint getVertexDataSize()  { return attribSetup & 0x01fc; }
bool getGenerateFlatNormals()  { return (attribSetup & 0x0001) != 0; }
)";
    }

    if (!optimizeMaterial) {
        output << R"(
// materialSetup
// bits 0..1: 0 - reserved, 1 - unlit, 2 - phong, 3 - metallicRoughness
// bits 2..7: texture offset (0, 4, 8, 12, .....252)
// bit 8: use color attribute for ambient and diffuse; material ambient and diffuse values are ignored
// bit 9: use color attribute for diffuse; material diffuse value is ignored
// bit 10: ignore color attribute alpha if color attribute is used (if bit 8 or 9 is set)
// bit 11: ignore material alpha
// bit 12: ignore base texture alpha if base texture is used
// bit 13: use sheen material
uint getMaterialModel()  { return materialSetup & 0x03; }
uint getMaterialFirstTextureOffset()  { return materialSetup & 0xfc; }
bool getMaterialUseColorAttribute()  { return (materialSetup & 0x0300) != 0; }
bool getMaterialUseColorAttributeForAmbientAndDiffuse()  { return (materialSetup & 0x0100) != 0; }
bool getMaterialUseColorAttributeForDiffuseOnly()  { return (materialSetup & 0x0200) != 0; }
bool getMaterialIgnoreColorAttributeAlpha()  { return (materialSetup & 0x0400) != 0; }
bool getMaterialIgnoreMaterialAlpha()  { return (materialSetup & 0x0800) != 0; }
bool getMaterialIgnoreBaseTextureAlpha()  { return (materialSetup & 0x1000) != 0; }
bool getMaterialUseSheen()  {  return (materialSetup & 0x2000) != 0; }
)";
    }

    output << R"(
//
// material structures
//

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
UnlitMaterialRef {
	layout(offset=0) vec4 colorAndAlpha;
	layout(offset=16)  float pointSize;
	// [... texture data starts at offset 20 ...]
};

layout(buffer_reference, std430, buffer_reference_align=16) restrict readonly buffer
PhongMaterialRef {
	layout(offset=0)  vec3 ambient;  //< ambient color might be ignored and replaced by diffuse color when specified by settings
    layout(offset=12)  uint pad1;
	layout(offset=16) vec4 diffuseAndAlpha;  //< Hoops uses four floats for diffuse and alpha in MaterialKit
	layout(offset=32) vec3 specular;  //< Hoops uses specular color in MaterialKit
	layout(offset=44) float shininess;  //< Hoops uses gloss (1x float) in MaterialKit
	layout(offset=48) vec3 emission;  //< Hoops uses 4x float in MaterialKit
	layout(offset=60) float pointSize;
	layout(offset=64) vec3 reflection;  //< Hoops uses mirror in MaterialKit
	// [... texture data starts at offset 76 ...]
};

layout(buffer_reference, std430, buffer_reference_align=64) restrict readonly buffer
MetallicRoughnessMaterialRef {
    // settings
    // bits 0..1: unlit, phong, metallicRoughness
    // bit 6..8: modelMatrix offset (0, 64, 128, 192,..., 448)
    // bit 9: use baseTexture
    layout(offset=0)  uvec4 settings;  // includes baseTextureCoordIndex, metallicRoughnessTextureCoordIndex,
                                       // normalTextureCoordIndex, occlusionTextureCoordIndex,
                                       // emissiveTextureCoordIndex, alphaMode, doubleSided, unlit,
                                       // from extensions: anisotropyTextureCoordIndex
    layout(offset=16) vec4 baseColorFactor;  //< Hoops uses alpha and baseColor (in PBRMaterialKit)
    float metallicFactor;  //< Hoops uses metalnessFactor (in PBRMaterialKit)
    float roughnessFactor;  //< Hoops uses RoughnessFactor (in PBRMaterialKit)
    float ior;
	float alphaCutoff;
	vec3 reflectance;
	float pointSize;
	vec3 emissiveFactor;
	// 24 additional floats (100 bytes)
	float dispersion;
	float anisotropyStrength;
	float anisotropyRotation;
	float clearcoatFactor;
	float clearcoatRoughnessFactor;
	float iridescenceFactor;
	float iridescenceIor;
	float iridescenceThicknessMinimum;
	float iridescenceThicknessMaximum;
	vec3 sheenColorFactor;
	float sheenRoughnessFactor;
	float specularFactor;
	vec3 specularColorFactor;
	float transmissionFactor;
	float thicknessFactor;
	float attenuationDistance;
	vec3 attenuationColor;
};

layout(buffer_reference, std430, buffer_reference_align=8) restrict readonly buffer
TextureInfoRef {
	// texCoordIndex - bits 0..7
	// type - bits 8..15
	//   0 - marks the end of TextureInfo array; TextureInfo array is ended by
	//       texCoordIndexTypeAndSettings set to zero
	//   1 - normal texture
	//   2 - occusion texture
	//   3 - emissive texture
	//   4 - base texture
	// settings - bits 16..31
	//   bit 16 - multiply texture value by strength member; if neither bit 16 nor bit 17 is set,
	//            the TextureInfoRef structure occupies only 8 bytes and following TextureInfo
	//            structure is placed on the address incremented by 8 bytes only
	//   bit 17 - transform texture coordinates by the transformation specified by t1..t6;
	//            if bits 16 and 17 are not set, the structure occupies only 8 bytes;
	//            if bit 16 is set and bit 17 not, the structure occupies 12 bytes; otherwise
	//            it occupies 36 bytes; these sizes should be used to compute address
	//            of the next TextureInfo structure
	//   bit 18 - blend color included in the structure
	//   bits 19..21 - for Phong and its base texture, texture environment:
	//                 0 - modulate, 1 - replace, 2 - decal, 3 - blend, 4 - add
	//   bits 22..23 - first component index;
	//   bits 26..31 - size of the structure (0..63 bytes), it must be multiple of 8
	uint texCoordIndexTypeAndSettings;
	uint textureIndex;
	float strength;
	float rs1,rs2,rs3,rs4,t1,t2;  // rotation and scale in 2x2 matrix, translation in vec2
	float blendR,blendG,blendB;  // texture blend color used in blend texture environment
};

)";

    if (!optimizeTextures) {
        output << R"(
bool getTextureUseStrengthFlag(TextureInfoRef textureInfo)  { return (textureInfo.texCoordIndexTypeAndSettings & 0x10000) != 0; }
uint getTextureEnvironment(TextureInfoRef textureInfo)  { return (textureInfo.texCoordIndexTypeAndSettings >> 19) & 0x7; }
vec3 getTextureBlendColor(TextureInfoRef textureInfo)  { return vec3(textureInfo.blendR, textureInfo.blendG, textureInfo.blendB); }
uint getTextureFirstComponentIndex(TextureInfoRef textureInfo)  { return (textureInfo.texCoordIndexTypeAndSettings >> 22) & 0x3; }

TextureInfoRef getNextTextureInfo(TextureInfoRef textureInfo)
{
	uint size = textureInfo.texCoordIndexTypeAndSettings >> 26;
	return TextureInfoRef(uint64_t(textureInfo) + size);
}

)";
    }
    if (!optimizeTextures || !optimizeAttribs) {
        output << R"(
uint getTextureCoordIndex(TextureInfoRef textureInfo) { return textureInfo.texCoordIndexTypeAndSettings & 0xff; }
)";
    }
    if (!optimizeAttribs) {
        output << R"(
bool getTextureTranformFlag(TextureInfoRef textureInfo)  { return (textureInfo.texCoordIndexTypeAndSettings & 0x20000) != 0; }
uint getTexCoordAccessInfo(TextureInfoRef textureInfo) { return getTexCoordAccessInfo(textureInfo.texCoordIndexTypeAndSettings & 0xff); }

vec2 transformTexCoord(vec2 tc, TextureInfoRef textureInfo)
{
    mat2x2 rotationAndScale = {
        { textureInfo.rs1, textureInfo.rs2, },
        { textureInfo.rs3, textureInfo.rs4, },
    };
    vec2 translation = { textureInfo.t1, textureInfo.t2, };
    return rotationAndScale * tc + translation;
}

vec2 computeTextureCoordinates(TextureInfoRef textureInfo, uint64_t vertex0DataPtr,
    uint64_t vertex1DataPtr, uint64_t vertex2DataPtr, vec3 barycentricCoords)
{
    // get texture coordinates
    uint texCoordAccessInfo = getTexCoordAccessInfo(textureInfo);
    vec2 uv0 = readVec2(vertex0DataPtr, texCoordAccessInfo);
    vec2 uv1 = readVec2(vertex1DataPtr, texCoordAccessInfo);
    vec2 uv2 = readVec2(vertex2DataPtr, texCoordAccessInfo);
    vec2 uv = uv0 * barycentricCoords.x + uv1 * barycentricCoords.y +
              uv2 * barycentricCoords.z;

    // transform texture coordinates
    // if(getTextureTranformFlag(textureInfo))
    //   uv = transformTexCoord(uv, textureInfo);

    return uv;
}
)";
    }

    output << R"(
//
// light source
//

struct OpenGLLightData {
    vec3 ambient;
    float constantAttenuation;
    vec3 diffuse;
    float linearAttenuation;
    vec3 specular;
    float quadraticAttenuation;
};

struct GltfLightData {
    vec3 color;
    float intensity;  // in candelas (lm/sr) for point light and spotlight and in luxes (lm/m2) for directional light
    float range;
};

struct SpotlightData {
    vec3  direction;  // spotlight direction in eye coordinates, it must be normalized
    float cosOuterConeAngle;  // cosinus of outer spotlight cone; outside the cone, there is zero light intensity
    float cosInnerConeAngle;  // cosinus of inner spotlight cone; if -1. is provided, OpenGL-style spotlight is used, ignoring inner cone and using spotExponent instead; if value is > -1., DirectX style spotlight is used, e.g. everything inside the inner cone receives full light intensity and light intensity between inner and outer cone is linearly interpolated starting from zero intensity on outer code to full intensity in inner cone
    float spotExponent;  // if cosInnerConeAngle is -1, OpenGL style spotlight is used, using spotExponent
};

layout(buffer_reference, std430, buffer_reference_align=64) restrict readonly buffer
LightRef {
    layout(offset=0)  vec3 positionOrDirection;  // for point light and spotlight: position in eye coordinates,
                                                 // for directional light: direction in eye coordinates, direction must be normalized
    uint settings;  // switches between point light, directional light and spotlight
    layout(offset=16) OpenGLLightData opengl;
    layout(offset=64) GltfLightData gltf;
    layout(offset=96) SpotlightData spotlight;
};
uint getLightDataSize()  { return 128; }
)";

    includeVertexDrawableInterface(output);
}

struct MetadataMaterial
{
    bool optimizeMaterialModel = false;
    bool optimizeMaterialColorAttribute = false;
    bool optimizeMaterialAlpha = false;

    uint32_t materialModel = 0;
    uint32_t materialFirstTextureOffset = 0;
    bool materialUseColorAttribute = false;
    bool materialUseColorAttributeForAmbientAndDiffuse = false;
    bool materialUseColorAttributeForDiffuseOnly = false;
    bool materialIgnoreColorAttributeAlpha = false;
    bool materialIgnoreMaterialAlpha = false;
    bool materialIgnoreBaseTextureAlpha = false;
    bool materialUseSheen = false;

    explicit MetadataMaterial(const ShaderState& state)
        : optimizeMaterialModel(state.optimizeFlags.to_ulong() & ShaderState::OptimizeMaterialModel.to_ulong())
        , optimizeMaterialColorAttribute(state.optimizeFlags.to_ulong() & ShaderState::OptimizeMaterialColorAttribute.to_ulong())
        , optimizeMaterialAlpha(state.optimizeFlags.to_ulong() & ShaderState::OptimizeMaterialAlpha.to_ulong())
    {

        if (optimizeMaterialModel) {
            materialModel = state.materialSetup & 0x03;
            materialFirstTextureOffset = state.materialSetup & 0xfc;
            materialUseSheen = state.materialSetup & 0x2000;
        }
        if (optimizeMaterialColorAttribute) {
            materialUseColorAttribute = state.materialSetup & 0x0300;
            materialUseColorAttributeForAmbientAndDiffuse = state.materialSetup & 0x0100;
            materialUseColorAttributeForDiffuseOnly = state.materialSetup & 0x0200;
        }
        if (optimizeMaterialAlpha) {
            materialIgnoreColorAttributeAlpha = state.materialSetup & 0x0400;
            materialIgnoreMaterialAlpha = state.materialSetup & 0x0800;
            materialIgnoreBaseTextureAlpha = state.materialSetup & 0x1000;
        }
    }
};

struct MetadataAttribs {
    bool optimizeAttribs = false;
    bool colorAttrib = false;
    bool colorAttribAlpha = false;
    bool getGenerateFlatNormals = false;
    uint32_t vertexSize = 0;
    uint8_t numTextureAttribs = 0;

    explicit MetadataAttribs(const ShaderState& state)
        : optimizeAttribs(state.optimizeFlags.to_ulong() & ShaderState::OptimizeAttribs.to_ulong())
    {
        if (optimizeAttribs) {
            colorAttrib = state.attribAccessInfo[3] != 0;

            if (colorAttrib) {
                auto colorType = static_cast<AttributeType>(state.attribAccessInfo[3] >> 8);
                colorAttribAlpha = colorType == AttributeType::vec4A16 || colorType == AttributeType::ubyte4A4Norm;
                if (!colorAttribAlpha && !(colorType == AttributeType::vec3A16 || colorType == AttributeType::ubyte3A4NormFirst3)) {
                    throw std::runtime_error("Unsupported color attribute type");
                }
            }

            vertexSize = state.attribSetup & 0x01fc;
            getGenerateFlatNormals = state.attribSetup & 0x0001;

            for (size_t i = 4; i < state.attribAccessInfo.size(); ++i) {
                if (state.attribAccessInfo[i] != 0) {
                    numTextureAttribs++;
                }
            }

        }
    }

    void generateFragmentInputInterface(OutputStream &output, bool idBuffer)
    {
        int location;
        if (!optimizeAttribs) {
            output() << "layout(location = 0) in flat u64vec4 inVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3; it occupies locations 0 and 1\n";
            location = 2;
        }
        else {
            output() << "layout(location = 0) in flat uint64_t inDrawableDataPtr;\n";
            location = 1;
        }

        if (!optimizeAttribs) {
            output() << "layout(location = " << location++ << ") in smooth vec3 inBarycentricCoords;  // barycentric coordinates using perspective correction\n";
        }
        output() << "layout(location = " << location++ << ") in smooth vec3 inFragmentPosition3;  // in eye coordinates\n";
        output() << "layout(location = " << location++ << ") in smooth vec3 inFragmentNormal;  // in eye coordinates\n";
        output() << "layout(location = " << location++ << ") in smooth vec3 inFragmentTangent;  // in eye coordinates\n";

        if (colorAttrib) {
            output() << "layout(location = " << location++ << ") in smooth " << (colorAttribAlpha? "vec4" : "vec3") << " inFragmentColor;  // color\n";
        }
        if (numTextureAttribs > 0) {
            output() << "layout(location = " << location++ << ") in smooth vec2 inTexCoords[" << static_cast<uint32_t>(numTextureAttribs) << "];  // texture coordinates\n";
        }
        if (idBuffer) {
            output() << "layout(location = " << location++ << ") in flat uvec2 inId;\n";
        }
    }

    void generateGeometryOutputInterface(OutputStream &output, bool idBuffer)
    {
        output() << "layout(location = 0) out flat u64vec4 outVertexAndDrawableDataPtr;  // VertexData on indices 0..2 and DrawableData on index 3; it occupies locations 0 and 1\n";

        int location = 2;
        if (!optimizeAttribs) {
            output() << "layout(location = " << location++ << ") out smooth vec3 outBarycentricCoords;  // barycentric coordinates using perspective correction\n";
        }

        output() << "layout(location = " << location++ << ") out smooth vec3 outVertexPosition3;  // in eye coordinates\n";
        output() << "layout(location = " << location++ << ") out smooth vec3 outVertexNormal;  // in eye coordinates\n";
        output() << "layout(location = " << location++ << ") out smooth vec3 outVertexTangent;  // in eye coordinates\n";

        if (colorAttrib) {
            output() << "layout(location = " << location++ << ") out smooth " << (colorAttribAlpha? "vec4" : "vec3") << " outVertexColor;  // color\n";
        }
        if (numTextureAttribs > 0) {
            output() << "layout(location = " << location++ << ") out smooth vec2 outTexCoords[" << static_cast<uint32_t>(numTextureAttribs) << "];  // texture coordinates\n";
        }
        if (idBuffer) {
            output() << "layout(location = " << location++ << ") out flat uvec2 outId; // ID_BUFFER\n";
        }
    }

};

// TODO deprecate
struct OptimizedIf
{

    std::string condition;
    std::string prereq;

    void generate(OutputStream& output, bool optimized, bool evaluation, const std::string &body) const {
        if (!optimized) {
            output() << "if(" << condition << ")\n";
        }
        if (!optimized || evaluation) {
            output << "{\n";
            output.push();
            output() << body << "\n";
            output.pop();
            output << "}\n";
        }
    }

};


class Texture
{

    static OptimizedIf strengthConditon;

public:

    uint32_t settings = {};

    explicit operator bool() const noexcept
    {
        return settings != 0;
    }

    size_t getTextureSize() const
    {
        //   bits 26..31 - size of the structure (0..63 bytes), it must be multiple of 8
        uint32_t size = settings >> 26;
        assert(size > 0 && size % 8 == 0 && "Texture has invalid size");
        return size;
    }

    void generateStrengthFlagCode(OutputStream& output, bool optimized, const std::string &body) const
    {
        strengthConditon.generate(output, optimized, getTextureUseStrengthFlag(), body);
    }

    void generateTextureEnvironmentCode(OutputStream& output, bool optimized) const
    {
        if (!optimized) {
            // apply texture using texEnv
            output << "uint texEnv = getTextureEnvironment(textureInfo);\n";
        }
    }

    bool getTextureUseStrengthFlag() const noexcept
    {
        return (settings & 0x10000) != 0;
    }
    bool getTextureTranformFlag() const noexcept
    {
        return (settings & 0x20000) != 0;
    }
    uint32_t getTextureEnvironment() const noexcept
    {
        return (settings >> 19) & 0x7;
    }
    uint32_t getTextureFirstComponentIndex() const noexcept
    {
        return (settings >> 22) & 0x3;
    }
    uint32_t getTexCoordIndex() const noexcept
    {
        return settings & 0xff;
    }

};

OptimizedIf Texture::strengthConditon{"getTextureUseStrengthFlag(textureInfo)", "bool getTextureUseStrengthFlag(TextureInfoRef textureInfo)  { return (textureInfo.texCoordIndexTypeAndSettings & 0x10000) != 0; }\n"};

struct MetadataTextures
{
    std::array<Texture, CadPL::MaxTextures> textures;
    bool optimizeTextures = false;

    explicit MetadataTextures(const ShaderState& state)
        : optimizeTextures(state.optimizeFlags.to_ulong() & ShaderState::OptimizeTextures.to_ulong())
    {
        if (optimizeTextures) {
            assert(state.numTextures < state.textureSetup.size() && "too many textures in ShaderState");

            for (size_t i = 0; i < state.numTextures; ++i) {
                auto type = (state.textureSetup[i] & 0xFF00) >> 8;
                if (type) {
                    assert(type <= textures.size() && "texture index out of range");
                    textures[type - 1].settings = state.textureSetup[i];
                }
            }
        }
    }

    static uint32_t textureCoordIndex(uint32_t textureInfo)
    {
        return textureInfo & 0xff;
    }

    Texture &getTexture(TextureType type)
    {
        auto index = static_cast<int>(type);
        assert(index > 0 && index <= textures.size() && "texture index out of range");
        return textures[index - 1];
    }

};


class VertexShaderGenerator : public MetadataAttribs
{

protected:
    void generateVertexCode(OutputStream &output, const std::string_view vertexPtrValue, const std::string_view barycentricValue, const std::string_view inId, bool vertexShader = false)
    {
        output() << "vertexDataPtr = " << vertexPtrValue << ";\n";
        if (optimizeAttribs) {
            output() << "position = ";
            generateOptimizedRead(output, state.attribAccessInfo[0], "vertexDataPtr", "vec3(0)");
            output() << ";\n";
        }
        else {
            output << "position = readVec3(vertexDataPtr, positionAccessInfo);\n";
        }

        output << "eyePosition = modelViewMatrix * vec4(position, 1);\n"
               // multiplication by projection "matrix"
               << "gl_Position.x = scene.p11*eyePosition.x + p31*eyePosition.z + p41*eyePosition.w;\n"
               << "gl_Position.y = scene.p22*eyePosition.y + p32*eyePosition.z + p42*eyePosition.w;\n"
               << "gl_Position.z = scene.p33*eyePosition.z + scene.p43*eyePosition.w;\n"
               << "gl_Position.w = p34*eyePosition.z + p44*eyePosition.w;\n";
        // set output variables
        if (!vertexShader) {
            output << "outVertexAndDrawableDataPtr = vertexAndDrawableDataPtr;\n";
        }
        output << "outVertexPosition3 = eyePosition.xyz / eyePosition.w;\n";

        if (!optimizeAttribs) {
            output() << "outBarycentricCoords = " << barycentricValue << ";\n";
        }

        if (optimizeAttribs) {
            if (AttributeInfo(state.attribAccessInfo[1]).type != 0) {
                output()  << "outVertexNormal = normalize(mat3(modelViewMatrix) * ";
                generateOptimizedRead(output, state.attribAccessInfo[1], "vertexDataPtr", "");
                output()<< ");\n";
            }
            else {
                output << "outVertexNormal = vec3(0,0,-1);\n";
            }
            if (AttributeInfo(state.attribAccessInfo[2]).type != 0) {
                output() << "outVertexTangent = normalize(mat3(modelViewMatrix) * ";
                generateOptimizedRead(output, state.attribAccessInfo[2], "vertexDataPtr", "");
                output() << ");\n";
            }
            else {
                output << "outVertexTangent = vec3(1,0,0);\n";
            }
            if (colorAttrib) {
                output() << "outVertexColor = ";
                generateOptimizedRead(output, state.attribAccessInfo[3], "vertexDataPtr", "");
                output() << ";\n";
            }
            for (size_t i = 0; i < numTextureAttribs; ++i) {
                output() << "outTexCoords[" << i << "] = ";
                generateOptimizedRead(output, state.attribAccessInfo[4 + i], "vertexDataPtr", "");
                output() << ";\n";
            }
        }
        else {
            // normal
            output << "if(normalAccessInfo != 0)\n";
            output.push();
            output << "outVertexNormal = normalize(mat3(modelViewMatrix) * readVec3(vertexDataPtr, normalAccessInfo));\n";
            output.pop();
            output << "else\n";
            output.push();
            output << "outVertexNormal = vec3(0,0,-1);\n";
            output.pop();
            // tangent
            output << "if(tangentAccessInfo != 0)\n";
            output.push();
            output << "outVertexTangent = normalize(mat3(modelViewMatrix) * readVec3(vertexDataPtr, tangentAccessInfo));\n";
            output.pop();
            output << "else\n";
            output.push();
            output << "outVertexTangent = vec3(1,0,0);\n";
            output.pop();
        }
        if (state.idBuffer) {
            output << "// ID_BUFFER\n";
            output << "outId = " << inId << ";\n";
        }
        if (!vertexShader) {
            output << "EmitVertex();\n";
        }

    }

public:

    static constexpr const char* ShaderName = "vertex";
    static constexpr shaderc_shader_kind ShaderKind = shaderc_vertex_shader;

    const ShaderState& state;

    explicit VertexShaderGenerator(const ShaderState& state)
        : MetadataAttribs(state)
        , state(state)
    {
    }

    std::string generate()
    {
        OutputStream output;
        output.reserve(CodeStringReservation);

        output() << "#version 460\n";

        if (!optimizeAttribs) {
            output() << "\n// output to geometry shader\n";
            output() << "layout(location = 0) out flat int outDrawIndex;\n";
            output() << "layout(location = 1) out flat int outInstanceIndex;\n";
            output() << "layout(location = 2) out flat int outVertexIndex;\n";

            if (state.idBuffer) {
                output << "// ID_BUFFER\n"
                       << "layout(location = 3) out flat uvec2 outId;\n";
            }

        }
        else {
            output << "#extension GL_EXT_buffer_reference : require\n";
            output << "#extension GL_ARB_gpu_shader_int64 : require\n";

            if (optimizeAttribs) {
                generateReadFunctions(output, state);
            }
            else {
                generateUberShaderReadFuncs(output);
            }
            generateSceneDataInterface(output);
            generatePushConstants(output, optimizeAttribs, state.idBuffer);
            includeVertexDrawableInterface(output);
            generateProjectionConstants(output);

            output << "\n// output to fragment shader\n";
            output << "layout(location = 0) out flat uint64_t outDrawableDataPtr;\n";

            int location = 1;

            output() << "layout(location = " << location++ << ") out smooth vec3 outVertexPosition3;  // in eye coordinates\n";
            output() << "layout(location = " << location++ << ") out smooth vec3 outVertexNormal;  // in eye coordinates\n";
            output() << "layout(location = " << location++ << ") out smooth vec3 outVertexTangent;  // in eye coordinates\n";

            if (colorAttrib) {
                output() << "layout(location = " << location++ << ") out smooth " << (colorAttribAlpha? "vec4" : "vec3") << " outVertexColor;  // color\n";
            }
            if (numTextureAttribs > 0) {
                output() << "layout(location = " << location++ << ") out smooth vec2 outTexCoords[" << static_cast<uint32_t>(numTextureAttribs) << "];  // texture coordinates\n";
            }
            if (state.idBuffer) {
                output() << "layout(location = " << location++ << ") out flat uvec2 outId; // ID_BUFFER\n";
            }

        }

        output() << "void main()\n" << "{\n";
        output.push();
        if (!optimizeAttribs) {
            output << "outDrawIndex = gl_DrawID;\n"
                   << "outInstanceIndex = gl_InstanceIndex;\n"
                   << "outVertexIndex = gl_VertexIndex;\n";
        }
        else {
            output << "DrawablePointersRef dp = DrawablePointersRef(drawablePointersBufferPtr + (gl_DrawID * DrawablePointersSize));\n"
                   << "outDrawableDataPtr = dp.drawableDataPtr;\n"
                   << "IndexDataRef indexData = IndexDataRef(dp.indexDataPtr);\n"
                   << "uint index = indexData.indices[gl_VertexIndex];\n"
                   << "// matrices and positions\n"
                   << "SceneDataRef scene = SceneDataRef(sceneDataPtr);\n"
                   << "mat4 modelMatrix = getDrawableMatrix(dp.matrixListPtr, gl_InstanceIndex);\n"
                   << "mat4 modelViewMatrix = scene.viewMatrix * modelMatrix;\n"
                   << "vec4 eyePosition;\n"
                   << "vec3 position;\n"
                   << "uint64_t vertexDataPtr;\n";
            std::string vertexPtr = "dp.vertexDataPtr + (index * " + std::to_string(vertexSize) + ")";
            generateVertexCode(output, vertexPtr, "vec3(0,0,0)", "gl_DrawID", true);
        }

        if (state.idBuffer) {
            output << "// ID_BUFFER\n"
                   << "outId[0] = gl_DrawID;\n"
                   << "outId[1] = gl_InstanceIndex;\n";
        }
        output.pop();
        output << "}\n";

        return std::move(output.string());
    }

};

class GeometryShaderGenerator : public VertexShaderGenerator
{

public:

    static constexpr const char* ShaderName = "geometry";
    static constexpr shaderc_shader_kind ShaderKind = shaderc_geometry_shader;

    const ShaderState& state;

    explicit GeometryShaderGenerator(const ShaderState& state)
        : VertexShaderGenerator(state)
        , state(state)
    {

    }

    std::string generate()
    {
        OutputStream output;
        output.reserve(CodeStringReservation);

        output << "#version 460\n"
               << "#extension GL_EXT_buffer_reference : require\n"
               << "#extension GL_ARB_gpu_shader_int64 : require\n";

        if (optimizeAttribs) {
            generateReadFunctions(output, state);
        }
        else {
            generateUberShaderReadFuncs(output);
        }
        includeUberShaderInterface(output, false, optimizeAttribs, optimizeAttribs, state.idBuffer);

        output << "layout(triangles) in;\n"
               << "layout(triangle_strip, max_vertices=3) out;\n";
               // input from vertex shader
        output << "layout(location = 0) in flat int inDrawIndex[3];\n"
               << "layout(location = 1) in flat int inInstanceIndex[3];\n"
               << "layout(location = 2) in flat int inVertexIndex[3];\n";
        if (state.idBuffer) {
            output << "// ID_BUFFER\n"
                   << "layout(location = 3) in flat uvec2 inId[3];\n";
        }
        generateGeometryOutputInterface(output, state.idBuffer);

        generateProjectionConstants(output);

        output << "void main()\n" << "{\n";
        output.push();
        // input from vertex shader
        output << "int drawIndex = inDrawIndex[0];\n"
               << "int instanceIndex = inInstanceIndex[0];\n";
        // DrawablePointers
        output << "u64vec4 vertexAndDrawableDataPtr;\n"
               << "DrawablePointersRef dp = DrawablePointersRef(drawablePointersBufferPtr + (drawIndex * DrawablePointersSize));\n";
        // vertex data
        output << "{\n";
        output.push();
        output << "IndexDataRef indexData = IndexDataRef(dp.indexDataPtr);\n"
               << "uint index0 = indexData.indices[inVertexIndex[0]];\n"
               << "uint index1 = indexData.indices[inVertexIndex[1]];\n"
               << "uint index2 = indexData.indices[inVertexIndex[2]];\n"
               << "uint vertexDataSize = getVertexDataSize();\n"
               << "vertexAndDrawableDataPtr.x = dp.vertexDataPtr + (index0 * vertexDataSize);\n"
               << "vertexAndDrawableDataPtr.y = dp.vertexDataPtr + (index1 * vertexDataSize);\n"
               << "vertexAndDrawableDataPtr.z = dp.vertexDataPtr + (index2 * vertexDataSize);\n"
               << "vertexAndDrawableDataPtr.w = dp.drawableDataPtr;\n";
        output.pop();
        output << "}\n";
        // matrices and positions
        output << "SceneDataRef scene = SceneDataRef(sceneDataPtr);\n";
        output << "mat4 modelMatrix = getDrawableMatrix(dp.matrixListPtr, instanceIndex);\n";
        output << "mat4 modelViewMatrix = scene.viewMatrix * modelMatrix;\n";
        // first vertex
        output << "vec4 eyePosition;\n"
               << "vec3 position;\n"
               << "uint64_t vertexDataPtr;\n";

        if (!optimizeAttribs) {
            // vertex positions (it is stored on offset 0 by convention)
            output << "const uint positionAccessInfo = getPositionAccessInfo();\n"
                   << "const uint normalAccessInfo = getNormalAccessInfo();\n"
                   << "const uint tangentAccessInfo = getTangentAccessInfo();\n";
        }

        output << "    // first vertex\n";
        generateVertexCode(output, "vertexAndDrawableDataPtr.x", "vec3(1,0,0)", "inId[0]");
        output << "    // second vertex\n";
        generateVertexCode(output, "vertexAndDrawableDataPtr.y", "vec3(0,1,0)", "inId[1]");
        output << "    // third vertex\n";
        generateVertexCode(output, "vertexAndDrawableDataPtr.z", "vec3(0,0,1)", "inId[2]");

        output.pop();
        output << "}\n";
        return std::move(output.string());
    }

};

class FragmentShaderGenerator : public MetadataAttribs, MetadataMaterial, MetadataTextures
{

    void generateTexture(OutputStream &output, TextureType type, const Texture &texture, const std::string_view sampleCode, const std::string_view sampleSuffix, const std::function<void(OutputStream &, const Texture &)> &body)
    {
        if (optimizeAttribs && numTextureAttribs == 0) {
            return;
        }
        if (!optimizeTextures) {
            uint32_t typeValue = static_cast<int>(type) << 8;
#ifndef NDEBUG
            if constexpr (ShaderValidation) {
                output() << "if(textureType != 0 && textureType < 0x" << toHex(typeValue) << ") {\n";
                output << "    outColor = vec4(1, 0, 1, 1);\n";
                output << "    return;\n";
                output << "}\n";
            }
#endif
            output() << "if(textureType == 0x" << toHex(typeValue) << ")\n";
        }
#ifndef NDEBUG
        if constexpr (ShaderValidation) {
            if (texture) {
                uint32_t typeValue = static_cast<int>(type) << 8;
                output() << "if((textureInfo.texCoordIndexTypeAndSettings & 0xff00) != 0x" << toHex(typeValue) << ") {\n";
                output << "    outColor = vec4(1, 0, 1, 1);\n";
                output << "    return;\n";
                output << "} else\n";
            }
        }
#endif
        if (!optimizeTextures || texture) {
            output << "{\n";
            output.push();
            if (optimizeAttribs) {
                if (optimizeTextures) {
                    const auto &index = texture.getTexCoordIndex();
                    assert(index >= 4 && "textureCoordIndex must be >= 4");
                    output() << "vec2 uv = inTexCoords[" << (index - 4) << "];\n";
                }
                else if (numTextureAttribs > 0) {
                    // assert(numTextureAttribs > 0 && "missing texCoords");
                    output() << "vec2 uv = inTexCoords[getTextureCoordIndex(textureInfo) - 4];\n";
                }
            }
            else {
                // compute texture coordinates from relevant data,
                // and transform them if requested
                output << "vec2 uv = computeTextureCoordinates(textureInfo, vertex0DataPtr, vertex1DataPtr, vertex2DataPtr, inBarycentricCoords);\n";
            }
            // sample texture
            output() << sampleCode << "texture(textureDB[textureInfo.textureIndex], uv)" << sampleSuffix << ";\n";

            body(output, texture);

            // update pointer to point to the next texture
            if (!optimizeTextures) {
                output << "textureInfo = getNextTextureInfo(textureInfo);\n";
            }
            else {
                output() << "textureInfo = TextureInfoRef(uint64_t(textureInfo) + " << texture.getTextureSize() << ");\n";
            }
        }
        if (!optimizeTextures) {
            output << "textureType = textureInfo.texCoordIndexTypeAndSettings & 0xff00;\n";
        }
        if (!optimizeTextures || texture) {
            output.pop();
            output << "}\n";
        }

    }

    void generateTexture(OutputStream &output, TextureType type, const std::string_view sampleCode, const std::string_view sampleSuffix, const std::function<void(OutputStream &, const Texture &)> &body)
    {
        generateTexture(output, type, getTexture(type), sampleCode, sampleSuffix, body);
    }

    void generateNormalTexture(OutputStream &output)
    {

        generateTexture(output, TextureType::normal, "vec3 tangentSpaceNormal = ", ".rgb", [&](OutputStream &output, const Texture &texture){
            // transform in tangent space and normalize
            output << "tangentSpaceNormal = tangentSpaceNormal * 2 - 1;  // transform from 0..1 to -1..1\n";

            texture.generateStrengthFlagCode(output, optimizeTextures,
                "tangentSpaceNormal *= vec3(textureInfo.strength, textureInfo.strength, 1);\n"
            );

            output << "tangentSpaceNormal = normalize(tangentSpaceNormal);\n";
            // transform normal
            output << "vec3 t = normalize(inFragmentTangent);\n"
                   << "mat3 tbn = { t, cross(normal, t), normal };\n"
                   << "normal = tbn * tangentSpaceNormal;\n";
        });

    }

    void generateBaseTexture(OutputStream &output)
    {
        generateTexture(output, TextureType::base, "vec4 baseTextureValue = ", "", [&](OutputStream &output, const Texture &texture){

            texture.generateStrengthFlagCode(output, optimizeTextures,
                "baseTextureValue *= textureInfo.strength;\n"
            );

            texture.generateTextureEnvironmentCode(output, optimizeTextures); // TODO

            generateOptimizedIf(output, optimizeMaterialAlpha, materialIgnoreBaseTextureAlpha,
                "getMaterialIgnoreBaseTextureAlpha()",
                [&](OutputStream &output) {
                    if (!optimizeTextures) {
                        output << "if(texEnv == 0)  // modulate\n";
                        output.push();
                        output() << "baseColor.rgb *= baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 1) // replace\n";
                        output.push();
                        output() << "baseColor.rgb = baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 2) // decal\n";
                        output.push();
                        output() << "baseColor.rgb = baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 3) // blend\n";
                        output.push();
                        output() << "baseColor.rgb = " << "baseColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 4) // add\n";
                        output.push();
                        output() << "baseColor.rgb = " << "baseColor.rgb + baseTextureValue.rgb;\n";
                        output.pop();
                    }
                    else {
                        auto texEnv = texture.getTextureEnvironment();
                        switch (texEnv) {
                            case 0:
                                output() << "baseColor.rgb *= baseTextureValue.rgb;\n";
                                break;
                            case 1:
                                output() << "baseColor.rgb = baseTextureValue.rgb;\n";
                                break;
                            case 2:
                                output() << "baseColor.rgb = baseTextureValue.rgb;\n";
                                break;
                            case 3:
                                output() << "baseColor.rgb = " << "baseColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb;\n";
                                break;
                            case 4:
                                output() << "baseColor.rgb = " << "baseColor.rgb + baseTextureValue.rgb;\n";
                                break;
                            default:
                                break;
                        }
                    }
                },
                [&](OutputStream &output) {
                    if (!optimizeTextures) {
                        output << "if(texEnv == 0)  // modulate\n";
                        output.push();
                        output() << "baseColor *= baseTextureValue;\n";
                        output.pop();
                        output << "else if(texEnv == 1) // replace\n";
                        output.push();
                        output() << "baseColor = vec4(baseTextureValue.rgb, baseTextureValue.a * baseColor.a);\n";
                        output.pop();
                        output << "else if(texEnv == 2) // decal\n";
                        output.push();
                        output() << "baseColor = vec4(baseColor.rgb*(1-baseTextureValue.a) + baseTextureValue.rgb*baseTextureValue.a, baseColor.a);\n";
                        output.pop();
                        output << "else if(texEnv == 3) // blend\n";
                        output.push();
                        output() << "baseColor = vec4(" << "baseColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb, baseColor.a*baseTextureValue.a);\n";
                        output.pop();
                        output << "else if(texEnv == 3) // add\n";
                        output.push();
                        output() << "baseColor = vec4(" << "baseColor.rgb + baseTextureValue.rgb, baseColor.a * baseTextureValue.a);\n";
                        output.pop();
                    }
                    else {
                        auto texEnv = texture.getTextureEnvironment();
                        switch (texEnv) {
                            case 0:
                                output << "baseColor *= baseTextureValue;\n";
                                break;
                            case 1:
                                output << "baseColor = vec4(baseTextureValue.rgb, baseTextureValue.a * baseColor.a);\n";
                                break;
                            case 2:
                                output << "baseColor = vec4(baseColor.rgb*(1-baseTextureValue.a) + baseTextureValue.rgb*baseTextureValue.a, baseColor.a);\n";
                                break;
                            case 3:
                                output << "baseColor = vec4(baseColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb, baseColor.a*baseTextureValue.a);\n";
                                break;
                            case 4:
                                output << "baseColor = vec4(baseColor.rgb + baseTextureValue.rgb, baseColor.a * baseTextureValue.a);\n";
                                break;
                            default:
                                break;
                        }
                    }
                }
            );
        });

    }

    void generateMetallicRoughnessTexture(OutputStream &output)
    {
        const auto &texture = getTexture(TextureType::metallicRoughness);
        if (!optimizeTextures || texture) {
            generateTexture(output, TextureType::metallicRoughness, "metallicRoughness *= ", ".rg", [&](OutputStream &output, const Texture &texture) {
            });
        }
    }

    void generateOcclusionTexture(OutputStream &output)
    {
        const auto &occlusionTexture = getTexture(TextureType::occlusion);
        if (!optimizeTextures || occlusionTexture) {
            output << "float occlusionTextureValue = 1.;\n";
            std::string componentAccess = "[";
            if (optimizeTextures) {
                componentAccess += std::to_string(occlusionTexture.getTextureFirstComponentIndex());
                componentAccess += "]";
            }
            else {
                componentAccess += "getTextureFirstComponentIndex(textureInfo)]";
            }
            generateTexture(output, TextureType::occlusion, "occlusionTextureValue = ", componentAccess, [&](OutputStream &output, const Texture &texture) {
                // glTF uses red component, so 0 should be provided for glTF
                texture.generateStrengthFlagCode(output, optimizeTextures,
                                                          "occlusionTextureValue *= textureInfo.strength;\n"
                );
            });
        }
    }

    void generateEmissiveTexture(OutputStream &output)
    {
        const auto &emissiveTexture = getTexture(TextureType::emissive);
        if (!optimizeTextures || emissiveTexture) {
            output << "vec3 emissiveTextureValue = vec3(1);\n";
            generateTexture(output, TextureType::emissive, "emissiveTextureValue = ", ".rgb", [&](OutputStream &output, const Texture &texture) {
                texture.generateStrengthFlagCode(output, optimizeTextures,
                                                          "emissiveTextureValue *= textureInfo.strength;\n"
                );
            });
        }
    }

    void generateOpenGLLightFunctions(OutputStream &output)
    {
        output << R"(
//               Viewer  Normal  Halfway vector
//     Reflected     V      N      H           Light source
//        light       V     N     H         LLL
//           RRR       V    N    H       LLL
//              RRR     V   N   H     LLL
//                 RRR   V  N  H   LLL
//                    RRR V N H LLL
//                       RRVNHLL
//      SSSSSSSSSSSSSSSSSSS F SSSSSSSSSSSSSSSSSSS
//
//  S - Surface of the primitive
//  F - Fragment being rendered
//  N - Normal - normalized normal at Fragment's surface position
//  V - Viewer - normalized vector from Fragment to Viewer
//  L - Light - normalized vector from Fragment to Light source
//  H - Halfway vector
//  R - Reflected light direction - normalized vector
//
//  All vectors are in eye coordinates.


void OpenGLDirectionalLight(
    in LightRef lightData,
    in vec3 normal,
    in vec3 viewerToFragmentDirection,
    in float shininess,
    inout vec3 ambient,
    inout vec3 diffuse,
    inout vec3 specular)
{
    // nDotL = normal . light direction
    vec3 l = lightData.positionOrDirection;  // directional light uses direction towards the incoming light here
    float nDotL = dot(normal, l);

    if(nDotL > 0.) {

        // Lambertian diffuse reflection
        diffuse += nDotL * lightData.opengl.diffuse;

        // nDotH = normal . halfway vector
        vec3 h = normalize(l - viewerToFragmentDirection);
        float nDotH = dot(normal, h);

        if(nDotH > 0.) {

            // specular term and its power factor
            float pf = pow(nDotH, shininess);
            specular = pf * lightData.opengl.specular;

        }
    }

    ambient += lightData.opengl.ambient;
}


void OpenGLPointLight(
    in LightRef lightData,
    in vec3 normal,
    in vec3 viewerToFragmentDirection,
    in float shininess,
    inout vec3 ambient,
    inout vec3 diffuse,
    inout vec3 specular)
{
    // nDotL = normal . light direction
    vec3 lPos = lightData.positionOrDirection;  // point light uses position of the light source in eye coordinates here
    lPos -= inFragmentPosition3;  // make lPos vector from fragment to light
    float lLen = length(lPos);
    vec3 lDir = lPos / lLen;  // direction from the fragment to the light source
    float nDotL = dot(normal, lDir);

    // attenuation
    float att = 1. / (lightData.opengl.constantAttenuation +
        lightData.opengl.linearAttenuation * lLen +
        lightData.opengl.quadraticAttenuation * lLen * lLen);

    if(nDotL > 0.) {

        // Lambertian diffuse reflection
        diffuse += nDotL * lightData.opengl.diffuse * att;

        // nDotH = normal . halfway vector
        vec3 h = normalize(lDir - viewerToFragmentDirection);
        float nDotH = dot(normal, h);

        if(nDotH > 0.) {

            // specular term and its power factor
            float pf = pow(nDotH, shininess);
            specular = pf * lightData.opengl.specular * att;

        }
    }

    ambient += lightData.opengl.ambient * att;
}


void OpenGLSpotlight(
    in LightRef lightData,
    in vec3 normal,
    in vec3 viewerToFragmentDirection,
    in float shininess,
    inout vec3 ambient,
    inout vec3 diffuse,
    inout vec3 specular)
{
    // light position and direction
    vec3 lPos = lightData.positionOrDirection;  // point light uses position of the light source in eye coordinates here
    lPos -= inFragmentPosition3;  // make lPos vector from fragment to light
    float lLen = length(lPos);
    vec3 lDir = lPos / lLen;  // direction from the fragment to the light source

    // skip everything outside of spotlight outer cone
    float spotEffect = dot(-lDir, lightData.spotlight.direction);
    if(spotEffect > lightData.spotlight.cosOuterConeAngle) {

        // compute spotEffect
        if(lightData.spotlight.cosInnerConeAngle == -1.) {

            // OpenGL spotlight
            spotEffect = pow(spotEffect, lightData.spotlight.spotExponent);

        } else {

            // DirectX style spotlight
            spotEffect =
                pow(
                    clamp(
                        (spotEffect - lightData.spotlight.cosOuterConeAngle) /
                        (lightData.spotlight.cosInnerConeAngle - lightData.spotlight.cosOuterConeAngle),
                        0., 1.),
                    lightData.spotlight.spotExponent
                );

            // Hermite interpolation
            // (result will be between 0 and 1)
            spotEffect = smoothstep(0., 1., spotEffect);

        }

        // nDotL = normal . light direction
        float nDotL = dot(normal, lDir);

        // attenuation
        float att = 1. / (lightData.opengl.constantAttenuation +
            lightData.opengl.linearAttenuation * lLen +
            lightData.opengl.quadraticAttenuation * lLen * lLen);

        if(nDotL > 0.) {

            // Lambertian diffuse reflection
            diffuse += nDotL * lightData.opengl.diffuse * att * spotEffect;

            // nDotH = normal . halfway vector
            vec3 h = normalize(lDir - viewerToFragmentDirection);
            float nDotH = dot(normal, h);

            if(nDotH > 0.) {

                // specular term and its power factor
                float pf = pow(nDotH, shininess);
                specular = pf * lightData.opengl.specular * att * spotEffect;

            }
        }

        ambient += lightData.opengl.ambient * att * spotEffect;
    }
}
)";
    }

    void generateUnlitModel(OutputStream &output)
    {
        output << R"(
        // material data
        UnlitMaterialRef materialData = UnlitMaterialRef(drawableDataPtr);

        outColor = vec4(1, 0, 0, 1);
/*
        if(getMaterialUseColorAttribute()) {
            uint colorAccessInfo = getColorAccessInfo();
            if(getMaterialIgnoreColorAttributeAlpha()) {
                vec3 c =
                    readVec3(vertex0DataPtr, colorAccessInfo) * inBarycentricCoords.x +
                    readVec3(vertex1DataPtr, colorAccessInfo) * inBarycentricCoords.y +
                    readVec3(vertex2DataPtr, colorAccessInfo) * inBarycentricCoords.z;
                outColor.rgb = c;
                if(getMaterialIgnoreMaterialAlpha())
                    outColor.a = 1;
                else
                    outColor.a *= materialData.colorAndAlpha.a;
            } else {
                vec4 c =
                    readVec4(vertex0DataPtr, colorAccessInfo) * inBarycentricCoords.x +
                    readVec4(vertex1DataPtr, colorAccessInfo) * inBarycentricCoords.y +
                    readVec4(vertex2DataPtr, colorAccessInfo) * inBarycentricCoords.z;
                outColor = c;
                if(!getMaterialIgnoreMaterialAlpha())
                    outColor.a *= materialData.colorAndAlpha.a;
            }
        } else {
            if(getMaterialIgnoreMaterialAlpha())
                outColor = vec4(materialData.colorAndAlpha.rgb, 1);
            else
                outColor = materialData.colorAndAlpha;
        }
*/

        /*
        if(textureType == 0x0400) {

            // compute texture coordinates from relevant data,
            // and transform them if requested
            vec2 uv = computeTextureCoordinates(textureInfo,
                vertex0DataPtr, vertex1DataPtr, vertex2DataPtr, inBarycentricCoords);

            // sample texture
            vec4 baseTextureValue = texture(textureDB[textureInfo.textureIndex], uv);

            // multiply by strength
            if(getTextureUseStrengthFlag(textureInfo))
                baseTextureValue *= textureInfo.strength;

            // apply texture using texEnv
            uint texEnv = getTextureEnvironment(textureInfo);
            if(getMaterialIgnoreBaseTextureAlpha()) {
                if(texEnv == 0)  // modulate
                    outColor.rgb *= baseTextureValue.rgb;
                else if(texEnv == 1) // replace
                    outColor.rgb = baseTextureValue.rgb;
                else if(texEnv == 2) // decal
                    outColor.rgb = baseTextureValue.rgb;
                else if(texEnv == 3) // blend
                    outColor.rgb = outColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb;
                else if(texEnv == 3) // add
                    outColor.rgb = outColor.rgb + baseTextureValue.rgb;
            } else {
                if(texEnv == 0)  // modulate
                    outColor *= baseTextureValue;
                else if(texEnv == 1) // replace
                    outColor = vec4(baseTextureValue.rgb, baseTextureValue.a * outColor.a);
                else if(texEnv == 2) // decal
                    outColor = vec4(outColor.rgb*(1-baseTextureValue.a) + baseTextureValue.rgb*baseTextureValue.a, outColor.a);
                else if(texEnv == 3) // blend
                    outColor = vec4(outColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb, outColor.a*baseTextureValue.a);
                else if(texEnv == 3) // add
                    outColor = vec4(outColor.rgb + baseTextureValue.rgb, outColor.a * baseTextureValue.a);
            }

            // update pointer to point to the next texture
            textureInfo = getNextTextureInfo(textureInfo);
            textureType = textureInfo.texCoordIndexTypeAndSettings & 0xff00;

        }
        */
)";
    }

    void generateColor(OutputStream &output, bool ambient)
    {

        generateOptimizedIf(output, optimizeMaterialColorAttribute, materialUseColorAttribute,
            "getMaterialUseColorAttribute()",
            [&](OutputStream &output){

                if (!optimizeAttribs) {
                    output << "uint colorAccessInfo = getColorAccessInfo();\n";
                    output << "if (colorAccessInfo != 0) {\n";
                    output.push();
                }
                generateOptimizedIf(output, optimizeMaterialColorAttribute, materialIgnoreColorAttributeAlpha,
                    "getMaterialIgnoreColorAttributeAlpha()",
                    [&](OutputStream &output){
                        if (!optimizeAttribs) {
                            output << "vec3 color =\n";
                            output << "    readVec3(vertex0DataPtr, colorAccessInfo) * inBarycentricCoords.x +\n";
                            output << "    readVec3(vertex1DataPtr, colorAccessInfo) * inBarycentricCoords.y +\n";
                            output << "    readVec3(vertex2DataPtr, colorAccessInfo) * inBarycentricCoords.z;\n";
                            output << "baseColor.rgb *= color;\n";
                        }
                        else if (colorAttrib) {
                            output << "baseColor.rgb *= inFragmentColor.rgb;\n";
                        }
                    },
                    [&](OutputStream &output){
                        if (!optimizeAttribs) {
                            output << "vec4 color =\n";
                            output << "    readVec4(vertex0DataPtr, colorAccessInfo) * inBarycentricCoords.x +\n";
                            output << "    readVec4(vertex1DataPtr, colorAccessInfo) * inBarycentricCoords.y +\n";
                            output << "    readVec4(vertex2DataPtr, colorAccessInfo) * inBarycentricCoords.z;\n";
                            output << "baseColor *= color;\n";
                        }
                        else if (colorAttrib) {
                            if (colorAttribAlpha) {
                                output << "baseColor *= inFragmentColor;\n";
                            }
                            else {
                                output << "baseColor.rgb *= inFragmentColor.rgb;\n";
                            }
                        }
                    }
                );
                if (ambient) {
                    generateOptimizedIf(output, optimizeMaterialColorAttribute, materialUseColorAttributeForAmbientAndDiffuse,
                        "getMaterialUseColorAttributeForAmbientAndDiffuse()",
                        "ambientColor = baseColor.rgb;\n",
                        "ambientColor = materialData.ambient;\n"
                    );
                }
                if (!optimizeAttribs) {
                    output.pop();
                    output << "}\n";
                }
            },
            [&](OutputStream &output){
                if (ambient) {
                    output << "ambientColor = materialData.ambient;\n";
                }
            }
        );

    }

    void generatePhongModel(OutputStream &output)
    {

        generateOcclusionTexture(output);

        // material data
        output << "PhongMaterialRef materialData = PhongMaterialRef(drawableDataPtr);\n";
        // ambientColor, diffuseColor and alpha
        output << "vec4 baseColor = materialData.diffuseAndAlpha;\n";
        output << "vec3 ambientColor;\n";

        generateBaseTexture(output);
        generateColor(output, true);

        generateEmissiveTexture(output);

        const auto &emissiveTexture = getTexture(TextureType::emissive);
        const auto &occlusionTexture = getTexture(TextureType::occlusion);

        const auto colorCalculation = [&](bool light){
            std::string output = "materialData.emission";
            if (!optimizeTextures || emissiveTexture) {
                output += " * emissiveTextureValue";
            }

            output += " + (";
            if (light) {
                output += "ambientProduct + ";
            }
            output += "scene.ambientLight) * ambientColor";
            if (!optimizeTextures || occlusionTexture) {
                output += " * occlusionTextureValue";
            }

            if (light) {
                output += " + diffuseProduct * baseColor.rgb + specularProduct * materialData.specular";
            }

            return output;
        };

        // light data
        output << "uint64_t lightDataPtr = sceneDataPtr + getLightDataOffset();\n"

               << "LightRef lightData = LightRef(lightDataPtr);\n";
        // iterate over all light sources
        output << "uint lightSettings = lightData.settings;\n"
               << "if(lightSettings != 0) {\n";
        output.push();
        // Phong color products

        output << "vec3 ambientProduct  = vec3(0);\n"
               << "vec3 diffuseProduct  = vec3(0);\n"
               << "vec3 specularProduct = vec3(0);\n";
        // iterate over all lights
        output << "do{\n";
        output.push();

        output << "uint lightType = lightSettings & 0x3;\n"
               << "if(lightType == 1)\n";
        output.push();
        output << "OpenGLDirectionalLight(lightData, normal,\n"
               << "    viewerToFragmentDirection, materialData.shininess,\n"
               << "    ambientProduct, diffuseProduct, specularProduct);\n";
        output.pop();
        output << "else if(lightType == 2)\n";
        output.push();
        output << "OpenGLPointLight(lightData, normal,\n"
               << "    viewerToFragmentDirection, materialData.shininess,\n"
               << "    ambientProduct, diffuseProduct, specularProduct);\n";
        output.pop();
        output << "else\n";
        output.push();
        output << "OpenGLSpotlight(lightData, normal,\n"
               << "    viewerToFragmentDirection, materialData.shininess,\n"
               << "    ambientProduct, diffuseProduct, specularProduct);\n";
        output.pop();
        output << "lightDataPtr += getLightDataSize();\n"
               << "lightData = LightRef(lightDataPtr);\n"
               << "lightSettings = lightData.settings;\n";

        output.pop();
        output << "} while(lightSettings != 0);\n";
        // Phong equation
        output() << "outColor.rgb = " << colorCalculation(true) << ";\n";
        output.pop();
        output << "} else {\n";
        output.push();
        // Phong equation without light sources
        output() << "outColor.rgb = " << colorCalculation(false) << ";\n";
        output.pop();
        output << "}\n";
    }

    void generatePRBFunctions(OutputStream &output) {
        output << R"(

float max3(vec3 v)
{
    return max(max(v.x, v.y), v.z);
}

/*
PBR functions derived from https://github.com/Nadrin/PBR:
MIT License
Copyright (c) 2017-2018 Michał Siejak
*/
const float PI = 3.1415926535897932384626433832795f;
const float INV_PI = 1.0 / PI;
const float Epsilon = 0.00001;

// Shlick's approximation of the Fresnel factor.
// see https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#fresnel
vec3 F_Schlick(float VdotH, vec3 F0)
{
	return F0 + (1 - F0) * pow((1 - VdotH), 5);
}

// Single term for separable Schlick-GGX below
// see https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#complete-model
float G_SchlickG1(float NdotV, float k)
{
	return NdotV / (NdotV * (1 - k) + k);
}
// Schlick-GGX approximation of geometric attenuation function using Smith's method.
float G_SchlickGGX(float NdotV, float NdotL, float roughness)
{
    float r = roughness + 1;
	float k = (r * r) / 8;
	return G_SchlickG1(NdotV, k) * G_SchlickG1(NdotL, k);
}
// Trowbridge-Reitz NDF (Normal Distribution Function)
float D_GGXTR(float NdotH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = (NdotH * NdotH) * (alpha2 - 1) + 1;
	return alpha2 / (PI * denom * denom);
}

)";

    }

    void generateGltfPBRModel(OutputStream &output)
    {
        generateOcclusionTexture(output);

        output << "MetallicRoughnessMaterialRef materialData = MetallicRoughnessMaterialRef(drawableDataPtr);\n";

        output << "vec4 baseColor = materialData.baseColorFactor;\n";

        output << "vec3 view = normalize(-inFragmentPosition3);\n";
        output << "float NdotV = max(dot(normal, view), 0);\n";

        generateBaseTexture(output);
        generateColor(output, false);

        generateEmissiveTexture(output);

        output << "vec2 metallicRoughness = vec2(materialData.metallicFactor, materialData.roughnessFactor);\n";
        generateMetallicRoughnessTexture(output);
        output << "vec3 F0 = mix(materialData.reflectance, baseColor.rgb, metallicRoughness.y);\n";
        output << "vec3 cDiff = mix(baseColor.rgb, vec3(0.f), metallicRoughness.y) * INV_PI;\n";

        output << "vec3 materialColor = baseColor.rgb;\n";
        output << "vec3 sheenColor = materialData.sheenColorFactor;\n";
        output << "float sheenRoughness = materialData.sheenRoughnessFactor;\n";
        if (!optimizeMaterialModel) { // TODO add separate flag
            output << "if (getMaterialUseSheen()) {\n";
        }
        if (!optimizeMaterialModel || materialUseSheen) {
            // sheen implementation
            // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md
            generateTexture(output, TextureType::sheen, "sheenColor *= ", ".rgb", [&](OutputStream &output, const Texture &texture) { });
            generateTexture(output, TextureType::sheenRoughness, "sheenRoughness *= ", ".r", [&](OutputStream &output, const Texture &texture) { });
            output << "vec2 uv = clamp(vec2(NdotV, sheenRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));\n";
            output << "float sheen_brdf = texture(textureDB[0], uv).r;\n";
            output << "float sheen_albedo_scaling = 1.0 - max3(sheenColor) * texture(textureDB[1], uv).r;\n";
            output << "materialColor = sheenColor * sheen_brdf + baseColor.rgb * sheen_albedo_scaling;\n";
        }
        if (!optimizeMaterialModel) { // TODO add separate flag
            output << "}\n";
        }

        output << "uint64_t lightDataPtr = sceneDataPtr + getLightDataOffset();\n"
               << "LightRef lightData = LightRef(lightDataPtr);\n";
        output << "uint lightSettings = lightData.settings;\n"
            << "if(lightSettings != 0) {\n";
        output.push();

        output << "vec3 lightProduct  = vec3(0);\n";
        // iterate over all lights
        output << "do{\n";
        output.push();
        output << "vec3 L;\n";
        output << "vec3 lightColor = lightData.gltf.color * lightData.gltf.intensity;\n";
        output << "uint lightType = lightSettings & 0x3;\n"
               << "if(lightType == 1) {\n";
        // directional light
        output.push();
        output << "L = lightData.positionOrDirection;\n";
        output.pop();
        output << "} else if(lightType == 2) {\n";
        // point light
        output.push();
        output << "L = lightData.positionOrDirection - inFragmentPosition3;\n";
        output << "float dist = length(L);\n";
        output << "L = normalize(L);\n";
        output << "lightColor = lightColor / (dist * dist);";
        output.pop();
        output << "} else {\n";
        // spotlight
        output.push();
        output << "L = lightData.positionOrDirection - inFragmentPosition3;\n";
        output << "float dist = length(L);\n";
        output << "L = normalize(L);\n";
        output << "lightColor = lightColor / (dist * dist);";
        output.pop();
        output << "}\n";
        output << R"(
/*
glTF based PBR light calculation
see https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metal-brdf-and-dielectric-brdf
*/)";
        output << "vec3  H = normalize(view + L);\n";
        output << "float NdotL = max(dot(normal, L), 0);\n";
        output << "float NdotH = max(dot(normal, H), 0);\n";
        output << "float VdotH = max(dot(view,   H), 0);\n";

        // Cook-Torrance specular microfacet BRDF
        output << "vec3  F = F_Schlick(VdotH, F0);\n";
        output << "float D = D_GGXTR(NdotH, metallicRoughness.x);\n";
        output << "float G = G_SchlickGGX(NdotV, NdotL, metallicRoughness.x);\n";
        output << "vec3  fSpecular = (F * D * G) / max(Epsilon, 4 * NdotV * NdotL);\n";
        // Lambert diffuse BRDF
        output << "vec3 fDiffuse = (vec3(1.f) - F) * cDiff;\n";

        output << "lightProduct += (fDiffuse + fSpecular) * lightColor * NdotL;\n";

        output << "lightDataPtr += getLightDataSize();\n"
               << "lightData = LightRef(lightDataPtr);\n"
               << "lightSettings = lightData.settings;\n";

        output.pop();
        output << "} while(lightSettings != 0);\n";

        const auto &emissiveTexture = getTexture(TextureType::emissive);
        const auto &occlusionTexture = getTexture(TextureType::occlusion);
        {
            auto line = output();
            line << "outColor.rgb = (lightProduct + materialColor.rgb * scene.ambientLight)";
            if (!optimizeTextures || occlusionTexture) {
                line << " * occlusionTextureValue";
            }
            line << " + materialData.emissiveFactor";
            if (!optimizeTextures || emissiveTexture) {
                line << " * emissiveTextureValue";
            }
            line << ";\n";
        }
        output() << "outColor.a = baseColor.a;\n";
        // debug
        // output() << "outColor.rgb = vec3(normal);";
        // output() << "outColor.rgb = vec3(metallicRoughness, 0);";
        // output() << "outColor.rgb = diffuseColor.rgb * scene.ambientLight;\n";
        output.pop();
        output << "}\n";

    }

    void generateMaterialModel(OutputStream &output, int materialModelValue, bool elseBranch, const std::function<void(FragmentShaderGenerator*, OutputStream&)> &body, const std::string_view comment)
    {
        const bool evaluation = !optimizeMaterialModel || materialModel == materialModelValue;
        if (evaluation) {
            output << comment;
        }
        if (!optimizeMaterialModel) {
            output() << (elseBranch? "else " : "") << "if(materialModel == " << materialModelValue << ")\n";
        }
        if (evaluation) {
            output << "{\n";
            output.push();
            body(this, output);
            output.pop();
            output << "}\n";
        }
    }

    void generateMain(OutputStream &output)
    {
        output << "void main()\n" << "{\n";
        output.push();
        // input data and structures
        output << "SceneDataRef scene = SceneDataRef(sceneDataPtr);\n";

        if (!optimizeAttribs) {
            output << "uint64_t vertex0DataPtr  = inVertexAndDrawableDataPtr.x;\n"
                   << "uint64_t vertex1DataPtr  = inVertexAndDrawableDataPtr.y;\n"
                   << "uint64_t vertex2DataPtr  = inVertexAndDrawableDataPtr.z;\n"
                   << "uint64_t drawableDataPtr = inVertexAndDrawableDataPtr.w;\n";
        }
        else {
            output << "uint64_t drawableDataPtr = inDrawableDataPtr;\n";
        }
        output << "vec3 viewerToFragmentDirection = normalize(inFragmentPosition3);\n";

        // TODO getMaterialTwoSidedLighting()
        if (!optimizeMaterialModel || materialModel >= 1) {
            // normal
            output << "vec3 normal;\n";
        }
        if (!optimizeMaterialModel) {
            output << "uint materialModel = getMaterialModel();\n"
                   << "if(materialModel >= 1) {\n";
            output.push();
        }
        if (!optimizeMaterialModel || materialModel >= 1) {
            generateOptimizedIf(output, optimizeAttribs, state.attribAccessInfo[1] == 0,
                    "getGenerateFlatNormals()",
                    "normal = -normalize(cross(dFdx(inFragmentPosition3), dFdy(inFragmentPosition3)));\n",
                    "normal = normalize(inFragmentNormal);\n"
            );
        }
        if (!optimizeMaterialModel) {
            output.pop();
            output << "}\n";
        }

        bool hasTextures = numTextureAttribs > 0 || state.numTextures > 0 || (!optimizeTextures && !optimizeAttribs);
        if (hasTextures) {
            if (optimizeMaterialModel) {
                output() << "TextureInfoRef textureInfo = TextureInfoRef(drawableDataPtr + " << materialFirstTextureOffset << ");\n";
            }
            else {
                output << "uint textureOffset = getMaterialFirstTextureOffset();\n"
                       << "TextureInfoRef textureInfo;\n";
            }
            if (!optimizeTextures) {
                output << "uint textureType = 0;\n";
            }
            if (!optimizeMaterialModel) {
                output << "if (textureOffset != 0) {\n";
                output.push();
                output << "textureInfo = TextureInfoRef(drawableDataPtr + textureOffset);\n";
            }
            if (!optimizeTextures) {
                output << "textureType = textureInfo.texCoordIndexTypeAndSettings & 0xff00;\n";
            }
            if (!optimizeMaterialModel) {
                output.pop();
                output << "}\n";
            }
        }

        generateNormalTexture(output);

        generateMaterialModel(output, 0, false, &FragmentShaderGenerator::generateUnlitModel,   "    // unlit material\n");
        generateMaterialModel(output, 1, true,  &FragmentShaderGenerator::generatePhongModel,   "    // Blinn-Phong material model, implemented in the similar way as OpenGL does\n");
        generateMaterialModel(output, 2, true,  &FragmentShaderGenerator::generateGltfPBRModel, "    // Metallic-Roughness material model from PBR family,\n""    // implemented in the similar way as glTF doesl\n");

        if (state.idBuffer) {
            // write id-buffer
            output << "outId[0] = stateSetID;\n"
                   << "outId[1] = inId[0];  // gl_DrawID - index of indirect drawing structure\n"
                   << "outId[2] = inId[1];  // gl_InstanceIndex\n"
                   << "outId[3] = gl_PrimitiveID;\n";
        }
        output.pop();
        output << "}\n";

    }

public:

    static constexpr const char* ShaderName = "fragment";
    static constexpr shaderc_shader_kind ShaderKind = shaderc_fragment_shader;

    const ShaderState& state;

    explicit FragmentShaderGenerator(const ShaderState& state)
        : MetadataAttribs(state)
        , MetadataMaterial(state)
        , MetadataTextures(state)
        , state(state)
    {
    }

    std::string generate()
    {
        OutputStream output;
        output.reserve(CodeStringReservation);

        output << "#version 460\n"
               << "#extension GL_EXT_buffer_reference : require\n"
               << "#extension GL_ARB_gpu_shader_int64 : require\n"
               << "#extension GL_EXT_nonuniform_qualifier : require  // this enables SPV_EXT_descriptor_indexing\n";

        if (!optimizeAttribs) {
            generateUberShaderReadFuncs(output);
        }
        includeUberShaderInterface(output, false, optimizeTextures, optimizeAttribs, state.idBuffer);

        generateFragmentInputInterface(output, state.idBuffer);

        output << "layout(location = 0) out vec4 outColor;\n";
        if (state.idBuffer) {
            output << "// ID_BUFFER\n";
            output << "layout(location = 1) out uvec4 outId;\n";
        }
        // textures
        output << "layout(set=0, binding=0) uniform sampler2D textureDB[];\n";

        generateOpenGLLightFunctions(output);

        generatePRBFunctions(output);

        generateMain(output);

        return std::move(output.string());
    }
};

#ifdef CADPL_USE_PREGEN
template<typename MapKey>
vk::ShaderModule createFromPregenerated(const ShaderState& state, CadR::VulkanDevice& device, const std::map<MapKey, std::pair<const uint32_t*, uint32_t>> &map)
{
    const auto it = map.find(state);
    if (it != map.end()) {
        std::cout << "Creating shader from pregen cache\n";
        vk::ShaderModuleCreateInfo info;
        info.flags = vk::ShaderModuleCreateFlags();
        info.codeSize = it->second.second;
        info.pCode = it->second.first;
        return device.createShaderModule(info);
    }
    return {};
}
#endif

template<typename T>
static vk::ShaderModule createShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName)
{
#ifdef CADPL_USE_PREGEN
    vk::ShaderModule shader = VK_NULL_HANDLE;
    if constexpr (std::is_same_v<T, VertexShaderGenerator>) {
        shader = createFromPregenerated(state, device, spirvShaderMapVertex);
    }
    else if constexpr (std::is_same_v<T, GeometryShaderGenerator>) {
        shader = createFromPregenerated(state, device, spirvShaderMapGeometry);
    }
    else if constexpr (std::is_same_v<T, FragmentShaderGenerator>) {
        shader = createFromPregenerated(state, device, spirvShaderMapFragment);
    }
    if (shader) {
        return shader;
    }
#endif

    shaderc::SpvCompilationResult compilation;
    vk::ShaderModuleCreateInfo info;
    info.flags = vk::ShaderModuleCreateFlags();

    std::string cacheFileName;
    std::vector<uint32_t> spirv;
    const bool useCache = cacheSpirV && !cacheDirectory.empty();
    if (useCache) {
        cacheFileName = ShaderGenerator::createCacheName(cacheDirectory, T::ShaderKind, cacheName);
        spirv = readSpirvFromFile(cacheFileName + ".spv");
    }
    if (spirv.empty()) {
        const auto start = std::chrono::steady_clock::now();
        T generator(state);
        const auto code = generator.generate();
#ifndef NDEBUG
        if (!cacheDirectory.empty() && !cacheName.empty()) {
            // debug files
            std::string debug = code;
            debug += "\n/*\n";
            debug += state.debugDump();
            debug += "*/\n";
            auto cacheFileName = ShaderGenerator::createCacheName(cacheDirectory, T::ShaderKind, cacheName);
            writeFile(cacheFileName + ".glsl", debug);
            // writeFile(cacheFileName + ".txt", compileToAssembly(name, kind, code, optimizeSpirV));
        }
#endif
        compilation = compileToSpirV(T::ShaderName + cacheName, T::ShaderKind, code, OptimizeSpirV);
        const auto size = (compilation.cend() - compilation.cbegin()) * sizeof(uint32_t);
        if (useCache) {
            writeFile(cacheFileName + ".spv", reinterpret_cast<const char *>(compilation.cbegin()), size);
        }
        CadPL::Debug::increment("glslCodeSize", code.size());
        CadPL::Debug::increment("spirvCodeSize", size);
        info.codeSize = size;
        info.pCode = compilation.cbegin();
        auto time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        CadPL::Debug::logAverage("glslToSpirV", time);
    }
    else {
        info.codeSize = spirv.size() * sizeof(uint32_t);
        info.pCode = spirv.data();
    }
    return device.createShaderModule(info);
}

template<typename T>
static std::vector<uint32_t> createShaderCppCode(const ShaderState& state, const std::string &cacheName)
{
    T generator(state);
    const auto code = generator.generate();
    if (!cacheDirectory.empty() && !cacheName.empty()) {
        // debug files
        std::string debug = code;
        debug += "\n/*\n";
        debug += state.debugDump();
        debug += "*/\n";
        auto cacheFileName = ShaderGenerator::createCacheName(cacheDirectory, T::ShaderKind, cacheName);
        writeFile(cacheFileName + ".glsl", debug);
    }
    shaderc::SpvCompilationResult compilation = compileToSpirV(T::ShaderName + cacheName, T::ShaderKind, code, OptimizeSpirV);
    const auto size = (compilation.cend() - compilation.cbegin());
    std::vector<uint32_t> spirv(size);
    std::memcpy(spirv.data(), compilation.cbegin(), size * sizeof(uint32_t));
    return spirv;
}

vk::ShaderModule ShaderGenerator::createVertexShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName)
{
    return createShader<VertexShaderGenerator>(state, device, cacheName);
}

vk::ShaderModule ShaderGenerator::createGeometryShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName)
{
    if (!usesGeometryShader(state)) {
        return {};
    }
    return createShader<GeometryShaderGenerator>(state, device, cacheName);
}

vk::ShaderModule ShaderGenerator::createFragmentShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName)
{
    return createShader<FragmentShaderGenerator>(state, device, cacheName);
}

std::vector<uint32_t> ShaderGenerator::createVertexShaderSpirV(const ShaderState &state, const std::string &cacheName)
{
    return createShaderCppCode<VertexShaderGenerator>(state, cacheName);
}

std::vector<uint32_t> ShaderGenerator::createGeometryShaderSpirV(const ShaderState &state, const std::string &cacheName)
{
    return createShaderCppCode<GeometryShaderGenerator>(state, cacheName);
}

std::vector<uint32_t> ShaderGenerator::createFragmentShaderSpirV(const ShaderState &state, const std::string &cacheName)
{
    return createShaderCppCode<FragmentShaderGenerator>(state, cacheName);
}

bool ShaderGenerator::usesGeometryShader(const ShaderState& state) noexcept
{
    return (state.optimizeFlags.to_ulong() & ShaderState::OptimizeAttribs.to_ulong()) == 0;
}