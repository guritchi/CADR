#include <CadPL/ShaderGenerator.h>
#include <CadPL/ShaderLibrary.h>
#include <CadR/VulkanDevice.h>

#include <ShaderGeneratorHash.hpp>

#include <shaderc/shaderc.hpp>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <variant>

using namespace std;
using namespace CadPL;


static constexpr bool ShaderValidation = false;
static constexpr bool OptimizeSpirV = true;
static constexpr uint32_t SpirvMagicNumber = 0x07230203;
static constexpr size_t CodeStringReservation = 128000; // 128kB


/*
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
*/


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


static std::string cacheDirectory()
{
    std::string name;
#ifdef _WIN32
    name += R"(\\?\)"; // long path
#endif
    name += std::filesystem::current_path().string();
    name += "\\ShaderCache";
    return name;
}

static std::string createCacheName(shaderc_shader_kind kind, const ShaderState& state)
{
    const auto &encoded = state.serialize();
    std::string name = cacheDirectory();
    name += "\\";
    if (!OptimizeSpirV) {
        name += "d";
    }
    name += std::to_string(kind);
    name += encoded;
    return name;
}

void ShaderGenerator::initializeCache()
{
    const auto &dir = cacheDirectory();
    std::string hash;
    {
        std::ifstream hashfile(dir + "\\hash.txt");
        if (hashfile.is_open()) {
            std::getline(hashfile, hash);
            hashfile.close();
        } else {
            std::filesystem::create_directory(dir);
        }
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

static vk::ShaderModule createShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &code, const std::string &name, shaderc_shader_kind kind)
{
    const auto start = std::chrono::system_clock::now();

    vk::ShaderModuleCreateInfo info;
    info.flags = vk::ShaderModuleCreateFlags();

    std::vector<uint32_t> spirv;
    shaderc::SpvCompilationResult compilation;
    const auto cacheFileName = createCacheName(kind, state);
    spirv = readSpirvFromFile(cacheFileName + ".spv");

    if (spirv.empty()) {
#ifndef NDEBUG
        // debug files
        std::string debug = code;
        debug += "\n/*\n";
        debug += state.debugDump();
        debug += "*/\n";
        writeFile(cacheFileName + ".glsl", debug);
        // writeFile(cacheFileName + ".txt", compileToAssembly(name, kind, code, optimizeSpirV));
#endif

        compilation = compileToSpirV(name, kind, code, OptimizeSpirV);
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
    // std::cout << name << " compile: " << std::chrono::duration_cast<std::chrono::microseconds>(spirvTime - start).count()
    //           << "us, createShaderModule(): " << std::chrono::duration_cast<std::chrono::microseconds>(createTime - spirvTime).count() << "us\n";

    return module;
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

    ~OutputLine() {
        _buffer += '\n';
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

    // OutputLine operator()(int indent)
    // {
    //     addIndent(indent);
    //     putIndent();
    //     return OutputLine{_buffer};
    // }
    // OutputLine operator++() {
    // 	return this->operator()(4);
    // }
    //
    // OutputLine operator--() {
    // 	return this->operator()(-4);
    // }

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

static std::string generateOptimizedRead(const AttributeInfo &attrib, const std::string_view address, const std::string_view defaultValue)
{
    std::string output;
    output.reserve(256);
    switch (static_cast<AttributeType>(attrib.type)) {
        case AttributeType::vec4align16:
            output += "AlignedVec4Ref(";
            break;
        case AttributeType::vec3align16:
            output += "AlignedVec3Ref(";
            break;
        case AttributeType::vec3align4:
            output += "UnalignedVec3Ref(";
            break;
        case AttributeType::vec2align8:
            output += "AlignedVec2Ref(";
            break;
        case AttributeType::vec2align4:
            output += "UnalignedVec2Ref(";
            break;
        default:
            output += defaultValue;
            return output;
    }
    output += address;
    output += " + ";
    output += std::to_string(attrib.offset);
    output += ").value";
    return output;
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


static void includeUberShaderReadFuncs(OutputStream &output)
{
    output << R"(
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
)" << R"(
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
)" << R"(


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
    //   - 0x59 - int2, alignment 8
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
)" << R"(


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

    uint offset = settings & 0x00ff;  // max offset is 255
    uint64_t addr = vertexDataPtr + offset;
    uint type = settings & 0xff00;

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
)" << R"(


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
	layout(offset=192) uint lightData[];  // array of OpenGLLight and GltfLight structures is stored here
};
uint getLightDataOffset()  { return 192; }
)";
}

static void generatePushConstants(OutputStream &output, bool optimizeAttribs, bool idBuffer)
{
    output << "// push constants"
           << "layout(push_constant) uniform pushConstants {";
    output.push();
    output << "layout(offset=0) uint64_t sceneDataPtr;  // pointer to SceneDataRef; usually updated per scene render pass or once per scene rendering"
           << "layout(offset=8) uint64_t drawablePointersBufferPtr;  // pointer to DrawablePointersRef array; there is one DrawablePointersRef array for each StateSet, so the pointer is updated before each StateSet rendering";
    // if (!optimizeAttribs) {
        output << "layout(offset=16) uint attribAccessInfoList[8];  // per-stateSet attribAccessInfo for 16 attribs"
               << "layout(offset=48) uint attribSetup;  // doc is provided bellow with getVertexDataSize()"
               << "layout(offset=52) uint materialSetup;  // doc is provided with UnlitMaterialRef, PhongMaterialRef and MetallicRoughnessMaterialRef";
    // }
    if (idBuffer) {
        output << "layout(offset=56) uint stateSetID;  // ID of the current StateSet";
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
uint getMaterialModel()  { return materialSetup & 0x03; }
uint getMaterialFirstTextureOffset()  { return materialSetup & 0xfc; }
bool getMaterialUseColorAttribute()  { return (materialSetup & 0x0300) != 0; }
bool getMaterialUseColorAttributeForAmbientAndDiffuse()  { return (materialSetup & 0x0100) != 0; }
bool getMaterialUseColorAttributeForDiffuseOnly()  { return (materialSetup & 0x0200) != 0; }
bool getMaterialIgnoreColorAttributeAlpha()  { return (materialSetup & 0x0400) != 0; }
bool getMaterialIgnoreMaterialAlpha()  { return (materialSetup & 0x0800) != 0; }
bool getMaterialIgnoreBaseTextureAlpha()  { return (materialSetup & 0x1000) != 0; }
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

/* TODO
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
    layout(offset=32) float metallicFactor;  //< Hoops uses metalnessFactor (in PBRMaterialKit)
    layout(offset=36) float roughnessFactor;  //< Hoops uses RoughnessFactor (in PBRMaterialKit)
    layout(offset=40) float normalTextureScale;  //< Hoops uses NormalFactor (in PBRMaterialKit)
    layout(offset=44) float occlusionTextureStrength;  //< Hoops uses OcclusionFactor (in PBRMaterialKit) with the same meaning
    layout(offset=48) vec3 emissiveFactor;
    layout(offset=60) float pointSize;
    layout(offset=64) float alphaCutoff;

    // 25 additional floats (100 bytes)
    float anisotropyStrength;
    float anisotropyRotation;
    float clearcoatFactor;
    float clearcoatRoughnessFactor;
    float dispersion;
    float emissiveStrength;
    float ior;
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

    // layout(offset=176)
    mat4 modelMatrix[];
};
*/

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
    vec3 diffuse;
    vec3 specular;
    float constantAttenuation;
    float linearAttenuation;
    float quadraticAttenuation;
};

struct GltfLightData {
    vec3 color;
    float intensity;  // in candelas (lm/sr) for point light and spotlight and in luxes (lm/m2) for directional light
    float range;
};

struct SpotlightData {
    vec3 direction;  // spotlight direction in eye coordinates, it must be normalized
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
    layout(offset=80) GltfLightData gltf;
    layout(offset=112) SpotlightData spotlight;
};
uint getLightDataSize()  { return 192; }
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

    explicit MetadataMaterial(const ShaderState& state)
        : optimizeMaterialModel(state.optimizeFlags.to_ulong() & ShaderState::OptimizeMaterialModel.to_ulong())
        , optimizeMaterialColorAttribute(state.optimizeFlags.to_ulong() & ShaderState::OptimizeMaterialColorAttribute.to_ulong())
        , optimizeMaterialAlpha(state.optimizeFlags.to_ulong() & ShaderState::OptimizeMaterialAlpha.to_ulong())
    {

        if (optimizeMaterialModel) {
            materialModel = state.materialSetup & 0x03;
            materialFirstTextureOffset = state.materialSetup & 0xfc;
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
            colorAttribAlpha = AttributeInfo(state.attribAccessInfo[3]).type == static_cast<uint32_t>(AttributeType::vec4align16);

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
    std::array<Texture, 6> textures;
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
        output() << "vertexDataPtr = " << vertexPtrValue << ";";
        if (optimizeAttribs) {
            output() << "position = " << generateOptimizedRead(state.attribAccessInfo[0], "vertexDataPtr", "vec3(0)") << ";";
        }
        else {
            output << "position = readVec3(vertexDataPtr, positionAccessInfo);";
        }

        output << "eyePosition = modelViewMatrix * vec4(position, 1);"
               // multiplication by projection "matrix"
               << "gl_Position.x = scene.p11*eyePosition.x + p31*eyePosition.z + p41*eyePosition.w;"
               << "gl_Position.y = scene.p22*eyePosition.y + p32*eyePosition.z + p42*eyePosition.w;"
               << "gl_Position.z = scene.p33*eyePosition.z + scene.p43*eyePosition.w;"
               << "gl_Position.w = p34*eyePosition.z + p44*eyePosition.w;";
        // set output variables
        if (!vertexShader) {
            output << "outVertexAndDrawableDataPtr = vertexAndDrawableDataPtr;";
        }
        output << "outVertexPosition3 = eyePosition.xyz / eyePosition.w;";

        if (!optimizeAttribs) {
            output() << "outBarycentricCoords = " << barycentricValue << ";";
        }

        if (optimizeAttribs) {
            if (AttributeInfo(state.attribAccessInfo[1]).type != 0) {
                output()  << "outVertexNormal = normalize(mat3(modelViewMatrix) * "
                        << generateOptimizedRead(state.attribAccessInfo[1], "vertexDataPtr", "")
                        << ");";
            }
            else {
                output << "outVertexNormal = vec3(0,0,-1);";
            }
            if (AttributeInfo(state.attribAccessInfo[2]).type != 0) {
                output() << "    outVertexTangent = normalize(mat3(modelViewMatrix) * "
                       << generateOptimizedRead(state.attribAccessInfo[2], "vertexDataPtr", "")
                       << ");";
            }
            else {
                output << "outVertexTangent = vec3(1,0,0);\n";
            }
            if (colorAttrib) {
                output() << "outVertexColor = "
                       << generateOptimizedRead(state.attribAccessInfo[3], "vertexDataPtr", "")
                       << ";\n";
            }
            for (size_t i = 0; i < numTextureAttribs; ++i) {
                output() << "outTexCoords[" << i << "] = "
                       << generateOptimizedRead(state.attribAccessInfo[4 + i], "vertexDataPtr", "")
                       << ";\n";
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

        output() << "#version 460";

        if (!optimizeAttribs) {
            output() << "\n// output to geometry shader";
            output() << "layout(location = 0) out flat int outDrawIndex;";
            output() << "layout(location = 1) out flat int outInstanceIndex;";
            output() << "layout(location = 2) out flat int outVertexIndex;";

            if (state.idBuffer) {
                output << "// ID_BUFFER"
                       << "layout(location = 3) out flat uvec2 outId;";
            }

        }
        else {
            output << "#extension GL_EXT_buffer_reference : require";
            output << "#extension GL_ARB_gpu_shader_int64 : require";

            includeUberShaderReadFuncs(output);
            generateSceneDataInterface(output);
            generatePushConstants(output, optimizeAttribs, state.idBuffer);
            includeVertexDrawableInterface(output);
            generateProjectionConstants(output);

            output << "\n// output to fragment shader";
            output << "layout(location = 0) out flat uint64_t outDrawableDataPtr;";

            int location = 1;

            output() << "layout(location = " << location++ << ") out smooth vec3 outVertexPosition3;  // in eye coordinates";
            output() << "layout(location = " << location++ << ") out smooth vec3 outVertexNormal;  // in eye coordinates";
            output() << "layout(location = " << location++ << ") out smooth vec3 outVertexTangent;  // in eye coordinates";

            if (colorAttrib) {
                output() << "layout(location = " << location++ << ") out smooth " << (colorAttribAlpha? "vec4" : "vec3") << " outVertexColor;  // color";
            }
            if (numTextureAttribs > 0) {
                output() << "layout(location = " << location++ << ") out smooth vec2 outTexCoords[" << static_cast<uint32_t>(numTextureAttribs) << "];  // texture coordinates";
            }
            if (state.idBuffer) {
                output() << "layout(location = " << location++ << ") out flat uvec2 outId; // ID_BUFFER";
            }

        }

        output() << "void main()"
                 << "{";
        output.push();
        if (!optimizeAttribs) {
            output << "outDrawIndex = gl_DrawID;"
                   << "outInstanceIndex = gl_InstanceIndex;"
                   << "outVertexIndex = gl_VertexIndex;";
        }
        else {
            output << "DrawablePointersRef dp = DrawablePointersRef(drawablePointersBufferPtr + (gl_DrawID * DrawablePointersSize));"
                   << "outDrawableDataPtr = dp.drawableDataPtr;"
                   << "IndexDataRef indexData = IndexDataRef(dp.indexDataPtr);"
                   << "uint index = indexData.indices[gl_VertexIndex];"
                   << "// matrices and positions"
                   << "SceneDataRef scene = SceneDataRef(sceneDataPtr);"
                   << "mat4 modelViewMatrix = scene.viewMatrix * getDrawableMatrix(dp.matrixListPtr, gl_InstanceIndex);"
                   << "vec4 eyePosition;"
                   << "vec3 position;"
                   << "uint64_t vertexDataPtr;";
            std::string vertexPtr = "dp.vertexDataPtr + (index * " + std::to_string(vertexSize) + ")";
            generateVertexCode(output, vertexPtr, "vec3(0,0,0)", "gl_DrawID", true);
        }

        if (state.idBuffer) {
            output << "// ID_BUFFER"
                   << "outId[0] = gl_DrawID;"
                   << "outId[1] = gl_InstanceIndex;";
        }
        output.pop();
        output << "}";

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

        output << "#version 460"
               << "#extension GL_EXT_buffer_reference : require"
               << "#extension GL_ARB_gpu_shader_int64 : require";

        // if (!optimizeAttribs) {
        includeUberShaderReadFuncs(output);
        // }
        includeUberShaderInterface(output, false, optimizeAttribs, optimizeAttribs, state.idBuffer);

        output << "layout(triangles) in;"
               << "layout(triangle_strip, max_vertices=3) out;";
               // input from vertex shader
        output << "layout(location = 0) in flat int inDrawIndex[3];"
               << "layout(location = 1) in flat int inInstanceIndex[3];"
               << "layout(location = 2) in flat int inVertexIndex[3];";
        if (state.idBuffer) {
            output << "// ID_BUFFER"
                   << "layout(location = 3) in flat uvec2 inId[3];";
        }
        generateGeometryOutputInterface(output, state.idBuffer);

        generateProjectionConstants(output);

        output << "void main()" << "{";
        output.push();
        // input from vertex shader
        output << "int drawIndex = inDrawIndex[0];"
               << "int instanceIndex = inInstanceIndex[0];";
        // DrawablePointers
        output << "u64vec4 vertexAndDrawableDataPtr;"
               << "DrawablePointersRef dp = DrawablePointersRef(drawablePointersBufferPtr + (drawIndex * DrawablePointersSize));";
        // vertex data
        output << "{";
        output.push();
        output << "IndexDataRef indexData = IndexDataRef(dp.indexDataPtr);"
               << "uint index0 = indexData.indices[inVertexIndex[0]];"
               << "uint index1 = indexData.indices[inVertexIndex[1]];"
               << "uint index2 = indexData.indices[inVertexIndex[2]];"
               << "uint vertexDataSize = getVertexDataSize();"
               << "vertexAndDrawableDataPtr.x = dp.vertexDataPtr + (index0 * vertexDataSize);"
               << "vertexAndDrawableDataPtr.y = dp.vertexDataPtr + (index1 * vertexDataSize);"
               << "vertexAndDrawableDataPtr.z = dp.vertexDataPtr + (index2 * vertexDataSize);"
               << "vertexAndDrawableDataPtr.w = dp.drawableDataPtr;";
        output.pop();
        output << "}";
        // matrices and positions
        output << "SceneDataRef scene = SceneDataRef(sceneDataPtr);";
        output << "mat4 modelViewMatrix = scene.viewMatrix * getDrawableMatrix(dp.matrixListPtr, instanceIndex);";
        // first vertex
        output << "vec4 eyePosition;"
               << "vec3 position;"
               << "uint64_t vertexDataPtr;";

        if (!optimizeAttribs) {
            // vertex positions (it is stored on offset 0 by convention)
            output << "const uint positionAccessInfo = getPositionAccessInfo();"
                   << "const uint normalAccessInfo = getNormalAccessInfo();"
                   << "const uint tangentAccessInfo = getTangentAccessInfo();";
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
        if (!optimizeTextures) {
            uint32_t typeValue = static_cast<int>(type) << 8;
            output() << "if(textureType == 0x" << toHex(typeValue) << ")\n";
        }
#ifndef NDEBUG
        else if (ShaderValidation && texture) {
            uint32_t typeValue = static_cast<int>(type) << 8;
            output() << "if((textureInfo.texCoordIndexTypeAndSettings & 0xff00) != 0x" << toHex(typeValue) << ") {\n";
            output << "    outColor = vec4(1, 0, 1, 1);";
            output << "    return;";
            output << "} else\n";
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
                else {
                    assert(numTextureAttribs > 0 && "missing texCoords");
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
                "tangentSpaceNormal *= vec3(textureInfo.strength, textureInfo.strength, 1);"
            );

            output << "tangentSpaceNormal = normalize(tangentSpaceNormal);";
            // transform normal
            output << "vec3 t = normalize(inFragmentTangent);"
                   << "mat3 tbn = { t, cross(normal, t), normal };"
                   << "normal = tbn * tangentSpaceNormal;";
        });

    }

    void generateBaseTexture(OutputStream &output)
    {
        generateTexture(output, TextureType::base, "vec4 baseTextureValue = ", "", [&](OutputStream &output, const Texture &texture){

            texture.generateStrengthFlagCode(output, optimizeTextures,
                "baseTextureValue *= textureInfo.strength;\n"
            );

            texture.generateTextureEnvironmentCode(output, optimizeTextures); // TODO

            // TODO optimize
            generateOptimizedIf(output, optimizeMaterialAlpha, materialIgnoreBaseTextureAlpha,
                "getMaterialIgnoreBaseTextureAlpha()",
                [&](OutputStream &output) {
                    if (!optimizeTextures) {
                        output << "if(texEnv == 0)  // modulate\n";
                        output.push();
                        output << "outColor.rgb *= baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 1) // replace\n";
                        output.push();
                        output << "outColor.rgb = baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 2) // decal\n";
                        output.push();
                        output << "outColor.rgb = baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 3) // blend\n";
                        output.push();
                        output << "outColor.rgb = outColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb;\n";
                        output.pop();
                        output << "else if(texEnv == 4) // add\n";
                        output.push();
                        output << "outColor.rgb = outColor.rgb + baseTextureValue.rgb;\n";
                        output.pop();
                    }
                    else {
                        auto texEnv = texture.getTextureEnvironment();
                        switch (texEnv) {
                            case 0:
                                output << "outColor.rgb *= baseTextureValue.rgb;\n";
                                break;
                            case 1:
                                output << "outColor.rgb = baseTextureValue.rgb;\n";
                                break;
                            case 2:
                                output << "outColor.rgb = baseTextureValue.rgb;\n";
                                break;
                            case 3:
                                output << "outColor.rgb = outColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb;\n";
                                break;
                            case 4:
                                output << "outColor.rgb = outColor.rgb + baseTextureValue.rgb;\n";
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
                        output << "outColor *= baseTextureValue;\n";
                        output.pop();
                        output << "else if(texEnv == 1) // replace\n";
                        output.push();
                        output << "outColor = vec4(baseTextureValue.rgb, baseTextureValue.a * outColor.a);\n";
                        output.pop();
                        output << "else if(texEnv == 2) // decal\n";
                        output.push();
                        output << "outColor = vec4(outColor.rgb*(1-baseTextureValue.a) + baseTextureValue.rgb*baseTextureValue.a, outColor.a);\n";
                        output.pop();
                        output << "else if(texEnv == 3) // blend\n";
                        output.push();
                        output << "outColor = vec4(outColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb, outColor.a*baseTextureValue.a);\n";
                        output.pop();
                        output << "else if(texEnv == 3) // add\n";
                        output.push();
                        output << "outColor = vec4(outColor.rgb + baseTextureValue.rgb, outColor.a * baseTextureValue.a);\n";
                        output.pop();
                    }
                    else {
                        auto texEnv = texture.getTextureEnvironment();
                        switch (texEnv) {
                            case 0:
                                output << "outColor *= baseTextureValue;\n";
                                break;
                            case 1:
                                output << "outColor = vec4(baseTextureValue.rgb, baseTextureValue.a * outColor.a);\n";
                                break;
                            case 2:
                                output << "outColor = vec4(outColor.rgb*(1-baseTextureValue.a) + baseTextureValue.rgb*baseTextureValue.a, outColor.a);\n";
                                break;
                            case 3:
                                output << "outColor = vec4(outColor.rgb*(1-baseTextureValue.rgb) + getTextureBlendColor(textureInfo)*baseTextureValue.rgb, outColor.a*baseTextureValue.a);\n";
                                break;
                            case 4:
                                output << "outColor = vec4(outColor.rgb + baseTextureValue.rgb, outColor.a * baseTextureValue.a);\n";
                                break;
                            default:
                                break;
                        }
                    }
                }
            );

        });

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

    void generatePhongColor(OutputStream &output)
    {

        generateOptimizedIf(output, optimizeMaterialColorAttribute, materialUseColorAttribute,
            "getMaterialUseColorAttribute()",
            [&](OutputStream &output){

                if (!optimizeAttribs) {
                    output << "uint colorAccessInfo = getColorAccessInfo();\n";
                }
                generateOptimizedIf(output, optimizeMaterialColorAttribute, materialIgnoreColorAttributeAlpha,
                    "getMaterialIgnoreColorAttributeAlpha()",
                    [&](OutputStream &output){
                        if (!optimizeAttribs) {
                            output << "vec3 color =\n";
                            output << "    readVec3(vertex0DataPtr, colorAccessInfo) * inBarycentricCoords.x +\n";
                            output << "    readVec3(vertex1DataPtr, colorAccessInfo) * inBarycentricCoords.y +\n";
                            output << "    readVec3(vertex2DataPtr, colorAccessInfo) * inBarycentricCoords.z;\n";
                            output << "diffuseColor = color;\n";
                        }
                        else if (colorAttrib) {
                            output << "diffuseColor = inFragmentColor.rgb;\n";
                        }
                        else {
                            output << "diffuseColor = vec3(1.0);\n";
                        }
                        generateOptimizedIf(output, optimizeMaterialAlpha, materialIgnoreMaterialAlpha,
                            "getMaterialIgnoreMaterialAlpha()",
                            "outColor.a = 1.0;\n",
                            "outColor.a *= materialData.diffuseAndAlpha.a;\n"
                        );
                    },
                    [&](OutputStream &output){
                        if (!optimizeAttribs) {
                            output << "vec4 color =\n";
                            output << "    readVec4(vertex0DataPtr, colorAccessInfo) * inBarycentricCoords.x +\n";
                            output << "    readVec4(vertex1DataPtr, colorAccessInfo) * inBarycentricCoords.y +\n";
                            output << "    readVec4(vertex2DataPtr, colorAccessInfo) * inBarycentricCoords.z;\n";
                            output << "diffuseColor = color.rgb;\n";
                            output << "outColor.a = color.a;\n";
                        }
                        else if (colorAttrib) {
                            output << "diffuseColor = inFragmentColor.rgb;\n";
                            if (colorAttribAlpha) {
                                output << "outColor.a = inFragmentColor.a;\n";
                            }
                            else {
                                output << "outColor.a = 1.0;\n";
                            }
                        }
                        else {
                            output << "diffuseColor = vec3(1.0);\n";
                            output << "outColor.a = 1.0;\n";
                        }
                        generateOptimizedIf(output, optimizeMaterialAlpha, !materialIgnoreMaterialAlpha,
                            "!getMaterialIgnoreMaterialAlpha()",
                            "outColor.a *= materialData.diffuseAndAlpha.a;\n"
                        );
                    }
                );
                generateOptimizedIf(output, optimizeMaterialColorAttribute, materialUseColorAttributeForAmbientAndDiffuse,
                    "getMaterialUseColorAttributeForAmbientAndDiffuse()",
                    "ambientColor = diffuseColor;\n",
                    "ambientColor = materialData.ambient;\n"
                );
            },
            [&](OutputStream &output){

                output << "ambientColor = materialData.ambient;\n";
                output << "diffuseColor = materialData.diffuseAndAlpha.rgb;\n";
                generateOptimizedIf(output, optimizeMaterialAlpha, materialIgnoreMaterialAlpha,
                    "getMaterialIgnoreMaterialAlpha()",
                    "outColor.a = 1;\n",
                    "outColor.a = materialData.diffuseAndAlpha.a;\n"
                );;

            }
        );

    }

    void generatePhongModel(OutputStream &output)
    {

        generateOcclusionTexture(output);
        generateEmissiveTexture(output);

        // material data
        output << "PhongMaterialRef materialData = PhongMaterialRef(drawableDataPtr);\n";
        // ambientColor, diffuseColor and alpha
        output << "vec3 ambientColor;\n";
        output << "vec3 diffuseColor;\n";

        generatePhongColor(output);
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
                output += " + diffuseProduct * diffuseColor + specularProduct * materialData.specular";
            }

            return output;
        };

        // light data
        output << "uint64_t lightDataPtr = sceneDataPtr + getLightDataOffset();"

               << "LightRef lightData = LightRef(lightDataPtr);";
        // iterate over all light sources
        output << "uint lightSettings = lightData.settings;"
               << "if(lightSettings != 0) {";
        output.push();
        // Phong color products

        output << "vec3 ambientProduct  = vec3(0);"
               << "vec3 diffuseProduct  = vec3(0);"
               << "vec3 specularProduct = vec3(0);";
        // iterate over all lights
        output << "do{";
        output.push();

        output << "uint lightType = lightSettings & 0x3;"
               << "if(lightType == 1)";
        output.push();
        output << "OpenGLDirectionalLight(lightData, normal,"
               << "    viewerToFragmentDirection, materialData.shininess,"
               << "    ambientProduct, diffuseProduct, specularProduct);";
        output.pop();
        output << "else if(lightType == 2)";
        output.push();
        output << "OpenGLPointLight(lightData, normal,"
               << "    viewerToFragmentDirection, materialData.shininess,"
               << "    ambientProduct, diffuseProduct, specularProduct);";
        output.pop();
        output << "else";
        output.push();
        output << "OpenGLSpotlight(lightData, normal,"
               << "    viewerToFragmentDirection, materialData.shininess,"
               << "    ambientProduct, diffuseProduct, specularProduct);";
        output.pop();
        output << "lightDataPtr += getLightDataSize();"
               << "lightData = LightRef(lightDataPtr);"
               << "lightSettings = lightData.settings;";

        output.pop();
        output << "} while(lightSettings != 0);";

        // Phong equation
        output() << "outColor.rgb = " << colorCalculation(true) << ";\n";

        output.pop();
        output << "} else {\n";
        output.push();
        // Phong equation without light sources
        output() << "outColor.rgb = " << colorCalculation(false) << ";\n";
        output.pop();
        output << "}\n";

        generateBaseTexture(output);

    }

    void generateGltfPBRModel(OutputStream &output)
    {

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
        output << "void main()" << "{";
        output.push();
        // input data and structures
        output << "SceneDataRef scene = SceneDataRef(sceneDataPtr);";

        if (!optimizeAttribs) {
            output << "uint64_t vertex0DataPtr  = inVertexAndDrawableDataPtr.x;"
                   << "uint64_t vertex1DataPtr  = inVertexAndDrawableDataPtr.y;"
                   << "uint64_t vertex2DataPtr  = inVertexAndDrawableDataPtr.z;"
                   << "uint64_t drawableDataPtr = inVertexAndDrawableDataPtr.w;";
        }
        else {
            output << "uint64_t drawableDataPtr = inDrawableDataPtr;";
        }
        output << "vec3 viewerToFragmentDirection = normalize(inFragmentPosition3);";

        // TODO getGenerateFlatNormals, getMaterialTwoSidedLighting()
        if (!optimizeMaterialModel) {
            // normal
            output << "vec3 normal;"
                   << "uint materialModel = getMaterialModel();"
                   << "if(materialModel >= 1) {";
            {
                output.push();
                output << "normal = normalize(inFragmentNormal);";
                output << "}";
                output.pop();
            }
        }
        else if (materialModel >= 1) {
            // normal
            output << "vec3 normal = normalize(inFragmentNormal);";
        }

        bool hasTextures = numTextureAttribs > 0 || (!optimizeTextures && !optimizeAttribs) || true;
        if (hasTextures) {
            if (optimizeMaterialModel) {
                if (materialFirstTextureOffset) {
                    output() << "TextureInfoRef textureInfo = TextureInfoRef(drawableDataPtr + " << materialFirstTextureOffset << ");";
                }
            }
            else {
                output << "uint textureOffset = getMaterialFirstTextureOffset();"
                       << "TextureInfoRef textureInfo;";
            }
            if (!optimizeTextures) {
                output << "uint textureType = 0;";
            }
            if (!optimizeMaterialModel) {
                output << "if (textureOffset != 0) {";
                output.push();
                output << "textureInfo = TextureInfoRef(drawableDataPtr + textureOffset);";
            }
            if (!optimizeTextures) {
                output << "textureType = textureInfo.texCoordIndexTypeAndSettings & 0xff00;";
            }
            if (!optimizeMaterialModel) {
                output.pop();
                output << "}";
            }
        }

        generateNormalTexture(output);

        generateMaterialModel(output, 0, false, &FragmentShaderGenerator::generateUnlitModel,   "    // unlit material\n");
        generateMaterialModel(output, 1, true,  &FragmentShaderGenerator::generatePhongModel,   "    // Blinn-Phong material model, implemented in the similar way as OpenGL does\n");
        generateMaterialModel(output, 2, true,  &FragmentShaderGenerator::generateGltfPBRModel, "    // Metallic-Roughness material model from PBR family,\n""    // implemented in the similar way as glTF doesl\n");

        if (state.idBuffer) {
            // write id-buffer
            output << "outId[0] = stateSetID;"
                   << "outId[1] = inId[0];  // gl_DrawID - index of indirect drawing structure"
                   << "outId[2] = inId[1];  // gl_InstanceIndex"
                   << "outId[3] = gl_PrimitiveID;";
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
            includeUberShaderReadFuncs(output);
        }
        includeUberShaderInterface(output, false, optimizeTextures, optimizeAttribs, state.idBuffer);

        generateFragmentInputInterface(output, state.idBuffer);

        output << "layout(location = 0) out vec4 outColor;\n";
        if (state.idBuffer) {
            output << "// ID_BUFFER";
            output << "layout(location = 1) out uvec4 outId;\n";
        }
        // textures
        output << "layout(set=0, binding=0) uniform sampler2D textureDB[];\n";

        generateOpenGLLightFunctions(output);

        generateMain(output);

        return std::move(output.string());
    }
};

template<typename T>
static vk::ShaderModule createShader(const ShaderState& state, CadR::VulkanDevice& device)
{
    T generator(state);
    const auto start = std::chrono::system_clock::now();
    const auto code = generator.generate();
    auto shader = createShader(state, device, code, T::ShaderName, T::ShaderKind);
    const auto end = std::chrono::system_clock::now();
    std::cout << "create " << T::ShaderName << ": " << static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0 << "ms\n";
    return shader;
}

vk::ShaderModule ShaderGenerator::createVertexShader(const ShaderState& state, CadR::VulkanDevice& device)
{
    return createShader<VertexShaderGenerator>(state, device);
}

vk::ShaderModule ShaderGenerator::createGeometryShader(const ShaderState& state, CadR::VulkanDevice& device)
{
    const bool optimizeAttribs = state.optimizeFlags.to_ulong() & ShaderState::OptimizeAttribs.to_ulong();
    if (optimizeAttribs) {
        return {};
    }
    return createShader<GeometryShaderGenerator>(state, device);
}

vk::ShaderModule ShaderGenerator::createFragmentShader(const ShaderState& state, CadR::VulkanDevice& device)
{
    return createShader<FragmentShaderGenerator>(state, device);
}
