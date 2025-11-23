#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <future>
#include <map>

#include "CadR/VulkanDevice.h"
#include <ShaderGeneratorHash.hpp>
#include "ShaderGenerator.h"
#include "ShaderState.h"

// #ifndef CADPL_USE_PREGEN
// #error "CadPL library was not build with CADPL_GENERATOR_PREGEN enabled"
// #endif

using namespace CadPL;


static std::map<std::string, std::future<std::size_t>> shaders;
static std::string_view outputDirectory;


static std::string optimizeFlagsToString(std::bitset<ShaderState::numOptimizeFlags> optimizeFlags)
{
    auto flags = optimizeFlags.to_ulong();
    std::string_view prefix = "ShaderState::Optimize";
    std::string out;
    if (flags == ShaderState::OptimizeNone.to_ulong()) {
        out += prefix;
        out += "None";
    }
    else if ((flags & ShaderState::OptimizeAll.to_ulong()) == ShaderState::OptimizeAll.to_ulong()) {
        out += prefix;
        out += "All";
    }
    else {
        if (flags & ShaderState::OptimizeAttribs.to_ulong()) {
            out += prefix;
            out += "Attribs|";
        }
        if ((flags & ShaderState::OptimizeMaterial.to_ulong()) == ShaderState::OptimizeMaterial.to_ulong()) {
            out += prefix;
            out += "Material|";
        }
        else {
            if (flags & ShaderState::OptimizeMaterialModel.to_ulong()) out += "MaterialModel|";
            if (flags & ShaderState::OptimizeMaterialColorAttribute.to_ulong()) out += "MaterialColorAttribute|";
            if (flags & ShaderState::OptimizeMaterialAlpha.to_ulong()) out += "MaterialAlpha|";
        }
        if ((flags & ShaderState::OptimizeTextures.to_ulong()) == ShaderState::OptimizeTextures.to_ulong()) {
            out += prefix;
            out += "Textures|";
        }
        else {
            if (flags & ShaderState::OptimizeTextureFlags.to_ulong()) out += "TextureFlags|";
            if (flags & ShaderState::OptimizeTextureTypesAndTexCoordIndices.to_ulong()) out += "TextureTypesAndTexCoordIndices|";
        }
        if ((flags & ShaderState::OptimizeLights.to_ulong()) == ShaderState::OptimizeLights.to_ulong()) {
            out += prefix;
            out += "Lights|";
        }
        else {
            if (flags & ShaderState::OptimizeLightTypes.to_ulong()) out += "LightTypes|";
        }
        if (*out.rbegin() == '|') {
            out.erase(out.begin() + out.size() - 1);
        }
    }
    return out;
}


void stateToCode(const ShaderState &state, std::stringstream &output)
{
    std::string_view var = "      state";
    output << var << ".optimizeFlags = " << optimizeFlagsToString(state.optimizeFlags) << ";\n";
    output << var << ".primitiveTopology = vk::PrimitiveTopology::e" << vk::to_string(state.primitiveTopology) << ";\n";;

    output << var << ".projectionHandling = ";
    switch (state.projectionHandling ) {
        case ShaderState::ProjectionHandling::SceneMatrix:
            output << "ShaderState::ProjectionHandling::SceneMatrix";
            break;
        case ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants:
            output << "ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants";
            break;
        default:
            throw std::runtime_error("Unhandled value");
    }
    output << ";\n";
    if (state.idBuffer) {
        output << var << ".idBuffer = true;\n";
    }

}

// static size_t shaderIndex = 0;

// return std::async(std::launch::async, &ShaderLibrary::createAsync<VertexShaderMapKey>, this, state, &ShaderGenerator::createVertexShader, &shader);

// if (it->second.empty()) {
//     throw std::runtime_error("Bad shader generated (" + str + ")");
// }

static size_t createShader(std::vector<uint32_t>(*create)(const ShaderState &state, const std::string &cacheName), const ShaderState &state, const std::string &name) {
    auto spirv = create(state, name);
    std::cout << "  " << name << ": " << spirv.size() * sizeof(uint32_t) << "B" << std::endl;

    std::string fileName{outputDirectory};
    fileName += "/";
    fileName += name;
    fileName += ".spv";
    std::ofstream file(fileName, std::ios::out);
    if (file.is_open()) {
        for (const uint32_t &s : spirv) {
            file << "0x" << std::hex << std::setfill('0') << std::setw(8) << s << std::dec << ',';
        }
        file.close();
    }
    else {
        throw std::runtime_error("Error writing file: " + fileName);
    }

    return spirv.size();
}

template<typename Key, shaderc_shader_kind kind>
static void addShader(std::vector<uint32_t>(*create)(const ShaderState &state, const std::string &cacheName), const ShaderState &state, const std::string_view keyType, const std::string_view mapName, std::stringstream &code, std::stringstream &initFunction)
{
    Key key(state);
    auto str = ShaderGenerator::createCacheName("", kind, key.serialize());
    auto [it, new_record] = shaders.try_emplace(str);
    if (new_record) {
        it->second = std::async(std::launch::async, createShader, create, state, str);

        auto var = "spirv_" + str;
        code << "  static constexpr uint32_t " << var << "[]={\n";
        code << "  #include \"shaders/" << str << ".spv\"\n";
        code << "  };\n";

        initFunction << "  {\n";
        initFunction << "    ShaderState state = {};\n";
        stateToCode(state, initFunction);
        initFunction << "    " << mapName << "[" << keyType << "(state)] = std::make_pair(" << var << ",static_cast<uint32_t>(sizeof(" << var << ")));";
        initFunction << "  }\n";
    }
}

int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            std::cerr << "Pregen missing arguments\n";
            return 2;
        }
        outputDirectory = argv[1];
        std::cout << "Running pregen (" << GEN_HASH << ") to " << outputDirectory << std::endl;
        std::filesystem::create_directory(outputDirectory);

        ShaderGenerator::initializeCache(outputDirectory);

        std::stringstream code;
        std::stringstream initFunction;

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        code << "#pragma once\n\n";
        code << "// Generated at " << std::put_time(&tm, "%d-%m-%Y %H:%M:%S") << ", hash is: " << GEN_HASH "\n\n";

        code << "#include <map>\n";
        code << "#include \"ShaderLibrary.h\"\n";

        code << "\nnamespace CadPL {\n";
        code << "  static std::map<VertexShaderState, std::pair<const uint32_t*, uint32_t>> spirvShaderMapVertex;\n";
        code << "  static std::map<GeometryShaderState, std::pair<const uint32_t*, uint32_t>> spirvShaderMapGeometry;\n";
        code << "  static std::map<FragmentShaderState, std::pair<const uint32_t*, uint32_t>> spirvShaderMapFragment;\n";

        struct Option {
        };

        {
            auto optimizeFlags = ShaderState::OptimizeNone;

            auto primitiveTopologies = std::array{
                vk::PrimitiveTopology::ePointList,
                vk::PrimitiveTopology::eLineList,
                vk::PrimitiveTopology::eTriangleList
            };
            auto projectionHandlings = std::array{
                ShaderState::ProjectionHandling::SceneMatrix,
                ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants
            };
            auto idBuffers = std::array {
                true,
                false
            };

            for (const auto &primitiveTopology : primitiveTopologies) {
                for (const auto &projectionHandling : projectionHandlings) {
                    for (const auto &idBuffer : idBuffers) {
                        ShaderState state = {};
                        state.optimizeFlags = optimizeFlags;
                        state.primitiveTopology = primitiveTopology;
                        state.projectionHandling = projectionHandling;
                        state.idBuffer = idBuffer;

                        addShader<VertexShaderState, shaderc_vertex_shader>(&ShaderGenerator::createVertexShaderSpirV, state, "VertexShaderState", "spirvShaderMapVertex",   code, initFunction);
                        if (ShaderGenerator::usesGeometryShader(state)) {
                            addShader<GeometryShaderState, shaderc_geometry_shader>(&ShaderGenerator::createGeometryShaderSpirV, state, "GeometryShaderState", "spirvShaderMapGeometry", code, initFunction);
                        }
                        addShader<FragmentShaderState, shaderc_fragment_shader>(&ShaderGenerator::createFragmentShaderSpirV, state, "FragmentShaderState", "spirvShaderMapFragment", code, initFunction);
                    }
                }
            }
        }

        size_t size = 0;
        for (auto &it : shaders) {
            size += it.second.get();
        }
        std::cout << "  Pregen total shaders: " << size * sizeof(uint32_t) << "B" << std::endl;

        code << "  void initializePregeneratedShaders(){\n";
        code << initFunction.str();
        code << "  }\n";

        code << "\n}\n"; // namespace CadPL

        std::string fileName{outputDirectory};
        fileName += "/PregeneratedShaders.hpp";
        std::ofstream file(fileName, std::ios::out);
        if (file.is_open()) {
            file << code.str();
            file.close();
        }
        else {
            std::cerr << "Error writing file: " << std::strerror(errno) << '\n';
            return 3;
        }

    } catch(std::exception &e) {
        std::cout << "Failed because of exception: " << e.what() << std::endl;
        return 1;
    } catch(...) {
        std::cout << "Failed because of unspecified exception." << std::endl;
        return 1;
    }
    return 0;
}
