#include <CadPL/ShaderGenerator.h>
#include <CadPL/ShaderLibrary.h>
#include <CadR/VulkanDevice.h>

#include <shaderc/shaderc.hpp>

#include <iostream>

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
