#pragma once

#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp> // shaderc_shader_kind
#include <string>
#include <string_view>

namespace CadR {
class VulkanDevice;
}

namespace CadPL {

struct ShaderState;

class CADPL_EXPORT ShaderGenerator {
public:

	static std::string defaultCacheDirectory();
	static std::string& getCacheDirectory();
	static std::string createCacheName(std::string_view directory, shaderc_shader_kind kind, const std::string& state);
	static void initialize();
	static void initializeCache(std::string_view directory = defaultCacheDirectory(), bool wipe = false);
	static void enableSpirVCache(bool enabled);
	static bool useSpirVCache();
	static void savePipelineCache(const CadR::VulkanDevice& device, vk::PipelineCache cache, const vk::PhysicalDeviceProperties &properties, std::string_view directory);
	static void savePipelineCache(const CadR::VulkanDevice& device, vk::PipelineCache cache, const vk::PhysicalDeviceProperties &properties);
	static std::vector<uint8_t> loadPipelineCacheData(const vk::PhysicalDeviceProperties &properties, std::string_view filename);
	static vk::PipelineCache loadPipelineCache(const CadR::VulkanDevice& device, const vk::PhysicalDeviceProperties &properties, vk::PipelineCacheCreateFlags flags, std::string_view filename, size_t *loadedCacheSize = nullptr);
	static vk::PipelineCache loadPipelineCache(const CadR::VulkanDevice& device, const vk::PhysicalDeviceProperties &properties, vk::PipelineCacheCreateFlags flags, size_t *loadedCacheSize = nullptr);

	static bool usesGeometryShader(const ShaderState& state) noexcept;

	// [[nodiscard]] static ShaderSet createShaderSet(const ShaderState& state, CadR::VulkanDevice& device);
	[[nodiscard]] static vk::ShaderModule createVertexShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName);
	[[nodiscard]] static vk::ShaderModule createGeometryShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName);
	[[nodiscard]] static vk::ShaderModule createFragmentShader(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName);
	static vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice> createVertexShaderUnique(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName);
	static vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice> createGeometryShaderUnique(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName);
	static vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice> createFragmentShaderUnique(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName);

	static std::vector<uint32_t> createVertexShaderSpirV(const ShaderState &state, const std::string &cacheName);
	static std::vector<uint32_t> createGeometryShaderSpirV(const ShaderState &state, const std::string &cacheName);
	static std::vector<uint32_t> createFragmentShaderSpirV(const ShaderState &state, const std::string &cacheName);
};


// inline functions
inline vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice> ShaderGenerator::createVertexShaderUnique(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName)  { return vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice>(createVertexShader(state, device, cacheName)); }
inline vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice> ShaderGenerator::createGeometryShaderUnique(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName)  { return vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice>(createGeometryShader(state, device, cacheName)); }
inline vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice> ShaderGenerator::createFragmentShaderUnique(const ShaderState& state, CadR::VulkanDevice& device, const std::string &cacheName)  { return vk::UniqueHandle<vk::ShaderModule, CadR::VulkanDevice>(createFragmentShader(state, device, cacheName)); }


}
