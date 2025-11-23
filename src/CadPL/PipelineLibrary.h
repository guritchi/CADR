#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <array>
#include <list>
#include <map>
#include <future>
#include <iostream>
#include <tuple>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <glm/mat4x4.hpp>
#include <CadPL/ShaderLibrary.h>
#include <CadR/Pipeline.h>

#include <iostream>

#include "PipelineLibrary.h"

namespace CadR {
class VulkanDevice;
}

namespace CadPL {

// forward declarations
class SharedPipeline;
class PipelineFamily;
class PipelineLibrary;


struct PipelineState {

	enum class ViewportAndScissorHandling {
		Value,  //< viewport and scissor is specified by the value of PipelineState::viewport and PipelineState::scissor
		DynamicState,  //< viewport and scissor is specified by Vulkan dynamic state; values of PipelineState::viewport and PipelineState::scissor are ignored
		SetFunction,  //< PipelineState::viewport and PipelineState::scissor values are set each time PipelineLibrary::setProjectionViewportAndScissor() is called
	};
	ViewportAndScissorHandling viewportAndScissorHandling = ViewportAndScissorHandling::SetFunction;
	unsigned projectionIndex = 0;  //< When ShaderState::projectionHandling is set to ProjectionHandling::PerspectivePushAndSpecializationConstants, it is the index into projection matrix list passed as parameter into PipelineLibrary::setProjectionViewportAndScissor() function.
	unsigned viewportIndex = 0;  //< When PipelineState::viewportAndScissorHandling is set to ViewportAndScissorHandling::SetFunction, it is the index into viewport list passed as parameter into PipelineLibrary::setProjectionViewportAndScissor() function.
	unsigned scissorIndex = 0;  //< When PipelineState::viewportAndScissorHandling is set to ViewportAndScissorHandling::SetFunction, it is the index into scissor list passed as parameter into PipelineLibrary::setProjectionViewportAndScissor() function.
	vk::Viewport viewport;  //< Viewport set by the user, or by PipelineLibrary::setProjectionViewportAndScissor() if PipelineState::viewportAndScissorHandling is set to ViewportAndScissorHandling::SetFunction.
	vk::Rect2D scissor;  //< Scissor set by the user, or by PipelineLibrary::setProjectionViewportAndScissor() if PipelineState::viewportAndScissorHandling is set to ViewportAndScissorHandling::SetFunction.

	vk::CullModeFlagBits cullMode = vk::CullModeFlagBits::eBack;
	vk::FrontFace frontFace = vk::FrontFace::eCounterClockwise;
	bool depthBiasDynamicState = false;
	bool depthBiasEnable = false;
	float depthBiasConstantFactor;
	float depthBiasClamp;
	float depthBiasSlopeFactor;
	bool lineWidthDynamicState = false;
	float lineWidth = 1.f;
	vk::SampleCountFlagBits rasterizationSamples = vk::SampleCountFlagBits::e1;
	bool sampleShadingEnable = false;
	float minSampleShading;
	bool depthTestEnable = true;
	bool depthWriteEnable = true;

	struct BlendAttachmentState {
		bool blendEnable = false;
		vk::BlendFactor srcColorBlendFactor = vk::BlendFactor::eZero;
		vk::BlendFactor dstColorBlendFactor = vk::BlendFactor::eZero;
		vk::BlendOp colorBlendOp = vk::BlendOp::eAdd;
		vk::BlendFactor srcAlphaBlendFactor = vk::BlendFactor::eZero;
		vk::BlendFactor dstAlphaBlendFactor = vk::BlendFactor::eZero;
		vk::BlendOp alphaBlendOp = vk::BlendOp::eAdd;
		vk::ColorComponentFlags colorWriteMask =
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

		bool operator<(const BlendAttachmentState& rhs) const;
	};
	std::vector<BlendAttachmentState> blendState;

	vk::RenderPass renderPass = nullptr;
	uint32_t subpass = 0;

	bool operator<(const PipelineState& rhs) const;

};


class CADPL_EXPORT SharedPipeline {
protected:
	void* _pipelineObject = nullptr;
public:

	// construction and destruction
	SharedPipeline() = default;
	~SharedPipeline() noexcept;

	// copy and move constructors and operators
	SharedPipeline(SharedPipeline&& other) noexcept;
	SharedPipeline(const SharedPipeline& other) noexcept;
	SharedPipeline& operator=(SharedPipeline&& rhs) noexcept;
	SharedPipeline& operator=(const SharedPipeline& rhs) noexcept;

	// functions
	void reset() noexcept;

	// getters
	const CadR::Pipeline* cadrPipeline() const;
	const PipelineFamily* pipelineFamily() const;
	const PipelineState* pipelineState() const;

protected:
	friend PipelineFamily;
	friend PipelineLibrary;
	SharedPipeline(void* pipelineOwner) noexcept;
	void replacePipelineHandle(vk::Pipeline pipeline, CadR::VulkanDevice& device) noexcept;
};


class CADPL_EXPORT PipelineFamily {
protected:

	CadR::VulkanDevice* _device;
	PipelineLibrary* _pipelineLibrary;
	std::mutex mutex; // need to exist
	std::map<ShaderState, PipelineFamily>::iterator _mapIterator;

	SharedShaderModule _vertexShader;
	SharedShaderModule _geometryShader;
	SharedShaderModule _fragmentShader;
	// vk::ShaderModuleIdentifierEXT _vertexShaderIdentifier;
	// vk::ShaderModuleIdentifierEXT _geometryShaderIdentifier;
	// vk::ShaderModuleIdentifierEXT _fragmentShaderIdentifier;
	vk::PrimitiveTopology _primitiveTopology;

	struct PipelineObject {
		size_t referenceCounter;
		CadR::Pipeline cadrPipeline;
		PipelineFamily* pipelineFamily;
		std::map<PipelineState, PipelineObject>::iterator mapIterator;
		std::mutex mutex;
		bool _compilePending;

		bool tryEnqueuePending() {
			std::unique_lock lock(mutex);
			if (_compilePending) {
				return false;
			}
			_compilePending = true;
			return true;
		}

		bool isCompilePending() {
		 	std::unique_lock lock(mutex);
			if (_compilePending) {
				return true;
			}
		 	return _compilePending;
		}

		bool clearCompilePending() {
			std::unique_lock lock(mutex);
			bool value = _compilePending;
			_compilePending = false;
			return value;
		}
	};

	std::map<PipelineState, PipelineObject> _pipelineMap;

	static void refPipeline(void* pipelineObject) noexcept;
	static void unrefPipeline(void* pipelineObject) noexcept;
	static void destroyPipeline(void* pipelineObject) noexcept;

	void initializeRecord(std::map<PipelineState, PipelineObject>::iterator &it);

	friend SharedPipeline;
	friend PipelineLibrary;

public:

	PipelineFamily() = delete;
	PipelineFamily(PipelineLibrary& pipelineLibrary) noexcept;
	~PipelineFamily() noexcept;

	void initialize(const ShaderState& shaderState, std::map<ShaderState, PipelineFamily>::iterator it, bool createShaders = true);

	SharedPipeline getOrCreatePipeline(const PipelineState& pipelineState, bool &canCreate);
	SharedPipeline getOrCreatePipeline(const PipelineState& pipelineState);
	SharedPipeline getPipeline(const PipelineState& pipelineState);
	const std::map<PipelineState, PipelineObject>& pipelineMap() const;

	const ShaderState& shaderState() const;

	size_t count() noexcept;

	vk::Pipeline createPipeline(const PipelineState& pipelineState);

};


class PipelineLibraryAsyncConsumer {

public:
	virtual ~PipelineLibraryAsyncConsumer() = default;

	virtual void pipelineCreated(SharedPipeline pipeline, void *userData) = 0;
};

class CADPL_EXPORT PipelineBinaryCache {

	const CadR::VulkanDevice* _device = {};
	PFN_vkDestroyPipelineBinaryKHR vkDestroyPipelineBinaryKHR = {};
	PFN_vkGetPipelineKeyKHR vkGetPipelineKeyKHR = {};
	PFN_vkGetPipelineBinaryDataKHR vkGetPipelineBinaryDataKHR = {};
	PFN_vkCreatePipelineBinariesKHR vkCreatePipelineBinariesKHR = {};
	PFN_vkReleaseCapturedPipelineDataKHR vkReleaseCapturedPipelineDataKHR = {};


	struct PipelineKey : VkPipelineBinaryKeyKHR {

		PipelineKey() = default;
		PipelineKey(const vk::PipelineBinaryKeyKHR &k) {

		}

		bool operator<(const PipelineKey& rhs) const
		{
			if(keySize < rhs.keySize)  return true;
			if(keySize > rhs.keySize)  return false;
			for (int i = 0; i < keySize; ++i) {
				if(key[i] < rhs.key[i])  return true;
				if(key[i] > rhs.key[i])  return false;
			}
			return false;
		}

	};

	std::map<PipelineKey, int> keys;

public:

	void init(const CadR::VulkanDevice &device) {
		_device = &device;
		vkDestroyPipelineBinaryKHR = (PFN_vkDestroyPipelineBinaryKHR)device.getProcAddr("vkDestroyPipelineBinaryKHR");
		vkGetPipelineKeyKHR = (PFN_vkGetPipelineKeyKHR)device.getProcAddr("vkGetPipelineKeyKHR");
		vkCreatePipelineBinariesKHR = (PFN_vkCreatePipelineBinariesKHR)device.getProcAddr("vkCreatePipelineBinariesKHR");
		vkGetPipelineBinaryDataKHR = (PFN_vkGetPipelineBinaryDataKHR)device.getProcAddr("vkGetPipelineBinaryDataKHR");
		vkReleaseCapturedPipelineDataKHR = (PFN_vkReleaseCapturedPipelineDataKHR)device.getProcAddr("vkReleaseCapturedPipelineDataKHR");

		if (!vkDestroyPipelineBinaryKHR || !vkGetPipelineKeyKHR || !vkGetPipelineBinaryDataKHR || !vkCreatePipelineBinariesKHR || !vkReleaseCapturedPipelineDataKHR) throw std::runtime_error("Unsupported Vulkan functionality");
	}


	~PipelineBinaryCache() {

	}

	vk::PipelineBinaryKeyKHR getPipelineKey(vk::GraphicsPipelineCreateInfo &createInfo) {
		vk::PipelineCreateInfoKHR pipelineCreateInfo;
		pipelineCreateInfo.pNext = &createInfo;
		vk::PipelineBinaryKeyKHR key;
		VkResult result = vkGetPipelineKeyKHR(_device->handle(), reinterpret_cast<const VkPipelineCreateInfoKHR*>(&pipelineCreateInfo), reinterpret_cast<VkPipelineBinaryKeyKHR*>(&key));
		if (result != VK_SUCCESS) {
			return {};
		}
		return key;
	}

	template<typename K>
	void printKey(const K &key) {
		for (uint32_t j = 0; j < key.keySize; ++j) {
			std::cout << std::hex << (int)key.key[j] << std::dec;
		}
	}

	void process(vk::GraphicsPipelineCreateInfo &createInfo) {

		auto key = getPipelineKey(createInfo);
		// std::cout << "pipeline key: ";
		// printKey(key);
		// std::cout << "\n";
		// auto &v = keys[PipelineKey(key)];
		// if (v > 0) {
		// 	std::cout << "  " << v << "matches\n";
		// }
		// v++;

		vk::PipelineCreateInfoKHR pipelineCreateInfo;
		pipelineCreateInfo.pNext = &createInfo;

		vk::PipelineBinaryCreateInfoKHR binaryCreateInfo = {};
		binaryCreateInfo.pPipelineCreateInfo = &pipelineCreateInfo;

		vk::PipelineBinaryHandlesInfoKHR info;
		std::vector<vk::PipelineBinaryKHR> binaries;
		VkResult result;
		do {
			result = vkCreatePipelineBinariesKHR(_device->handle(), reinterpret_cast<const VkPipelineBinaryCreateInfoKHR*>(&binaryCreateInfo), nullptr, reinterpret_cast<VkPipelineBinaryHandlesInfoKHR*>(&info));
			if ((result == VK_SUCCESS) && info.pipelineBinaryCount) {
				binaries.resize(info.pipelineBinaryCount);
				info.pPipelineBinaries = binaries.data();
				result = vkCreatePipelineBinariesKHR(_device->handle(), reinterpret_cast<const VkPipelineBinaryCreateInfoKHR*>(&binaryCreateInfo), nullptr, reinterpret_cast<VkPipelineBinaryHandlesInfoKHR*>(&info));
			}
		} while (result == VK_INCOMPLETE);
		if (info.pipelineBinaryCount < binaries.size()) {
			binaries.resize(info.pipelineBinaryCount);
		}
		if(result == VK_PIPELINE_BINARY_MISSING_KHR) {
			// std::cout << "No binary data\n";
			return;
		}
		else if (result != VK_SUCCESS) {
			std::cerr << "vkCreatePipelineBinariesKHR() failed: " << vk::to_string((vk::Result)result) << '\n';
			return;
		}

		std::cout << "got " << binaries.size() << " handles\n";
		// std::vector<vk::PipelineBinaryKeyKHR> keys;
		// keys.resize(binaries.size());
		for (size_t i = 0; i < binaries.size(); ++i) {
			vk::PipelineBinaryDataInfoKHR binaryInfo;
			binaryInfo.pipelineBinary = binaries[i];

			vk::PipelineBinaryKeyKHR key = {};
			size_t binaryDataSize = 0;
			VkResult result = vkGetPipelineBinaryDataKHR(_device->handle(), reinterpret_cast<const VkPipelineBinaryDataInfoKHR*>(&binaryInfo), reinterpret_cast<VkPipelineBinaryKeyKHR*>(&key), &binaryDataSize, nullptr);
			if (key.keySize > 0) {
				std::cout << "[" << i << "] key: ";
				printKey(key);
				std::cout << ", data: " << binaryDataSize << "B\n";
			}
			if(result != VK_SUCCESS) {
				std::cerr << "vkGetPipelineBinaryDataKHR() failed: " << vk::to_string((vk::Result)result) << '\n';
				break;
			}

			std::vector<uint8_t> binaryData;
			binaryData.resize(binaryDataSize);
			result = vkGetPipelineBinaryDataKHR(_device->handle(), reinterpret_cast<const VkPipelineBinaryDataInfoKHR*>(&binaryInfo), reinterpret_cast<VkPipelineBinaryKeyKHR*>(&key), &binaryDataSize, binaryData.data());
		}

	}

	void add(vk::Pipeline pipeline) {
		const auto handle = _device->handle();
		vk::PipelineBinaryCreateInfoKHR createInfo = {};
		createInfo.pipeline = pipeline;
		vk::PipelineBinaryHandlesInfoKHR info;
		std::vector<vk::PipelineBinaryKHR> binaries;

		VkResult result;
		do {
			result = vkCreatePipelineBinariesKHR(handle, reinterpret_cast<const VkPipelineBinaryCreateInfoKHR*>(&createInfo), nullptr, reinterpret_cast<VkPipelineBinaryHandlesInfoKHR*>(&info));
			if ((result == VK_SUCCESS) && info.pipelineBinaryCount) {
				binaries.resize(info.pipelineBinaryCount);
				info.pPipelineBinaries = binaries.data();
				result = vkCreatePipelineBinariesKHR(handle, reinterpret_cast<const VkPipelineBinaryCreateInfoKHR*>(&createInfo), nullptr, reinterpret_cast<VkPipelineBinaryHandlesInfoKHR*>(&info));
			}
		} while (result == VK_INCOMPLETE);
		if (info.pipelineBinaryCount < binaries.size()) {
			binaries.resize(info.pipelineBinaryCount);
		}
		if (result != VK_SUCCESS) {
			std::cerr << "vkCreatePipelineBinariesKHR() failed: " << vk::to_string((vk::Result)result) << '\n';
		}
		else {
			std::cout << "got " << binaries.size() << " handles\n";
			if (binaries.size() == 0) {
				return;
			}
			std::vector<vk::PipelineBinaryKeyKHR> keys;
			keys.resize(binaries.size());
			for (size_t i = 0; i < binaries.size(); ++i) {
				vk::PipelineBinaryDataInfoKHR binaryInfo;
				binaryInfo.pipelineBinary = binaries[i];

				size_t binaryDataSize = 0;
				VkResult result = vkGetPipelineBinaryDataKHR(handle, reinterpret_cast<const VkPipelineBinaryDataInfoKHR*>(&binaryInfo), reinterpret_cast<VkPipelineBinaryKeyKHR*>(&keys[i]), &binaryDataSize, nullptr);
				if(result != VK_SUCCESS) {
					break;
				}

				std::cout << "[" << i << "] key: ";
				for (uint32_t j = 0; j < keys[i].keySize; ++j) {
					std::cout << std::hex << (int)keys[i].key[j] << std::dec;
				}
				std::cout << ", data: " << binaryDataSize << "B\n";

			}
			for (size_t i = 0; i < binaries.size(); ++i) {
				vkDestroyPipelineBinaryKHR(handle, binaries[i], nullptr);
			}
			if(result != VK_SUCCESS) {
#if VK_HEADER_VERSION < 256  // throwResultException moved to detail namespace on 2023-06-28 and the change went public in 1.3.256
				vk::throwResultException(vk::Result(r), "vk::Device::vkGetPipelineBinaryDataKHR");
#else
				vk::detail::throwResultException(vk::Result(result), "vk::Device::vkGetPipelineBinaryDataKHR");
#endif
			}
		}
		vk::ReleaseCapturedPipelineDataInfoKHR releaseInfo;
		releaseInfo.pipeline = pipeline;
		vkReleaseCapturedPipelineDataKHR(handle, reinterpret_cast<const VkReleaseCapturedPipelineDataInfoKHR*>(&releaseInfo), nullptr);
	}

};


class CADPL_EXPORT GraphicsPipelineLibrary {

protected:
	ShaderLibrary* _shaderLibrary;

	struct LibraryObject {
		vk::Pipeline pipeline;
	};


	struct PrerasterState {
		vk::Viewport viewport;
		vk::Rect2D scissor;

		vk::CullModeFlagBits cullMode = vk::CullModeFlagBits::eBack;
		vk::FrontFace frontFace = vk::FrontFace::eCounterClockwise;
		bool depthBiasDynamicState = false;
		bool depthBiasEnable = false;
		float depthBiasConstantFactor;
		float depthBiasClamp;
		float depthBiasSlopeFactor;
		bool lineWidthDynamicState = false;
		float lineWidth = 1.f;

		vk::RenderPass renderPass = nullptr;
		uint32_t subpass = 0;
	};

	struct LibraryFamily {
		std::map<PrerasterState, LibraryObject> objects;
		SharedShaderModule vertex;
	};
	std::map<VertexShaderState, LibraryFamily> _preraster;

public:

	void getOrCreate(const ShaderState& shaderState) {
/*
		auto shaders = _shaderLibrary->getOrCreateShaders(shaderState);

		vk::SpecializationInfo* specializationInfo =
	(shaderState.projectionHandling == ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants)
		? &get<1>(creationDataSet->specializationList.at(pipelineState.projectionIndex))
		: nullptr;

		const auto setStage = [&](vk::PipelineShaderStageCreateInfo &stage, vk::ShaderStageFlagBits stageFlags, SharedShaderModule &module) {
		stage =
			vk::PipelineShaderStageCreateInfo{
				vk::PipelineShaderStageCreateFlags(),  // flags
				stageFlags,  // stage
				nullptr,  // module
				"main",  // pName
				specializationInfo,  // pSpecializationInfo
			};
		if (module) {
			auto* identifier = module.getIdentifier();
			if (module.get()) {
				stage.module = module;
			}
			else if (identifier->identifierSize > 0) {
				std::cerr << "Not implemented yet\n";
				// auto index = numShaderIdentifiers++;
				// shaderIdentifierList[index].identifierSize = identifier->identifierSize;
				// shaderIdentifierList[index].pIdentifier = identifier->identifier;
				// stage.pNext = &shaderIdentifierList[index];
				// if (pipelineFamily._pipelineLibrary->_usePipelineBinary) {
				// 	*flags |= vk::PipelineCreateFlagBits2KHR::eFailOnPipelineCompileRequired;
				// }
				// else {
				// 	createInfo.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
				// }
			}
			else {
				std::cerr << "Have no module\n";
			}
		}
	};

		auto [it1, newRecord1] = _preRasterMap.try_emplace(ShaderLibrary::VertexShaderMapKey(shaderState));
		if(newRecord1) {
			vk::GraphicsPipelineCreateInfo info = {};
			if (shaders.fragment.get()) {

			}
		}

		// auto [it2, newRecord2] = _preRasterMap.try_emplace(ShaderLibrary::VertexShaderMapKey(shaderState));
		// if(newRecord2) {
		//
		//
		// }
*/
	}


};


class CADPL_EXPORT PipelineLibrary {
protected:

	ShaderLibrary* _shaderLibrary;
	std::map<ShaderState, PipelineFamily> _pipelineFamilyMap;
	std::mutex _pipelineFamilyMapMutex;
	CadR::VulkanDevice* _device;

	PipelineBinaryCache _binaryCache; // TODO to pointer
	vk::PipelineCache _pipelineCache;

	std::vector<std::array<float,6>> _specializationData;
	std::vector<vk::Viewport> _viewportList;
	std::vector<vk::Rect2D> _scissorList;

	bool _useFeedbackInfo = false;
	bool _usePipelineBinary = false;

	struct CreationDataSet;  // forward declaration

	struct CreationDataBatch
	{
		static const size_t numPipelines = 16;
		static const size_t numAttachmentsPerPipeline = 3;
		CreationDataSet* creationDataSet;
		std::array<SharedPipeline,numPipelines> sharedPipelineList;  //< Pipelines that are going to be created.
		unsigned numSharedPipelines = 0;
		std::array<vk::PipelineShaderStageModuleIdentifierCreateInfoEXT,numPipelines*3> shaderIdentifierList;
		unsigned numShaderIdentifiers = 0;
		std::array<vk::PipelineShaderStageCreateInfo,numPipelines*3> shaderStageList;
		unsigned numShaderStages = 0;
		std::array<vk::PipelineInputAssemblyStateCreateInfo,numPipelines> inputAssemblyStateList;
		unsigned numInputAssemblyStates = 0;
		std::array<std::tuple<vk::PipelineViewportStateCreateInfo,vk::Viewport,vk::Rect2D>,numPipelines> viewportStateList;
		unsigned numViewportStates = 0;
		std::array<vk::PipelineRasterizationStateCreateInfo,numPipelines> rasterizationStateList;
		unsigned numRasterizationStates = 0;
		std::array<vk::PipelineMultisampleStateCreateInfo,numPipelines> multisampleStateList;
		unsigned numMultisampleStates = 0;
		std::array<vk::PipelineDepthStencilStateCreateInfo,numPipelines> depthStencilStateList;
		unsigned numDepthStencilStates = 0;
		std::array<vk::PipelineColorBlendAttachmentState,numPipelines> colorBlendAttachmentStateList;
		unsigned numColorBlendAttachmentStates = 0;
		std::array<vk::PipelineColorBlendStateCreateInfo,numPipelines> colorBlendStateList;
		unsigned numColorBlendStates = 0;
		std::array<vk::PipelineCreateFlags2CreateInfoKHR,numPipelines>  createFlagsList;
		std::array<vk::GraphicsPipelineCreateInfo,numPipelines> createInfoList;
		std::array<PipelineFamily*,numPipelines> familyList;
		unsigned numCreateInfos = 0;
		CreationDataBatch(CreationDataSet* creationDataSet);
		bool isFull() const;
		void append(SharedPipeline&& sharedPipeline, const PipelineState& pipelineState);
		[[nodiscard]] std::array<vk::Pipeline,PipelineLibrary::CreationDataBatch::numPipelines>
			createPipelines(CadR::VulkanDevice& device, vk::PipelineCache pipelineCache);
	};
public:
	struct AsyncCreationData {
		PipelineLibraryAsyncConsumer *consumer = {};
		void *userData = {};
	};
	struct CompilationResultData {
		vk::Pipeline pipeline;
		SharedPipeline sharedPipeline;
		std::vector<AsyncCreationData> targets;
	};
protected:
	struct CreationDataSet
	{
		const PipelineLibrary* pipelineLibrary;
		std::vector<std::tuple<std::array<float,6>, vk::SpecializationInfo>> specializationList;
		std::vector<vk::Viewport> viewportList;
		std::vector<vk::Rect2D> scissorList;

		std::list<CreationDataBatch> batchList;

		CreationDataSet(const PipelineLibrary& pipelineLibrary);
		void append(SharedPipeline&& sharedPipeline, const PipelineState& pipelineState);
		void createPipelines(const PipelineLibrary& pipelineLibrary);
		size_t count() const;

		void append(std::map<PipelineState, std::pair<SharedPipeline, std::vector<AsyncCreationData>>> &pipelines, PipelineFamily &family, std::vector<CompilationResultData> &compilationResults);
		void createPipelines(PipelineLibrary& pipelineLibrary, std::vector<CompilationResultData> &&compilationResults);
	};


	std::thread _compilationThread;
	std::mutex _compilationThreadMutex;
	bool _compilationThreadExit = false;

	std::condition_variable _compilationThreadCondition;

	struct CompilationFamilyData {
		std::map<PipelineState, std::pair<SharedPipeline, std::vector<AsyncCreationData>>> pipelines;
		std::map<ShaderState, PipelineFamily>::iterator familyIt;
		bool newRecord = false;
	};
	std::map<ShaderState, CompilationFamilyData> _compilationQueue;

	std::list<std::vector<CompilationResultData>> _compilationOutputQueue;
	std::mutex _compilationOutputMutex;

	void compilationWorker();

	friend PipelineFamily;

public:

	// construction and destruction
	PipelineLibrary() noexcept;
	PipelineLibrary(ShaderLibrary& shaderLibrary, vk::PipelineCache pipelineCache = nullptr);
	~PipelineLibrary() noexcept;
	void init(ShaderLibrary& shaderLibrary, vk::PipelineCache pipelineCache = nullptr);

	// projection, viewport and scissor for pipelines
	void setProjectionViewportAndScissor(const glm::mat4x4& projectionMatrix, const vk::Viewport& viewport, const vk::Rect2D& scissor);
	void setProjectionViewportAndScissor(const std::vector<glm::mat4x4>& projectionMatrixList,
		const std::vector<vk::Viewport>& viewportList, const std::vector<vk::Rect2D>& scissorList);

	void recompileAllPipelines();
	void recompilePipelines(const std::vector<std::pair<ShaderState, PipelineState>> &states);

	// synchronous API to get and create pipelines
	SharedPipeline getOrCreatePipeline(const ShaderState& shaderState, const PipelineState& pipelineState);
	SharedPipeline getPipeline(const ShaderState& shaderState, const PipelineState& pipelineState);
	std::vector<SharedPipeline> getOrCreatePipelines(const std::vector<std::pair<ShaderState, PipelineState>> &states);

	// asynchronous API to get and create pipelines
	void asyncEnqueuePipeline(const ShaderState& shaderState, const PipelineState& pipelineState, PipelineLibraryAsyncConsumer *consumer, void *userData, bool delayStart = false);
	void asyncEnqueuePipelines(const std::vector<std::pair<SharedPipeline*, AsyncCreationData>> &pipelines);
	void processRequests();
	void processAsyncQueue(const std::function<void(SharedPipeline, void*)> &callback = {});
	void stopBackgroundThread();

	void processAsyncDebug(size_t maxCount, const std::function<void(SharedPipeline, void*)> &callback = {});

	// getters
	CadR::VulkanDevice& device() const;
	ShaderLibrary& shaderLibrary() const;
	vk::PipelineCache pipelineCache() const;
	vk::PipelineLayout pipelineLayout() const;
	vk::DescriptorSetLayout descriptorSetLayout() const;
	const std::vector<vk::DescriptorSetLayout>& descriptorSetLayoutList() const;

	size_t compileQueueOutputCount() { size_t count = 0; std::unique_lock lk(_compilationOutputMutex); for (const auto &set : _compilationOutputQueue) { count += set.size(); } return count; }

	size_t count() noexcept;
	void setFeedbackInfoEnabled(bool enabled);
	void setPipelineBinaryEnabled(bool enabled, const CadR::VulkanDevice &device);

	enum class CompilationState {
		idle,
		running,
		creating_shader,
		creating_pipeline,
	};
	std::atomic<CompilationState> compilationState = CompilationState::idle;
};


// inline functions
inline SharedPipeline::SharedPipeline(void* pipelineObject) noexcept  : _pipelineObject(pipelineObject) { PipelineFamily::refPipeline(pipelineObject); }
inline SharedPipeline::~SharedPipeline() noexcept  { if(_pipelineObject) PipelineFamily::unrefPipeline(_pipelineObject); }
inline SharedPipeline::SharedPipeline(SharedPipeline&& other) noexcept  : _pipelineObject(other._pipelineObject) { other._pipelineObject=nullptr; }
inline SharedPipeline::SharedPipeline(const SharedPipeline& other) noexcept  : _pipelineObject(other._pipelineObject) { if(_pipelineObject) PipelineFamily::refPipeline(_pipelineObject); }
inline SharedPipeline& SharedPipeline::operator=(SharedPipeline&& rhs) noexcept  { if(_pipelineObject) PipelineFamily::unrefPipeline(_pipelineObject); _pipelineObject=rhs._pipelineObject; rhs._pipelineObject=nullptr; return *this; }
inline SharedPipeline& SharedPipeline::operator=(const SharedPipeline& rhs) noexcept  { if(_pipelineObject) PipelineFamily::unrefPipeline(_pipelineObject); _pipelineObject=rhs._pipelineObject; if(_pipelineObject) PipelineFamily::refPipeline(_pipelineObject); return *this; }
inline void SharedPipeline::reset() noexcept  { if(!_pipelineObject) return; PipelineFamily::unrefPipeline(_pipelineObject); _pipelineObject=nullptr; }
inline const CadR::Pipeline* SharedPipeline::cadrPipeline() const  { return (_pipelineObject) ? &static_cast<PipelineFamily::PipelineObject*>(_pipelineObject)->cadrPipeline : nullptr; }
inline const PipelineFamily* SharedPipeline::pipelineFamily() const  { return (_pipelineObject) ? static_cast<PipelineFamily::PipelineObject*>(_pipelineObject)->pipelineFamily : nullptr; }
inline const PipelineState* SharedPipeline::pipelineState() const  { return (_pipelineObject) ? &static_cast<PipelineFamily::PipelineObject*>(_pipelineObject)->mapIterator->first : nullptr; }
inline void SharedPipeline::replacePipelineHandle(vk::Pipeline newPipeline, CadR::VulkanDevice& device) noexcept  { assert(_pipelineObject && "SharedPipeline was not properly initialized yet."); CadR::Pipeline& cadrPipeline = static_cast<PipelineFamily::PipelineObject*>(_pipelineObject)->cadrPipeline; device.destroy(cadrPipeline.get()); cadrPipeline.set(newPipeline);  }

inline void PipelineFamily::refPipeline(void* pipelineObject) noexcept  { PipelineObject* po=static_cast<PipelineObject*>(pipelineObject); po->referenceCounter++; }
inline void PipelineFamily::unrefPipeline(void* pipelineObject) noexcept  { PipelineObject* po=static_cast<PipelineObject*>(pipelineObject); if(po->referenceCounter==1) PipelineFamily::destroyPipeline(po); else po->referenceCounter--; }
inline PipelineFamily::PipelineFamily(PipelineLibrary& pipelineLibrary) noexcept  : _device(pipelineLibrary._device), _pipelineLibrary(&pipelineLibrary) {}
inline SharedPipeline PipelineFamily::getPipeline(const PipelineState& pipelineState)  { auto it=_pipelineMap.find(pipelineState); return (it!=_pipelineMap.end()) ? SharedPipeline(&it->second) : SharedPipeline(); }
inline const std::map<PipelineState, PipelineFamily::PipelineObject>& PipelineFamily::pipelineMap() const  { return _pipelineMap; }
inline const ShaderState& PipelineFamily::shaderState() const  { return _mapIterator->first; }

inline PipelineLibrary::CreationDataBatch::CreationDataBatch(PipelineLibrary::CreationDataSet* creationDataSet_)  : creationDataSet(creationDataSet_) { }
inline bool PipelineLibrary::CreationDataBatch::isFull() const  { return numSharedPipelines == numPipelines; }
inline void PipelineLibrary::CreationDataSet::append(SharedPipeline&& sharedPipeline, const PipelineState& pipelineState)  { if(batchList.empty() || batchList.back().isFull()) batchList.emplace_back(this); batchList.back().append(std::move(sharedPipeline), pipelineState); }

inline PipelineLibrary::PipelineLibrary() noexcept  : _shaderLibrary(nullptr), _device(nullptr), _compilationThread(&PipelineLibrary::compilationWorker, this) {}
inline PipelineLibrary::PipelineLibrary(ShaderLibrary& shaderLibrary, vk::PipelineCache pipelineCache)  : _shaderLibrary(&shaderLibrary), _device(&shaderLibrary.device()), _pipelineCache(pipelineCache), _compilationThread(&PipelineLibrary::compilationWorker, this) {}
inline void PipelineLibrary::init(ShaderLibrary& shaderLibrary, vk::PipelineCache pipelineCache)  { _device=&shaderLibrary.device(); _shaderLibrary=&shaderLibrary; _pipelineCache=pipelineCache; }
inline void PipelineLibrary::setProjectionViewportAndScissor(const glm::mat4x4& projectionMatrix, const vk::Viewport& viewport, const vk::Rect2D& scissor)  { setProjectionViewportAndScissor(std::vector{projectionMatrix}, std::vector{viewport}, std::vector{scissor}); }
inline SharedPipeline PipelineLibrary::getPipeline(const ShaderState& shaderState, const PipelineState& pipelineState)  { std::unique_lock lk(_pipelineFamilyMapMutex); auto it=_pipelineFamilyMap.find(shaderState); return (it!=_pipelineFamilyMap.end()) ? it->second.getPipeline(pipelineState) : SharedPipeline(); }
inline CadR::VulkanDevice& PipelineLibrary::device() const  { return *_device; }
inline ShaderLibrary& PipelineLibrary::shaderLibrary() const  { return *_shaderLibrary; }
inline vk::PipelineCache PipelineLibrary::pipelineCache() const  { return _pipelineCache; }
inline vk::PipelineLayout PipelineLibrary::pipelineLayout() const  { return _shaderLibrary->pipelineLayout(); }
inline vk::DescriptorSetLayout PipelineLibrary::descriptorSetLayout() const  { return _shaderLibrary->descriptorSetLayout(); }
inline const std::vector<vk::DescriptorSetLayout>& PipelineLibrary::descriptorSetLayoutList() const  { return _shaderLibrary->descriptorSetLayoutList(); }

inline size_t PipelineFamily::count() noexcept { return _pipelineMap.size(); }
inline size_t PipelineLibrary::count() noexcept { size_t s = 0; std::unique_lock lk(_pipelineFamilyMapMutex); for (auto &p : _pipelineFamilyMap) s += p.second.count(); return s; }
inline void PipelineLibrary::setFeedbackInfoEnabled(bool enabled) { _useFeedbackInfo = enabled; }
inline void PipelineLibrary::setPipelineBinaryEnabled(bool enabled, const CadR::VulkanDevice &device) { _usePipelineBinary = enabled; if (enabled) { _binaryCache.init(device); } }

}
