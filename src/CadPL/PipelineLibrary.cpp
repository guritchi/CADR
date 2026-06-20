#include <CadPL/PipelineLibrary.h>
#include <CadR/VulkanDevice.h>
#include <CadPL/ShaderGenerator.h>

#include <CadPL/DebugUtils.h>

#include <iostream>
#include <chrono>

using namespace std;
using namespace CadPL;


// pipeline vertex input
static constexpr const vk::PipelineVertexInputStateCreateInfo pipelineVertexInputStateCreateInfo(
	vk::PipelineVertexInputStateCreateFlags(),  // flags
	0,  // vertexBindingDescriptionCount
	nullptr,  // pVertexBindingDescriptions
	0,  // vertexAttributeDescriptionCount
	nullptr  // pVertexAttributeDescriptions
);

// specialization data map
static constexpr const std::array specializationMap {
	vk::SpecializationMapEntry{0,0,4},  // constantID, offset, size
	vk::SpecializationMapEntry{1,4,4},
	vk::SpecializationMapEntry{2,8,4},
	vk::SpecializationMapEntry{3,12,4},
	vk::SpecializationMapEntry{4,16,4},
	vk::SpecializationMapEntry{5,20,4},
};




PipelineLibrary::~PipelineLibrary()
{
	stopBackgroundThread();
	{ // clear hanging pipelines
		std::unique_lock lk(_compilationOutputMutex);
		for (auto &batch : _compilationOutputQueue) {
			for (auto &data : batch) {
				if (data.pipeline) {
					_device->destroy(data.pipeline);
				}
			}
		}
		_compilationOutputQueue.clear();
	}
	assert(_pipelineFamilyMap.empty() && "PipelineLibrary::~PipelineLibrary(): All pipelines "
		"owned by PipelineLibrary must be released before destroying PipelineLibrary.");
}


PipelineFamily::~PipelineFamily()
{
	assert(_pipelineMap.empty() && "PipelineFamily::~PipelineFamily(): All SharedPipelines must be released before destroying PipelineFamily or PipelineLibrary.");
}

void PipelineFamily::initialize(const ShaderState& shaderState, std::map<ShaderState, PipelineFamily>::iterator it, bool createShaders)
{
	assert(_pipelineLibrary && "PipelineFamily is missing prior initialization");

	it->second._mapIterator = it;
	it->second._primitiveTopology = shaderState.primitiveTopology;

	auto* shaderLibrary = _pipelineLibrary->_shaderLibrary;
	auto set = createShaders? shaderLibrary->getOrCreateShaders(shaderState) : shaderLibrary->getShadersWithoutCompilation(shaderState);
	it->second._vertexShader   = set.vertex;
	it->second._geometryShader = set.geometry;
	it->second._fragmentShader = set.fragment;
}

void PipelineFamily::destroyPipeline(void* pipelineObject) noexcept
{
	PipelineObject* po = reinterpret_cast<PipelineObject*>(pipelineObject); 
	PipelineFamily* pf = po->pipelineFamily;
	po->cadrPipeline.destroyPipeline(*pf->_device);
	pf->_pipelineMap.erase(po->mapIterator);

	if(pf->_pipelineMap.empty())
		pf->_pipelineLibrary->_pipelineFamilyMap.erase(pf->_mapIterator);
}

void PipelineFamily::initializeRecord(std::map<PipelineState, PipelineObject>::iterator &it)
{
	// initialize new record
	// (do not throw until SharedPipeline is created)
	it->second.cadrPipeline.init(
		nullptr,
		_pipelineLibrary->pipelineLayout(),
		&_pipelineLibrary->descriptorSetLayoutList()
	);
	it->second.referenceCounter = 0;
	it->second.pipelineFamily = this;
	it->second.mapIterator = it;
	it->second._compilePending = false;
}

SharedPipeline PipelineFamily::getOrCreatePipeline(const PipelineState& pipelineState, bool &canCreate)
{
	canCreate = false;
	auto [it, newRecord] = _pipelineMap.try_emplace(pipelineState);
	if(!newRecord)
		return SharedPipeline(&it->second);;
	initializeRecord(it);
	SharedPipeline sharedPipeline(&it->second);

	// if viewport, scissor or projection matrix were not set yet
	// and they are required by the pipeline, skip pipeline creation;
	// (the pipeline will be created in setProjectionViewportAndScissor() or similar function)
	if(pipelineState.viewportAndScissorHandling == PipelineState::ViewportAndScissorHandling::SetFunction &&
		(_pipelineLibrary->_viewportList.empty() || _pipelineLibrary->_scissorList.empty()))
		return sharedPipeline;
	if(_mapIterator->first.projectionHandling == ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants &&
		_pipelineLibrary->_specializationData.empty())
		return sharedPipeline;

	canCreate = true;
	return sharedPipeline;
}

SharedPipeline PipelineFamily::getOrCreatePipeline(const PipelineState& pipelineState)
{
	const auto start = std::chrono::system_clock::now();
	bool canCreate;
	SharedPipeline sharedPipeline = getOrCreatePipeline(pipelineState, canCreate);
	if (canCreate) {
		PipelineLibrary::CreationDataSet creationDataSet(*_pipelineLibrary);
		creationDataSet.append(SharedPipeline(sharedPipeline), pipelineState);
		creationDataSet.createPipelines(*_pipelineLibrary);
		const auto end = std::chrono::system_clock::now();
		CadPL::Debug::log("", "mainCompileTime", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
		CadPL::Debug::increment("mainCreateCount", 1);
	}
	return sharedPipeline;
}

SharedPipeline PipelineLibrary::getOrCreatePipeline(const ShaderState& shaderState, const PipelineState& pipelineState) {
	std::unique_lock lk(_pipelineFamilyMapMutex);
	const auto start = std::chrono::system_clock::now();
	auto [it, newRecord] = _pipelineFamilyMap.try_emplace(shaderState, *this);
	if(newRecord) {
		try {
			it->second.initialize(shaderState, it);
			const auto end = std::chrono::system_clock::now();
			CadPL::Debug::log("", "mainCompileTime", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
		} catch(...) {
			_pipelineFamilyMap.erase(it);
			throw;
		}
	}
	return it->second.getOrCreatePipeline(pipelineState);
}

std::vector<SharedPipeline> PipelineLibrary::getOrCreatePipelines(const std::vector<std::pair<ShaderState, PipelineState>> &states)
{
	const auto start = std::chrono::system_clock::now();
	std::vector<SharedPipeline> pipelines;
	CreationDataSet creationDataSet(*this);
	{
		std::unique_lock lk(_pipelineFamilyMapMutex);
		for (const auto &state : states) {

			const PipelineState& pipelineState = state.second;
			if(pipelineState.viewportAndScissorHandling == PipelineState::ViewportAndScissorHandling::SetFunction)
			{
				// update viewport and scissor
				const_cast<vk::Viewport&>(pipelineState.viewport) =
					_viewportList.at(pipelineState.viewportIndex);
				const_cast<vk::Rect2D&>(pipelineState.scissor) =
					_scissorList.at(pipelineState.scissorIndex);
			}
			else if(state.first.projectionHandling != ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants)
				continue;

			auto [it, newRecord] = _pipelineFamilyMap.try_emplace(state.first, *this);
			if(newRecord) {
				it->second.initialize(state.first, it);
			}
			bool canCreate;
			SharedPipeline sharedPipeline = it->second.getOrCreatePipeline(state.second, canCreate);
			if (canCreate) {
				creationDataSet.append(SharedPipeline(sharedPipeline), pipelineState);
			}
		}
	}
	const auto count = creationDataSet.count();
	creationDataSet.createPipelines(*this);
	const auto end = std::chrono::system_clock::now();
	CadPL::Debug::log("", "mainCompileTime", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
	CadPL::Debug::increment("mainCreateCount", count);
	return pipelines;
}

void PipelineLibrary::setProjectionViewportAndScissor(const std::vector<glm::mat4x4>& projectionMatrixList,
	const std::vector<vk::Viewport>& viewportList, const std::vector<vk::Rect2D>& scissorList)
{
	// update specialization constants based on projection matrices
	_specializationData.resize(projectionMatrixList.size());
	for(size_t i=0,c=projectionMatrixList.size(); i<c; i++)
	{
		const auto& projectionMatrix = projectionMatrixList[i];
		_specializationData[i] =
			array<float,6>{
				projectionMatrix[2][0], projectionMatrix[2][1], projectionMatrix[2][3],
				projectionMatrix[3][0], projectionMatrix[3][1], projectionMatrix[3][3],
			};
	}

	// update viewports and scissors
	_viewportList = viewportList;
	_scissorList = scissorList;

	// recompileAllPipelines();
}

void PipelineLibrary::recompileAllPipelines()
{
	// CreationDataSet - used for storing all pipeline creation data
	// so we create pipelines in batches and not one by one.
	// This might allow Vulkan driver for speeding up creation.
	CreationDataSet creationDataSet(*this);

	// create list of pipelines to be recompiled
	for(auto familyIt=_pipelineFamilyMap.begin(); familyIt!=_pipelineFamilyMap.end(); familyIt++) {

		const ShaderState& shaderState = familyIt->first;
		PipelineFamily& f = familyIt->second;
		for(auto pipelineIt=f._pipelineMap.begin(); pipelineIt!=f._pipelineMap.end(); pipelineIt++) {
			const PipelineState& pipelineState = pipelineIt->first;
			if(pipelineState.viewportAndScissorHandling == PipelineState::ViewportAndScissorHandling::SetFunction)
			{
				// update viewport and scissor
				const_cast<vk::Viewport&>(pipelineState.viewport) =
					_viewportList.at(pipelineState.viewportIndex);
				const_cast<vk::Rect2D&>(pipelineState.scissor) =
					_scissorList.at(pipelineState.scissorIndex);

				// append pipeline into the set for recompilation
				creationDataSet.append(SharedPipeline(&pipelineIt->second), pipelineState);
			}
			else if(shaderState.projectionHandling == ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants)
				// append pipeline into the set for recompilation
				creationDataSet.append(SharedPipeline(&pipelineIt->second), pipelineState);
		}
	}

	// create pipelines
	const auto count = creationDataSet.count();
	const auto start = std::chrono::system_clock::now();
	creationDataSet.createPipelines(*this);
	const auto end = std::chrono::system_clock::now();
	CadPL::Debug::log("", "mainCompileTime", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
	CadPL::Debug::increment("mainCreateCount", count);
}

void PipelineLibrary::recompilePipelines(const std::vector<std::pair<ShaderState, PipelineState>> &states)
{
	CreationDataSet creationDataSet(*this);
	std::cout << "recompilePipelines(): " << states.size() << std::endl;
	for (const auto& state : states) {
		auto familyIt = _pipelineFamilyMap.find(state.first);
		if (familyIt != _pipelineFamilyMap.end()) {
			auto pipelineIt = familyIt->second._pipelineMap.find(state.second);
			if (pipelineIt != familyIt->second._pipelineMap.end()) {
				const PipelineState& pipelineState = pipelineIt->first;
				if(pipelineState.viewportAndScissorHandling == PipelineState::ViewportAndScissorHandling::SetFunction)
				{
					// update viewport and scissor
					const_cast<vk::Viewport&>(pipelineState.viewport) =
						_viewportList.at(pipelineState.viewportIndex);
					const_cast<vk::Rect2D&>(pipelineState.scissor) =
						_scissorList.at(pipelineState.scissorIndex);

					std::cout << "new viewport: " << pipelineState.viewport.x << "," << pipelineState.viewport.y << ", " << pipelineState.viewport.width << "," << pipelineState.viewport.height << std::endl;

					// append pipeline into the set for recompilation
					creationDataSet.append(SharedPipeline(&pipelineIt->second), pipelineState);
				}
				else if(state.first.projectionHandling == ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants)
					// append pipeline into the set for recompilation
					creationDataSet.append(SharedPipeline(&pipelineIt->second), pipelineState);
			}
		}
	}

	// create pipelines
	const auto count = creationDataSet.count();
	const auto start = std::chrono::system_clock::now();
	creationDataSet.createPipelines(*this);
	const auto end = std::chrono::system_clock::now();
	CadPL::Debug::log("", "mainCompileTime", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
	CadPL::Debug::increment("mainCreateCount", count);
}

PipelineLibrary::CreationDataSet::CreationDataSet(const PipelineLibrary& pipelineLibrary)
	: pipelineLibrary(&pipelineLibrary)
    , specializationList(pipelineLibrary._specializationData.size())
	, viewportList(pipelineLibrary._viewportList)
	, scissorList(pipelineLibrary._scissorList)
{
	for(size_t i=0,c=pipelineLibrary._specializationData.size(); i<c; i++) {
		auto& s = specializationList[i];
		get<0>(s) = pipelineLibrary._specializationData[i];
		get<1>(s) =
			vk::SpecializationInfo(
				uint32_t(specializationMap.size()),  // mapEntryCount
				specializationMap.data(),  // pMapEntries
				6 * sizeof(float),  // dataSize
				get<0>(s).data()  // pData
			);

	}
}


void PipelineLibrary::CreationDataBatch::append(SharedPipeline&& sharedPipeline, const PipelineState& pipelineState) {
	assert(numSharedPipelines < sharedPipelineList.size() && "CreationDataBatch::append(): CreationDataBatch is full. Cannot append more pipelines.");
	assert(sharedPipeline.cadrPipeline() != nullptr && "SharedPipeline object must not be empty.");

	// get info from sharedPipeline before we move it
	PipelineFamily& pipelineFamily = const_cast<PipelineFamily&>(*sharedPipeline.pipelineFamily());
	vk::PipelineLayout pipelineLayout = pipelineFamily._pipelineLibrary->pipelineLayout();
	const ShaderState& shaderState = pipelineFamily.shaderState();

	if(pipelineState.viewportAndScissorHandling == PipelineState::ViewportAndScissorHandling::SetFunction) {
		// update viewport and scissor
		const_cast<vk::Viewport&>(pipelineState.viewport) =
			creationDataSet->viewportList.at(pipelineState.viewportIndex);
		const_cast<vk::Rect2D&>(pipelineState.scissor) =
			creationDataSet->scissorList.at(pipelineState.scissorIndex);
	}

	// move sharedPipeline
	sharedPipelineList[numSharedPipelines] = sharedPipeline;
	numSharedPipelines++;

	// flags
	auto& createInfo = createInfoList[numCreateInfos];
	createInfo.flags = vk::PipelineCreateFlags();
	numCreateInfos++;

	// specializationInfo
	vk::SpecializationInfo* specializationInfo =
		(shaderState.projectionHandling == ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants)
			? &get<1>(creationDataSet->specializationList.at(pipelineState.projectionIndex))
			: nullptr;

	// stageCount and pStages
	vk::PipelineShaderStageCreateInfo* shaderStages = &shaderStageList[numShaderStages];
	numShaderStages += 3;
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
			module.waitIfCompiling();
			if (module.get()) {
				stage.module = module;
			}
			else if (identifier->identifierSize > 0) {
				auto index = numShaderIdentifiers++;
				shaderIdentifierList[index].identifierSize = identifier->identifierSize;
				shaderIdentifierList[index].pIdentifier = identifier->identifier;
				stage.pNext = &shaderIdentifierList[index];
				createInfo.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
			}
			else {
				std::cerr << "Have no module\n";
			}
		}
	};

	setStage(shaderStages[0], vk::ShaderStageFlagBits::eVertex, pipelineFamily._vertexShader);
	setStage(shaderStages[1], vk::ShaderStageFlagBits::eFragment, pipelineFamily._fragmentShader);
	if (pipelineFamily._geometryShader) {
		setStage(shaderStages[2], vk::ShaderStageFlagBits::eGeometry, pipelineFamily._geometryShader);
		createInfo.stageCount = 3;
	}
	else
		createInfo.stageCount = 2;

	createInfo.pStages = shaderStages;

	// pVertexInputState
	createInfo.pVertexInputState = &pipelineVertexInputStateCreateInfo;

	// pInputAssemblyState
	for(size_t i=0; i<numInputAssemblyStates; i++)
		if(inputAssemblyStateList[i].topology == pipelineFamily._primitiveTopology) {
			createInfo.pInputAssemblyState = &inputAssemblyStateList[i];
			goto foundInputAssemblyState;
		}
	inputAssemblyStateList[numInputAssemblyStates] =
		vk::PipelineInputAssemblyStateCreateInfo(
			vk::PipelineInputAssemblyStateCreateFlags(),  // flags
			pipelineFamily._primitiveTopology,  // topology
			VK_FALSE  // primitiveRestartEnable
		);
	createInfo.pInputAssemblyState = &inputAssemblyStateList[numInputAssemblyStates];
	numInputAssemblyStates++;
	foundInputAssemblyState:;

	// pTessellationState
	createInfo.pTessellationState = nullptr;

	// pViewportState
	for(size_t i=0; i<numViewportStates; i++)
		if(get<1>(viewportStateList[i]) == pipelineState.viewport &&
		   get<2>(viewportStateList[i]) == pipelineState.scissor)
		{
			createInfo.pViewportState = &get<0>(viewportStateList[i]);
			goto foundViewportState;
		}
	get<0>(viewportStateList[numViewportStates]) =
		vk::PipelineViewportStateCreateInfo(
			vk::PipelineViewportStateCreateFlags(),  // flags
			1,  // viewportCount
			&get<1>(viewportStateList[numViewportStates]),  // pViewports
			1,  // scissorCount
			&get<2>(viewportStateList[numViewportStates])  // pScissors
		);
	get<1>(viewportStateList[numViewportStates]) = pipelineState.viewport;
	get<2>(viewportStateList[numViewportStates]) = pipelineState.scissor;
	createInfo.pViewportState = &get<0>(viewportStateList[numViewportStates]);
	numViewportStates++;
	foundViewportState:;

	// pRasterizationState
	for(size_t i=0; i<numRasterizationStates; i++) {
		const auto& rasterizationState = rasterizationStateList[i];
		if(rasterizationState.cullMode == pipelineState.cullMode &&
		   rasterizationState.frontFace == pipelineState.frontFace &&
		   (rasterizationState.depthBiasEnable!=0) == pipelineState.depthBiasEnable &&
		   rasterizationState.depthBiasConstantFactor == pipelineState.depthBiasConstantFactor &&
		   rasterizationState.depthBiasClamp == pipelineState.depthBiasClamp &&
		   rasterizationState.depthBiasSlopeFactor == pipelineState.depthBiasSlopeFactor &&
		   rasterizationState.lineWidth == pipelineState.lineWidth)
		{
			createInfo.pRasterizationState = &rasterizationStateList[i];
			goto foundRasterizationState;
		}
	}
	rasterizationStateList[numRasterizationStates] =
		vk::PipelineRasterizationStateCreateInfo{
			vk::PipelineRasterizationStateCreateFlags(),  // flags
			VK_FALSE,  // depthClampEnable
			VK_FALSE,  // rasterizerDiscardEnable
			vk::PolygonMode::eFill,  // polygonMode
			pipelineState.cullMode,  // cullMode
			pipelineState.frontFace,  // frontFace
			pipelineState.depthBiasEnable,  // depthBiasEnable
			pipelineState.depthBiasConstantFactor,  // depthBiasConstantFactor
			pipelineState.depthBiasClamp,  // depthBiasClamp
			pipelineState.depthBiasSlopeFactor,  // depthBiasSlopeFactor
			pipelineState.lineWidth  // lineWidth
		};
	createInfo.pRasterizationState = &rasterizationStateList[numRasterizationStates];
	numRasterizationStates++;
	foundRasterizationState:;

	// pMultisampleState
	for(size_t i=0; i<numMultisampleStates; i++) {
		const auto& multisampleState = multisampleStateList[i];
		if(multisampleState.rasterizationSamples == pipelineState.rasterizationSamples &&
		   (multisampleState.sampleShadingEnable!=0) == pipelineState.sampleShadingEnable &&
		   multisampleState.minSampleShading == pipelineState.minSampleShading)
		{
			createInfo.pMultisampleState = &multisampleStateList[i];
			goto foundMultisampleState;
		}
	}
	multisampleStateList[numMultisampleStates] =
		vk::PipelineMultisampleStateCreateInfo{
			vk::PipelineMultisampleStateCreateFlags(),  // flags
			pipelineState.rasterizationSamples,  // rasterizationSamples
			pipelineState.sampleShadingEnable,  // sampleShadingEnable
			pipelineState.minSampleShading,  // minSampleShading
			nullptr,   // pSampleMask
			VK_FALSE,  // alphaToCoverageEnable
			VK_FALSE   // alphaToOneEnable
		};
	createInfo.pMultisampleState = &multisampleStateList[numMultisampleStates];
	numMultisampleStates++;
	foundMultisampleState:;

	// pDepthStencilState
	for(size_t i=0; i<numDepthStencilStates; i++) {
		const auto& depthStencilState = depthStencilStateList[i];
		if((depthStencilState.depthTestEnable!=0) == pipelineState.depthTestEnable &&
		   (depthStencilState.depthWriteEnable!=0) == pipelineState.depthWriteEnable)
		{
			createInfo.pDepthStencilState = &depthStencilStateList[i];
			goto foundDepthStencilState;
		}
	}
	depthStencilStateList[numDepthStencilStates] =
		vk::PipelineDepthStencilStateCreateInfo{
			vk::PipelineDepthStencilStateCreateFlags(),  // flags
			pipelineState.depthTestEnable,  // depthTestEnable
			pipelineState.depthWriteEnable,  // depthWriteEnable
			vk::CompareOp::eLess,  // depthCompareOp
			VK_FALSE,  // depthBoundsTestEnable
			VK_FALSE,  // stencilTestEnable
			vk::StencilOpState(),  // front
			vk::StencilOpState(),  // back
			0.f,  // minDepthBounds
			0.f   // maxDepthBounds
		};
	createInfo.pDepthStencilState = &depthStencilStateList[numDepthStencilStates];
	numDepthStencilStates++;
	foundDepthStencilState:;

	// colorBlendAttachmentState
	vk::PipelineColorBlendAttachmentState* colorBlendAttachmentsPtr = nullptr;
	auto isBlendAttachmentStateEqual =
		[](const vk::PipelineColorBlendAttachmentState& s1,
		   const PipelineState::BlendAttachmentState& s2) -> bool
		{
			if((s1.blendEnable!=0) != s2.blendEnable || s1.colorWriteMask != s2.colorWriteMask)
				return false;
			if(s2.blendEnable == false)  // blendEnable is the same on s1 and s2; if both are false, no need to compare blend settings
				return true;
			return s1.srcColorBlendFactor != s2.srcColorBlendFactor ||
			       s1.dstColorBlendFactor != s2.dstColorBlendFactor ||
			       s1.colorBlendOp        != s2.colorBlendOp ||
			       s1.srcAlphaBlendFactor != s2.srcAlphaBlendFactor ||
			       s1.dstAlphaBlendFactor != s2.dstAlphaBlendFactor ||
			       s1.alphaBlendOp        != s2.alphaBlendOp;
		};
	for(size_t i=0,c=numColorBlendAttachmentStates; i<c; i+=numAttachmentsPerPipeline) {
		for(size_t j=0,d=pipelineState.blendState.size(); j<d; j++)
			if(!isBlendAttachmentStateEqual(colorBlendAttachmentStateList[i+j], pipelineState.blendState[j]))
				goto colorBlendAttachmentsDiffer;
		colorBlendAttachmentsPtr = &colorBlendAttachmentStateList[i];
		goto colorBlendAttachmentsFound;
		colorBlendAttachmentsDiffer:;
	}
	colorBlendAttachmentsPtr = &colorBlendAttachmentStateList[numColorBlendAttachmentStates];
	for(size_t i=0,c=pipelineState.blendState.size(); i<c; i++) {
		const auto& src = pipelineState.blendState[i];
		colorBlendAttachmentStateList[numColorBlendAttachmentStates+i] =
			vk::PipelineColorBlendAttachmentState(
				src.blendEnable,
				src.srcColorBlendFactor,
				src.dstColorBlendFactor,
				src.colorBlendOp,
				src.srcAlphaBlendFactor,
				src.dstAlphaBlendFactor,
				src.alphaBlendOp,
				src.colorWriteMask
			);
	}
	numColorBlendAttachmentStates += numAttachmentsPerPipeline;
	colorBlendAttachmentsFound:;

	// pColorBlendState
	for(size_t i=0; i<numColorBlendStates; i++) {
		const auto& colorBlendState = colorBlendStateList[i];
		if(colorBlendState.attachmentCount != pipelineState.blendState.size())
			continue;
		if(colorBlendState.pAttachments != colorBlendAttachmentsPtr)
			continue;
		createInfo.pColorBlendState = &colorBlendState;
		goto foundColorBlendState;
	}
	colorBlendStateList[numColorBlendStates] =
		vk::PipelineColorBlendStateCreateInfo(
			vk::PipelineColorBlendStateCreateFlags(),  // flags
			VK_FALSE,  // logicOpEnable
			vk::LogicOp::eClear,  // logicOp
			uint32_t(pipelineState.blendState.size()),  // attachmentCount
			colorBlendAttachmentsPtr,  // pAttachments
			array<float,4>{0.f,0.f,0.f,0.f}  // blendConstants
		);
	createInfo.pColorBlendState = &colorBlendStateList[numColorBlendStates];
	numColorBlendStates++;
	foundColorBlendState:;

	// pDynamicState
	createInfo.pDynamicState = nullptr;

	// remaining createInfo members
	createInfo.layout = pipelineLayout;
	createInfo.renderPass = pipelineState.renderPass;
	createInfo.subpass = pipelineState.subpass;
	createInfo.basePipelineHandle = nullptr;
	createInfo.basePipelineIndex = -1;
}


[[nodiscard]] array<vk::Pipeline,PipelineLibrary::CreationDataBatch::numPipelines>
	PipelineLibrary::CreationDataBatch::createPipelines(
		CadR::VulkanDevice& device, vk::PipelineCache pipelineCache)
{
	struct Feedback {
		vk::PipelineCreationFeedbackCreateInfo info;
		vk::PipelineCreationFeedback feedback;
		std::array<vk::PipelineCreationFeedback, 3> stageFeedbacks;
		struct ExtraFeedback {
			bool hasIdentifier = false;
		};
		std::array<ExtraFeedback, 3> stageExtraFeedbacks;
		bool compileRequired = false;

		void bind(uint32_t stageCount) {
			assert(stageCount <= stageFeedbacks.size() && "too many stages for feedback");
			info.pPipelineCreationFeedback = &feedback;
			info.pipelineStageCreationFeedbackCount = stageCount;
			info.pPipelineStageCreationFeedbacks = stageFeedbacks.data();
		}
	};
	std::vector<Feedback> feedbacks;
	if (creationDataSet->pipelineLibrary->_useFeedbackInfo) {
		feedbacks.resize(numCreateInfos);
		for (size_t i = 0; i < numCreateInfos; ++i) {
			auto *dst = reinterpret_cast<vk::BaseOutStructure*>(&createInfoList[i]);
			while (dst->pNext) {
				dst = dst->pNext;
			}
			feedbacks[i].bind(createInfoList[i].stageCount);
			dst->pNext = reinterpret_cast<vk::BaseOutStructure*>(&feedbacks[i].info);
			for (uint32_t j = 0; j < createInfoList[i].stageCount; ++j) {
				if (createInfoList[i].pStages[j].pNext) {
					feedbacks[i].stageExtraFeedbacks[j].hasIdentifier = true;
				}
			}
		}
	}

	// create pipelines
	array<vk::Pipeline,numPipelines> pipelines;
	const auto start = std::chrono::system_clock::now();
	VkResult r =
		device.vkCreateGraphicsPipelines(
			device.handle(),
			pipelineCache,
			numCreateInfos,
			reinterpret_cast<VkGraphicsPipelineCreateInfo*>(createInfoList.data()),
			nullptr,
			reinterpret_cast<VkPipeline*>(pipelines.data())
		);
	const auto end = std::chrono::system_clock::now();
	CadPL::Debug::log("", "vkCreateGraphicsPipelines()", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
	if(r == VK_PIPELINE_COMPILE_REQUIRED) {
		array<VkGraphicsPipelineCreateInfo,numPipelines> info2;
		array<vk::Pipeline,numPipelines> pipelines2;
		array<size_t,numPipelines> pipelineTargets;
		uint32_t count = 0;
		// reduce info array
		{
			auto* dst = info2.data();
			size_t i = 0;
			while (i<numCreateInfos) {
				if (pipelines[i]) {
					++i;
				}
				else {
					auto* src = &createInfoList[i];
					int span = 1;
					pipelineTargets[count] = i;
					++i;
					while (i<numCreateInfos && !pipelines[i]) {
						pipelineTargets[count + span] = i;
						++span;
						++i;
					}
					std::memcpy(dst, src, span*sizeof(vk::GraphicsPipelineCreateInfo));
					dst += span;
					count += span;
				}
			}
		}

		const auto setStage = [&](vk::PipelineShaderStageCreateInfo &stage, vk::ShaderModule module) {
			stage.module = module;
			stage.pNext = nullptr;
		};
		if (creationDataSet->pipelineLibrary->_useFeedbackInfo) {
			for (uint32_t i=0; i<count; ++i) {
				auto target = pipelineTargets[i];
				feedbacks[target].compileRequired = true;
			}
		}
		for (uint32_t i=0; i<count; ++i) {
			auto target = pipelineTargets[i];
			PipelineFamily& pipelineFamily = const_cast<PipelineFamily&>(*sharedPipelineList[target].pipelineFamily());
			auto &info = reinterpret_cast<vk::GraphicsPipelineCreateInfo&>(info2[i]);
			info.flags = {};
			assert(info.stageCount >= 2 && "missing pStages");
			const ShaderState& shaderState = pipelineFamily.shaderState();

			auto &vertex = pipelineFamily._vertexShader;
			auto &geometry = pipelineFamily._geometryShader;
			auto &fragment = pipelineFamily._fragmentShader;
			if (!vertex.get()) {
				vertex.getIdentifier()->identifierSize = 0;
			}
			if (!fragment.get()) {
				fragment.getIdentifier()->identifierSize = 0;
			}
			if (geometry && !geometry.get()) {
				geometry.getIdentifier()->identifierSize = 0;
			}
			pipelineFamily._pipelineLibrary->shaderLibrary().createShaders(
				shaderState,
				vertex,
				geometry,
				fragment
			);

			setStage(const_cast<vk::PipelineShaderStageCreateInfo &>(info.pStages[0]), vertex.get());
			setStage(const_cast<vk::PipelineShaderStageCreateInfo &>(info.pStages[1]), fragment.get());
			if (info.stageCount > 2) {
				setStage(const_cast<vk::PipelineShaderStageCreateInfo &>(info.pStages[2]), geometry.get());
			}
		}
		const auto start = std::chrono::system_clock::now();
		r = device.vkCreateGraphicsPipelines(
		device.handle(),
			pipelineCache,
			count,
			info2.data(),
			nullptr,
			reinterpret_cast<VkPipeline*>(pipelines2.data())
		);
		const auto end = std::chrono::system_clock::now();
		CadPL::Debug::log("", "vkCreateGraphicsPipelines()", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
		for (uint32_t i=0; i<count; ++i) {
			pipelines[pipelineTargets[i]] = pipelines2[i];
		}
		CadPL::Debug::increment("VK_PIPELINE_COMPILE_REQUIRED", count);
	}

	if(r != VK_SUCCESS) {
		for(vk::Pipeline p : pipelines)
			device.destroy(p);
	#if VK_HEADER_VERSION < 256  // throwResultException moved to detail namespace on 2023-06-28 and the change went public in 1.3.256
		vk::throwResultException(vk::Result(r), "vk::Device::createGraphicsPipelines");
	#else
		vk::detail::throwResultException(vk::Result(r), "vk::Device::createGraphicsPipelines");
	#endif
	}

	if (creationDataSet->pipelineLibrary->_useFeedbackInfo) {

		size_t hitCount = 0;
		size_t missCount = 0;
		const auto processFeedback = [&](const Feedback &feedback, int index){
			const auto &flags = feedback.feedback.flags;
			if (flags & vk::PipelineCreationFeedbackFlagBits::eValid) {
				std::stringstream debug;
				debug << "feedback[" << index << "]: { ";
				if (flags & vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit) {
					debug << "CacheHit ";
					++hitCount;
				}
				else {
					++missCount;
				}
				if (flags & vk::PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration) {
					debug << "BaseAcceleration ";
				}
				debug << "} ";
				if (feedback.compileRequired) {
					debug << "CompileRequired ";
				}
				// std::cout << debug.str() << "duration: " << feedback.feedback.duration << " ns\n";
				// ShaderGenerator::logDebugEvent(std::move(debug.str()), feedback.feedback.duration);
				for (size_t j=0; j<feedback.stageFeedbacks.size(); ++j) {
					const auto &stage = feedback.stageFeedbacks[j];
					const auto &flags = stage.flags;
					if (flags & vk::PipelineCreationFeedbackFlagBits::eValid) {
						std::stringstream debug;
						debug << "  [" << j << "]: { ";
						if (flags & vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit) {
							debug << "CacheHit ";
						}
						if (flags & vk::PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration) {
							debug << "BaseAcceleration ";
						}
						if (flags & ~(vk::PipelineCreationFeedbackFlagBits::eBasePipelineAcceleration | vk::PipelineCreationFeedbackFlagBits::eApplicationPipelineCacheHit | vk::PipelineCreationFeedbackFlagBits::eValid)) {
							debug << static_cast<uint32_t>(flags);
						}
						debug << "} ";
						if (feedback.stageExtraFeedbacks[j].hasIdentifier) {
							debug << "ModuleIdentifier ";
						}
						debug << "duration: " << stage.duration << " ns\n";
					}
				}
			}
			else {
				std::cout << "feedback[" << index << "]: invalid\n";
			}
		};

		for (size_t i=0; i<numCreateInfos; ++i) {
			processFeedback(feedbacks[i], i);
		}
		CadPL::Debug::increment("cacheHitCount", hitCount);
		CadPL::Debug::increment("cacheMissCount", missCount);
	}
	CadPL::Debug::increment("totalCreateCount", numCreateInfos);

	return pipelines;
}


void PipelineLibrary::CreationDataSet::createPipelines(const PipelineLibrary& pipelineLibrary)
{
	array<vk::Pipeline,CreationDataBatch::numPipelines> pipelines;
	while(!batchList.empty())
	{
		CreationDataBatch& batch = batchList.front();

		// create pipelines
		// note: do not throw in the following code until pipelines are safely replaced through
		// SharedPipeline objects; otherwise pipeline handles will be leaked
		pipelines = batch.createPipelines(*pipelineLibrary._device, pipelineLibrary._pipelineCache);

		// update pipelines inside SharedPipeline object
		for(size_t i=0, c=batch.numSharedPipelines; i<c; i++)
			batch.sharedPipelineList[i].replacePipelineHandle(pipelines[i], *pipelineLibrary._device);

		// release CreationDataBatch
		batchList.pop_front();
	}
}


bool PipelineState::operator<(const PipelineState& rhs) const
{
	if(cullMode < rhs.cullMode)  return true;
	if(cullMode > rhs.cullMode)  return false;
	if(frontFace < rhs.frontFace)  return true;
	if(frontFace > rhs.frontFace)  return false;
	if(depthBiasEnable < rhs.depthBiasEnable)  return true;
	if(depthBiasEnable > rhs.depthBiasEnable)  return false;
	if(depthBiasEnable) {
		if(depthBiasDynamicState < rhs.depthBiasDynamicState)  return true;
		if(depthBiasDynamicState > rhs.depthBiasDynamicState)  return false;
		if(!depthBiasDynamicState) {
			if(depthBiasConstantFactor < rhs.depthBiasConstantFactor)  return true;
			if(depthBiasConstantFactor > rhs.depthBiasConstantFactor)  return false;
			if(depthBiasClamp < rhs.depthBiasClamp)  return true;
			if(depthBiasClamp > rhs.depthBiasClamp)  return false;
			if(depthBiasSlopeFactor < rhs.depthBiasSlopeFactor)  return true;
			if(depthBiasSlopeFactor > rhs.depthBiasSlopeFactor)  return false;
		}
	}
	if(lineWidthDynamicState < rhs.lineWidthDynamicState)  return true;
	if(lineWidthDynamicState > rhs.lineWidthDynamicState)  return false;
	if(!lineWidthDynamicState) {
		if(lineWidth < rhs.lineWidth)  return true;
		if(lineWidth > rhs.lineWidth)  return false;
	}
	if(rasterizationSamples < rhs.rasterizationSamples)  return true;
	if(rasterizationSamples > rhs.rasterizationSamples)  return false;
	if(sampleShadingEnable < rhs.sampleShadingEnable)  return true;
	if(sampleShadingEnable > rhs.sampleShadingEnable)  return false;
	if(minSampleShading < rhs.minSampleShading)  return true;
	if(minSampleShading > rhs.minSampleShading)  return false;
	if(depthTestEnable < rhs.depthTestEnable)  return true;
	if(depthTestEnable > rhs.depthTestEnable)  return false;
	if(depthWriteEnable < rhs.depthWriteEnable)  return true;
	if(depthWriteEnable > rhs.depthWriteEnable)  return false;

	if(blendState < rhs.blendState)  return true;
	if(blendState > rhs.blendState)  return false;
	if(renderPass < rhs.renderPass)  return true;
	if(renderPass > rhs.renderPass)  return false;
	if(subpass < rhs.subpass)  return true;
	if(subpass > rhs.subpass)  return false;

	if(projectionIndex < rhs.projectionIndex)  return true;
	if(projectionIndex > rhs.projectionIndex)  return false;
	if(viewportAndScissorHandling < rhs.viewportAndScissorHandling)  return true;
	if(viewportAndScissorHandling > rhs.viewportAndScissorHandling)  return false;
	if(viewportAndScissorHandling == ViewportAndScissorHandling::SetFunction) {
		if(viewportIndex < rhs.viewportIndex)  return true;
		if(viewportIndex > rhs.viewportIndex)  return false;
		return scissorIndex < rhs.scissorIndex;
	}
	else if(viewportAndScissorHandling == ViewportAndScissorHandling::Value) {
		if(viewport.x < rhs.viewport.x)  return true;
		if(viewport.x > rhs.viewport.x)  return false;
		if(viewport.y < rhs.viewport.y)  return true;
		if(viewport.y > rhs.viewport.y)  return false;
		if(viewport.width < rhs.viewport.width)  return true;
		if(viewport.width > rhs.viewport.width)  return false;
		if(viewport.height < rhs.viewport.height)  return true;
		if(viewport.height > rhs.viewport.height)  return false;
		if(viewport.minDepth < rhs.viewport.minDepth)  return true;
		if(viewport.minDepth > rhs.viewport.minDepth)  return false;
		if(viewport.maxDepth < rhs.viewport.maxDepth)  return true;
		if(viewport.maxDepth > rhs.viewport.maxDepth)  return false;
		if(scissor.offset.x < rhs.scissor.offset.x)  return true;
		if(scissor.offset.x > rhs.scissor.offset.x)  return false;
		if(scissor.offset.y < rhs.scissor.offset.y)  return true;
		if(scissor.offset.y > rhs.scissor.offset.y)  return false;
		if(scissor.extent.width < rhs.scissor.extent.width)  return true;
		if(scissor.extent.width > rhs.scissor.extent.width)  return false;
		return scissor.extent.height < rhs.scissor.extent.height;
	}
	else
		return false;
}


bool PipelineState::BlendAttachmentState::operator<(const BlendAttachmentState& rhs) const
{
	if(blendEnable < rhs.blendEnable)  return true;
	if(blendEnable > rhs.blendEnable)  return false;
	
	if(blendEnable) {
		if(srcColorBlendFactor < rhs.srcColorBlendFactor)  return true;
		if(srcColorBlendFactor > rhs.srcColorBlendFactor)  return false;
		if(dstColorBlendFactor < rhs.dstColorBlendFactor)  return true;
		if(dstColorBlendFactor > rhs.dstColorBlendFactor)  return false;
		if(colorBlendOp < rhs.colorBlendOp)  return true;
		if(colorBlendOp > rhs.colorBlendOp)  return false;
		if(srcAlphaBlendFactor < rhs.srcAlphaBlendFactor)  return true;
		if(srcAlphaBlendFactor > rhs.srcAlphaBlendFactor)  return false;
		if(dstAlphaBlendFactor < rhs.dstAlphaBlendFactor)  return true;
		if(dstAlphaBlendFactor > rhs.dstAlphaBlendFactor)  return false;
		if(alphaBlendOp < rhs.alphaBlendOp)  return true;
		if(alphaBlendOp > rhs.alphaBlendOp)  return false;
	}

	return colorWriteMask < rhs.colorWriteMask;
}


size_t PipelineLibrary::CreationDataSet::count() const
{
	size_t count = 0;
	for (const auto &batch : batchList) count += batch.numCreateInfos;
	return count;
}

void PipelineLibrary::CreationDataSet::append(std::map<PipelineState, std::pair<SharedPipeline, std::vector<AsyncCreationData>>> &pipelines, PipelineFamily &family, std::vector<CompilationResultData> &compilationResults)
{
	for (auto &data : pipelines) {
		void *object; // PipelineFamily::PipelineObject
		if (!data.second.first._pipelineObject) {
			auto [pipelineIt, newRecord] = family._pipelineMap.try_emplace(data.first);
			if(newRecord) {
				family.initializeRecord(pipelineIt);
			}
			object = &pipelineIt->second;
		}
		else {
			object = data.second.first._pipelineObject;
		}
		append(SharedPipeline(object), data.first);

		compilationResults.emplace_back(CompilationResultData{nullptr, SharedPipeline(object), std::move(data.second.second)});
	}
}

void PipelineLibrary::CreationDataSet::createPipelines(PipelineLibrary& pipelineLibrary, std::vector<CompilationResultData> &&compilationResults)
{
	size_t compilationResultsIndex = 0;

	// const auto inBatch = compilationResults.size();

	array<vk::Pipeline,CreationDataBatch::numPipelines> pipelines;
	while(!batchList.empty())
	{
		CreationDataBatch& batch = batchList.front();
		if (batch.numSharedPipelines == 0) {
			std::cout << "COMPILE NUM: " << batch.numSharedPipelines << "\n";
		}

		// create pipelines
		// note: do not throw in the following code until pipelines are safely replaced through
		// SharedPipeline objects; otherwise pipeline handles will be leaked
		pipelines = batch.createPipelines(*pipelineLibrary._device, pipelineLibrary._pipelineCache);

		for(size_t i=0, c=batch.numSharedPipelines; i<c; i++)
			compilationResults[compilationResultsIndex + i].pipeline = pipelines[i];
		compilationResultsIndex += batch.numSharedPipelines;
		CadPL::Debug::increment("threadCreateCount", 1);

		// release CreationDataBatch
		batchList.pop_front();
	}

	std::unique_lock lk(pipelineLibrary._compilationOutputMutex);
	pipelineLibrary._compilationOutputQueue.emplace_back(std::move(compilationResults));
}

void PipelineLibrary::stopBackgroundThread()
{
	{
		std::lock_guard lk(_compilationThreadMutex);
		if (_compilationThreadExit) {
			return;
		}
		_compilationThreadExit = true;
		_compilationThreadCondition.notify_one();
	}
	if (_compilationThread.joinable()) {
		_compilationThread.join();
	}
}

void PipelineLibrary::asyncEnqueuePipeline(const ShaderState& shaderState, const PipelineState& pipelineState, PipelineLibraryAsyncConsumer *consumer, void *userData, bool delayStart)
{
	std::lock_guard lk(_compilationThreadMutex);
	_compilationQueue[shaderState].pipelines[pipelineState].second.emplace_back(AsyncCreationData{consumer, userData});
	if (!delayStart) {
		_compilationThreadCondition.notify_one();
	}
}

void PipelineLibrary::asyncEnqueuePipelines(const std::vector<std::pair<SharedPipeline*, AsyncCreationData>> &pipelines)
{
	std::lock_guard lk(_compilationThreadMutex);
	for (auto &data : pipelines) {
		if (data.first->_pipelineObject) {
			auto* object = reinterpret_cast<PipelineFamily::PipelineObject*>(data.first->_pipelineObject);
			auto &entry = _compilationQueue[object->pipelineFamily->shaderState()].pipelines[*data.first->pipelineState()];
			entry.first = *data.first;
			entry.second.emplace_back(data.second);
		}
	}
}

void PipelineLibrary::processRequests()
{
	// start background compilation if pending
	{
		std::lock_guard lk(_compilationThreadMutex);
		if (!_compilationQueue.empty()) {
			_compilationThreadCondition.notify_one();
		}
	}
}

void PipelineLibrary::processAsyncQueue(const std::function<void(SharedPipeline, void*)> &callback)
{
	std::list<std::vector<CompilationResultData>> output;
	{
		std::unique_lock lk(_compilationOutputMutex);
		if (!_compilationOutputQueue.empty()) {
			output.swap(_compilationOutputQueue);
		}
	}
	for (auto &batch : output) {
		for (auto &result : batch) {
			result.sharedPipeline.replacePipelineHandle(result.pipeline, *_device);
			for (const auto &target : result.targets) {
				if (target.consumer) {
					target.consumer->pipelineCreated(result.sharedPipeline, target.userData);
				}
				else if (callback) {
					callback(result.sharedPipeline, target.userData);
				}
			}
		}
	}
}

void PipelineLibrary::processAsyncDebug(size_t maxCount, const std::function<void(SharedPipeline, void*)> &callback) {
	std::vector<CompilationResultData> output;
	{
		std::unique_lock lk(_compilationOutputMutex);
		if (!_compilationOutputQueue.empty() && output.size() < maxCount) {
			auto &batch = _compilationOutputQueue.front();
			while (!batch.empty() && output.size() < maxCount) {
				output.emplace_back(batch.front());
				batch.erase(batch.begin());
			}
			if (batch.empty()) {
				_compilationOutputQueue.pop_front();
			}
		}
	}
	for (auto &result : output) {
		result.sharedPipeline.replacePipelineHandle(result.pipeline, *_device);
		for (const auto &target : result.targets) {
			if (target.consumer) {
				target.consumer->pipelineCreated(result.sharedPipeline, target.userData);
			}
			else if (callback) {
				callback(result.sharedPipeline, target.userData);
			}
		}

	}
}

void PipelineLibrary::compilationWorker() {
	while (true) {
		try {
			decltype(_compilationQueue) compilations;
			// wait for signal and take all compilation requests
			{
				std::unique_lock lk(_compilationThreadMutex);
				_compilationThreadCondition.wait(lk, [&]{ return !_compilationQueue.empty() || _compilationThreadExit; });
				if (_compilationThreadExit) {
					break;
				}
				compilationState = CompilationState::running;
				Debug::enterThread("compilationWorker");
				compilations.swap(_compilationQueue);
			}
#ifndef NDEBUG
			// debug validation
			bool ok = true;
			for (auto &data : compilations) {
				for (auto &state : data.second.pipelines) {
					if(state.first.viewportAndScissorHandling == PipelineState::ViewportAndScissorHandling::SetFunction) {
						if (_viewportList.size() <= state.first.viewportIndex
							|| _scissorList.size() <= state.first.scissorIndex)
						{
							std::cerr << "COMPILE THREAD: Missing ViewportAndScissorHandling data.\n";
							ok = false;
							break;
						}
					}
				}
			}
			if (!ok) {
				continue;
			}
#endif
			const auto start = std::chrono::system_clock::now();

			uint32_t count = 0;
			for (auto &family : compilations) {
				count += family.second.pipelines.size();
			}
			// std::cout << "COMPILE THREAD: Got " << count << " compilations in " << compilations.size() << "groups \n";
			std::vector<std::future<void>> futures;
			{
				std::unique_lock lk(_pipelineFamilyMapMutex);
				for (auto &family : compilations) {
					std::tie(family.second.familyIt, family.second.newRecord) = _pipelineFamilyMap.try_emplace(family.first, *this);
					auto &f = family.second.familyIt->second;
					if (family.second.newRecord) {
						f.initialize(family.first, family.second.familyIt, false);
						if (f._vertexShader.aquireCompileFlag()) {
							futures.emplace_back(_shaderLibrary->createVertexShaderAsync(family.first, f._vertexShader));
						}
						if (f._fragmentShader.aquireCompileFlag()) {
							futures.emplace_back(_shaderLibrary->createFragmentShaderAsync(family.first, f._fragmentShader));
						}
						if (f._geometryShader && f._geometryShader.aquireCompileFlag()) {
							futures.emplace_back(_shaderLibrary->createGeometryShaderAsync(family.first, f._geometryShader));
						}
					}
				}
			}

			std::vector<CompilationResultData> compilationResults;
			CreationDataSet set(*this);

			if (!futures.empty()) {
				compilationState = CompilationState::creating_shader;
				for (auto &f : futures) {
					if (f.valid()) {
						f.get();
					}
				}
			}

			compilationState = CompilationState::creating_pipeline;
			for (auto &familyGroup : compilations) {
				auto &family = familyGroup.second.familyIt->second;
				set.append(familyGroup.second.pipelines, family, compilationResults);
			}
			set.createPipelines(*this, std::move(compilationResults));

			const auto end = std::chrono::system_clock::now();
			// std::cout << "TH" << std::this_thread::get_id() << "(compile): compilations done in ";
			// CadPL::Debug::printDuration(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), std::cout);
			// std::cout << "\n";
			CadPL::Debug::log("", "threadCompileTime", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
		} catch(exception &e) {
			cout << "Failed because of exception: " << e.what() << endl;
		} catch(...) {
			cout << "Failed because of unspecified exception." << endl;
		}
		compilationState = CompilationState::idle;
		Debug::exitThread("compilationWorker");
	}
}