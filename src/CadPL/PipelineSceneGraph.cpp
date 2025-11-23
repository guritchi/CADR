#include <CadPL/PipelineSceneGraph.h>

#include <iostream>

using namespace std;
using namespace CadPL;

CadR::StateSet& PipelineSceneGraph::createStateSet(const ShaderState& shaderState,
	const PipelineState& pipelineState, decltype(_stateSetMap)::insert_commit_data& insertData)
{

	StateSetMapItem* leafItem = new StateSetMapItem(shaderState, pipelineState, _root->renderer());
	_stateSetMap.insert_commit(*leafItem, insertData);
	leafItem->type = StateSetMapItem::Type::leaf;
	struct {
		array<uint16_t,ShaderState::maxNumAttribs> attribAccessInfo;
		uint32_t attribSetup;
		uint32_t materialSetup;
	} pushData = {
		shaderState.attribAccessInfo,
		shaderState.attribSetup,
		shaderState.materialSetup,
	};

	leafItem->stateSet.recordCallList.emplace_back(
		[pushData, leafItem](CadR::StateSet& ss, vk::CommandBuffer commandBuffer, vk::PipelineLayout currentPipelineLayout) {
			if (leafItem->stateSet.pipeline) {
				return;
			}
			ss.renderer().device().cmdPushConstants(
				commandBuffer,  // commandBuffer
				currentPipelineLayout,  // pipelineLayout
				vk::ShaderStageFlagBits::eAllGraphics,  // stageFlags
				16,  // offset
				sizeof(pushData),  // size
				&pushData  // pValues
			);
		}
	);

	if (shaderState.optimizeFlags.to_ulong() != _optimizationLevels[0].to_ulong() && _optimizationLevels.size() > 1) {
		auto flags = _optimizationLevels[1] & shaderState.optimizeFlags;
		if (flags == shaderState.optimizeFlags) {
			_pipelineLibrary->asyncEnqueuePipeline(shaderState, pipelineState, this, leafItem, true);
			leafItem->pending = true;
		}
		else {
			ShaderState nextShaderState{};
			nextShaderState.set(shaderState, flags);

			decltype(_stateSetMap)::insert_commit_data insertData;
			auto [it, canInsert]=_stateSetMap.insert_check(std::tuple{nextShaderState, pipelineState}, insertData);
			if(canInsert) {
				StateSetMapItem* item = new StateSetMapItem(nextShaderState, pipelineState, _root->renderer());
				item->leafNodes.emplace_back(leafItem);
				item->type = StateSetMapItem::Type::optimized;
				_stateSetMap.insert_commit(*item, insertData);
				_pipelineLibrary->asyncEnqueuePipeline(nextShaderState, pipelineState, this, item, true);
				item->pending = true;
			}
			else if (it->sharedPipeline.cadrPipeline()) {
				it->leafNodes.emplace_back(leafItem);
				leafItem->linkTo(it->stateSet);
				// top node should exist, return early
				return leafItem->stateSet;
			}
		}
	}

	{
		ShaderState topShaderState{};
		topShaderState.set(shaderState, _optimizationLevels[0] & shaderState.optimizeFlags);

		decltype(_stateSetMap)::insert_commit_data insertData;
		auto [it, canInsert]=_stateSetMap.insert_check(std::tuple{topShaderState, pipelineState}, insertData);
		if(canInsert) {
			StateSetMapItem* item = new StateSetMapItem(topShaderState, pipelineState, _root->renderer());
			item->leafNodes.emplace_back(leafItem);
			item->type = StateSetMapItem::Type::leastOptimized;
			// create pipeline (blocking)
			item->sharedPipeline = _pipelineLibrary->getOrCreatePipeline(topShaderState, pipelineState);
			item->stateSet.pipeline = item->sharedPipeline.cadrPipeline();
			item->linkTo(*_root);
			_stateSetMap.insert_commit(*item, insertData);
			leafItem->linkTo(item->stateSet);
		}
		else if (it->sharedPipeline.cadrPipeline()) {
			it->leafNodes.emplace_back(leafItem);
			leafItem->linkTo(it->stateSet);
		}
	}
	return leafItem->stateSet;
}

void PipelineSceneGraph::pipelineCreated(SharedPipeline sharedPipeline, void *userData)
{
	// std::cout << "T" << std::this_thread::get_id() << " PipelineSceneGraph::pipelineCreated: " << userData << "\n";
	StateSetMapItem *item = reinterpret_cast<StateSetMapItem*>(userData);
	if (item->sharedPipeline.cadrPipeline() != sharedPipeline.cadrPipeline()) {
		item->sharedPipeline = sharedPipeline;
	}
	item->stateSet.pipeline = item->sharedPipeline.cadrPipeline();
	item->pending = false;

	if (item->type == StateSetMapItem::Type::leaf) {
		item->linkTo(*_root);
	}
	else if (item->type == StateSetMapItem::Type::optimized) {
		// std::cout << "Node " << item << ", " << item->shaderState.serialize() << " bind: \n";
		for (auto &node : item->leafNodes) {
			// std::cout << "    " << node << ", " << node->shaderState.serialize() << "\n";
			node->linkTo(*item);
		}
		item->linkTo(*_root);
	}
}

bool PipelineSceneGraph::isWaitingOnPipelines() const
{
	for (const auto &item : _stateSetMap) {
		if (item.pending) {
			return true;
		}
	}
	return false;
}