// SPDX-FileCopyrightText: 2019-2025 PCJohn (Jan Pečiva, peciva@fit.vut.cz)
//
// SPDX-License-Identifier: MIT-0

#include <CadR/BoundingBox.h>
#include <CadR/BoundingSphere.h>
#include <CadR/Geometry.h>
#include <CadR/Exceptions.h>
#include <CadR/ImageAllocation.h>
#include <CadR/Pipeline.h>
#include <CadR/Sampler.h>
#include <CadR/StagingBuffer.h>
#include <CadR/StagingData.h>
#include <CadR/StateSet.h>
#include <CadR/Texture.h>
#include <CadR/VulkanDevice.h>
#include <CadR/VulkanInstance.h>
#include <CadR/VulkanLibrary.h>
#include <CadPL/PipelineSceneGraph.h>
#include "VulkanWindow.h"
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>
#include "../../3rdParty/stb/stb_image.h"
#include "imgui.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_sdl3.h"
#if defined(USE_PLATFORM_WIN32)
#include "backends/imgui_impl_win32.h"
#elif defined(USE_PLATFORM_SDL3)
#include "backends/imgui_impl_sdl3.h"
#elif defined(USE_PLATFORM_SDL2)
#include "backends/imgui_impl_sdl2.h"
#elif defined(USE_PLATFORM_GLFW)
#include "backends/imgui_impl_glfw.h"
#endif
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN  // reduce amount of included files by windows.h
# include <windows.h>  // needed for SetConsoleOutputCP()
# include <shellapi.h>  // needed for CommandLineToArgvW()
#endif

#include <CadPL/ShaderLibrary.h>
#include <CadPL/PipelineLibrary.h>
#include <CadPL/ShaderGenerator.h>

using namespace std;

using json = nlohmann::json;

typedef logic_error GltfError;

// static constexpr uint32_t NumDescriptorsStreaming  = 2048;
// static constexpr uint32_t NumDescriptorsNonUniform = 1024;

static const auto OptimizeLevels = std::array{
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeNone).to_ullong()),
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeAttribs).to_ullong()),
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeTextures).to_ullong()),
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeMaterial).to_ullong()),
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeTextures | CadPL::ShaderState::OptimizeMaterial).to_ullong()),
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeAttribs | CadPL::ShaderState::OptimizeMaterial).to_ullong()),
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeAttribs | CadPL::ShaderState::OptimizeTextures).to_ullong()),
	static_cast<uint32_t>((CadPL::ShaderState::OptimizeAttribs | CadPL::ShaderState::OptimizeTextures | CadPL::ShaderState::OptimizeMaterial).to_ullong()),
};

// Shader data structures
struct OpenGLLightGpuData {
	glm::vec3 ambient;
	float constantAttenuation;
	glm::vec3 diffuse;
	float linearAttenuation;
	glm::vec3 specular;
	float quadraticAttenuation;
};
static_assert(sizeof(OpenGLLightGpuData) == 48, "Wrong OpenGLLightGpuData data size");
struct GltfLightGpuData {
	glm::vec3 color;
	float intensity;  // in candelas (lm/sr) for point light and spotlight and in luxes (lm/m2) for directional light
	float range;
	array<uint32_t,3> padding;
};
static_assert(sizeof(GltfLightGpuData) == 32, "Wrong GltfLightGpuData data size");
struct SpotlightGpuData {
	glm::vec3 eyeDirection;  // spotlight direction in eye coordinates, it must be normalized
	float cosOuterConeAngle;  // cosinus of outer spotlight cone; outside the cone, there is zero light intensity
	float cosInnerConeAngle;  // cosinus of inner spotlight cone; if -1. is provided, OpenGL-style spotlight is used, ignoring inner cone and using spotExponent instead; if value is > -1., DirectX style spotlight is used, e.g. everything inside the inner cone receives full light intensity and light intensity between inner and outer cone is linearly interpolated starting from zero intensity on outer code to full intensity in inner cone
	float spotExponent;  // if cosInnerConeAngle is -1, OpenGL style spotlight is used, using spotExponent
	array<uint32_t,2> padding;
};
static_assert(sizeof(SpotlightGpuData) == 32, "Wrong SpotlightGpuData data size");
struct LightGpuData {
	glm::vec3 eyePositionOrDirection;  // for point light and spotlight: position in eye coordinates,
	                                   // for directional light: direction in eye coordinates, direction must be normalized
	uint32_t settings;  // switches between point light, directional light and spotlight
	OpenGLLightGpuData opengl;
	GltfLightGpuData gltf;
	SpotlightGpuData spotlight;
};
static constexpr const size_t lightGpuDataSize = 128;
static_assert(sizeof(LightGpuData) == lightGpuDataSize, "Wrong LightGpuData data size");

constexpr const uint32_t maxLights = 4;
struct SceneGpuData {
	glm::mat4 viewMatrix;    // current camera view matrix
	glm::mat4 projectionMatrix;  // current camera view matrix
	float p11,p22,p33,p43;   // projectionMatrix - members that depend on zNear and zFar clipping planes
	glm::vec3 ambientLight;  // we use vec4 instead of vec3 for the purpose of memory alignment; alpha component for light intensities is unused
	uint32_t numLights;
	array<uint32_t,8> padding;
	LightGpuData lights[maxLights];
};
static_assert(sizeof(SceneGpuData) == 192+(lightGpuDataSize*maxLights), "Wrong SceneGpuData data size");

class MedianCounter {
    static constexpr unsigned BUFFER_SIZE = 1024;

	std::array<double, BUFFER_SIZE> _counters = {};
	unsigned _current = 0;

public:

    bool update(double value) {
        _counters[_current] = value;
        _current = (_current+1) % BUFFER_SIZE;
    	return _current == 0;
    }

    double get() const {
    	auto values = _counters;
    	std::nth_element(values.begin(), values.begin() + BUFFER_SIZE / 2, values.end());
    	if constexpr (BUFFER_SIZE % 2 == 0) {
    		return std::midpoint(values[BUFFER_SIZE / 2], values[BUFFER_SIZE / 2 + 1]);
    	}
    	else {
    		return values[BUFFER_SIZE / 2];
    	}
    }

};


static void printStateSet(const CadR::StateSet &set, int indent = 0) {

    for (int i = 0; i < indent; ++i) {
        std::cout << "  ";
    }

    const auto &descriptors = set.descriptorSetList();
    std::cout << &set << ": " << set.getNumDrawables() << " drawables, pipeline: ";
    if (!set.pipeline) {
        std::cout << "null";
    }
    else {
        std::cout << set.pipeline->get() << ", layout: " << set.pipeline->layout();
    }
    if (!descriptors.empty()) {
        std::cout << ", " << descriptors.size() << " descriptors";
    }
	if (!set.childList.empty()) {
		std::cout << ", " << set.childList.size() << " children";
	}
    std::cout << "\n";
    indent++;
    for (const auto &c : set.childList) {
        printStateSet(c, indent);
    }
}

// application class
class App {
public:

	App(int argc, char* argv[]);
	~App();

	void init();
	void resize(VulkanWindow& window,
		const vk::SurfaceCapabilitiesKHR& surfaceCapabilities, vk::Extent2D newSurfaceExtent);
	void frame(VulkanWindow& window);
	void mouseMove(VulkanWindow& window, const VulkanWindow::MouseState& mouseState);
	void mouseButton(VulkanWindow&, size_t button, VulkanWindow::ButtonState buttonState, const VulkanWindow::MouseState& mouseState);
	void mouseWheel(VulkanWindow& window, float wheelX, float wheelY, const VulkanWindow::MouseState& mouseState);
	void key(VulkanWindow& window, VulkanWindow::KeyState keyState, VulkanWindow::ScanCode scanCode);

	// Vulkan core objects
	// (The order of members is not arbitrary but defines construction and destruction order.)
	// (If App object is on the stack, do not call std::exit() as the App destructor might not be called.)
	// (It is probably good idea to destroy vulkanLib and vulkanInstance after the display connection.)
	CadR::VulkanLibrary vulkanLib;
	CadR::VulkanInstance vulkanInstance;

	// window needs to be destroyed after the swapchain
	// (this is required especially on Wayland)
	VulkanWindow window;

	// Vulkan variables, handles and objects
	// (they need to be destructed in non-arbitrary order in the destructor)
	vk::PhysicalDevice physicalDevice;
	uint32_t graphicsQueueFamily;
	uint32_t presentationQueueFamily;
	CadR::VulkanDevice device;
	vk::Queue graphicsQueue;
	vk::Queue presentationQueue;
	vk::SurfaceFormatKHR surfaceFormat;
	vk::Format depthFormat;
	float maxSamplerAnisotropy;
	vk::RenderPass renderPass;
	vk::SwapchainKHR swapchain;
	vk::Image depthImage;
	vk::DeviceMemory depthImageMemory;
	vk::ImageView depthImageView;
	vector<vk::ImageView> swapchainImageViews;
	vector<vk::Framebuffer> framebuffers;
	vector<vk::Semaphore> renderingFinishedSemaphores;
	vk::Semaphore imageAvailableSemaphore;
	vk::Fence renderingFinishedFence;

	CadR::Renderer renderer;
	std::unique_ptr<CadPL::ShaderLibrary> shaderLibrary;
	std::unique_ptr<CadPL::PipelineLibrary> pipelineLibrary;
	vk::CommandPool commandPool;
	vk::CommandBuffer commandBuffer;

	vk::PipelineCache imguiPipelineCache;
	vk::DescriptorPool imguiDescriptorPool;
	bool imguiPanel = true;
	std::string guiFrameTime;


	// camera control
	static constexpr const float maxZNearZFarRatio = 1000.f;  //< Maximum z-near distance and z-far distance ratio. High ratio might lead to low z-buffer precision.
	static constexpr const float zoomStepRatio = 0.9f;  //< Camera zoom speed. Camera distance from the model center is multiplied by this value on each mouse wheel up event and divided by it on each wheel down event.
	float fovy = 80.f / 180.f * glm::pi<float>();  //< Initial field-of-view in y-axis (in vertical direction) is 80 degrees.
	float cameraHeading = 0.f;
	float cameraElevation = 0.f;
	float cameraDistance;
	float startMouseX, startMouseY;
	float startCameraHeading, startCameraElevation;
	CadR::BoundingSphere sceneBoundingSphere;

	bool newCamera = true;
	glm::vec3 cameraEye;
	glm::vec3 cameraDirection;
	glm::vec3 cameraUp;
	glm::vec3 cameraRight;
	glm::vec2 cameraAngles;
	float prevMouseX, prevMouseY;
	float cameraSpeed = 10.f;
	float cameraSensitivity = 0.01f;

	filesystem::path filePath;
	string utf8FilePath;  // File path stored as utf-8. MSVC has problems to convert some characters from utf-16 to utf-8. So we keep the extra string. See comment for utf16toUtf8() for more info.
	string utf8FileName;  // File name as utf-8. No parent directories and no file name suffix. MSVC has problems to convert some characters from utf-16 to utf-8. So we keep the extra string. See comment for utf16toUtf8() for more info.

	CadR::HandlelessAllocation sceneDataAllocation;
	CadR::StateSet stateSetRoot;
	CadR::StateSet stateSetDrawables;
	CadR::StateSet stateSetGUI;
	struct TextureStateSet {
		CadR::StateSet stateSet;
		CadR::Texture* texture;
	};
	struct PipelineStateSet {
		CadPL::SharedPipeline pipeline;
		CadR::StateSet stateSet;
		bool compile = false;

		PipelineStateSet(CadR::Renderer& r) noexcept  : stateSet(r) {}
        void destroy() noexcept  { stateSet.destroy(); pipeline.reset(); }
	};

	struct TextureSetup {
		uint32_t settings = {};
		size_t index = ~size_t(0);

		operator bool() const {
			return index != ~size_t(0);
		}
	};

    struct PipelineDescriptor {
        uint32_t optimizeFlags = {};
        uint32_t materialSettings = {};
        uint32_t attibSetup = {};
        std::array<uint16_t, 16> attribInfo = {};
        std::array<uint32_t, 6> textureInfo = {};
        uint8_t numAttributes = {};
        uint8_t numTextures = {};
        // TODO lights
        vk::CullModeFlagBits cullMode = {};
        vk::FrontFace frontFace = {};
        vk::PrimitiveTopology primitiveTopology = {};

        auto operator<=>(const PipelineDescriptor&) const = default;
    };
	struct UberPipelineDescriptor {
		vk::CullModeFlagBits cullMode = {};
		vk::FrontFace frontFace = {};
		vk::PrimitiveTopology primitiveTopology = {};

		UberPipelineDescriptor();
		explicit UberPipelineDescriptor(const PipelineDescriptor &desc)
			: cullMode(desc.cullMode)
			, frontFace(desc.frontFace)
			, primitiveTopology(desc.primitiveTopology)
		{}

		auto operator<=>(const UberPipelineDescriptor&) const = default;
	};

    std::map<PipelineDescriptor, PipelineStateSet> pipelineStateSets;
	std::map<UberPipelineDescriptor, PipelineStateSet> uberPipelineStateSets;
	CadPL::PipelineSceneGraph pipelineSceneGraph;
	CadR::Pipeline layoutOnlyPipeline;
	vector<CadR::Geometry> geometryList;
	vector<CadR::Drawable> drawableList;
	vector<CadR::MatrixList> matrixLists;
	vector<CadR::DataAllocation> materialList;
	vector<CadR::ImageAllocation> imageList;
	vector<CadR::Sampler> samplerList;
	vector<CadR::Texture> textureList;
	CadR::Sampler defaultSampler;
	CadR::DataAllocation defaultMaterial;

	uint32_t optimizeFlags = *OptimizeLevels.rbegin();

    CadPL::PipelineState getPipelineState(
            vk::RenderPass renderPass,
            vk::Extent2D surfaceExtent,
            vk::FrontFace frontFace,
            vk::CullModeFlagBits cullMode)
    const {

        CadPL::PipelineState pipelineState{
        	.viewportAndScissorHandling = CadPL::PipelineState::ViewportAndScissorHandling::SetFunction,
        	.projectionIndex = 0,
			.viewportIndex = 0,
			.scissorIndex = 0,
            .viewport = vk::Viewport(0.f, 0.f, float(surfaceExtent.width), float(surfaceExtent.height), 0.f, 1.f),
            .scissor = vk::Rect2D(vk::Offset2D(0, 0), surfaceExtent),
            .cullMode = cullMode,
            .frontFace = frontFace,
        	.depthBiasDynamicState = false,
			.depthBiasEnable = false,
			.depthBiasConstantFactor = 0.f,
			.depthBiasClamp = 0.f,
			.depthBiasSlopeFactor = 0.f,
			.lineWidthDynamicState = false,
			.lineWidth = 1.f,
			.rasterizationSamples = vk::SampleCountFlagBits::e1,
			.sampleShadingEnable = false,
			.minSampleShading = 0.f,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.blendState = { { .blendEnable = false } },
			.renderPass = renderPass,
			.subpass = 0,
        };

        // pipelineState.blendState.emplace_back(
        //     VK_FALSE, // blendEnable
        //     vk::BlendFactor::eZero,  // srcColorBlendFactor
        //     vk::BlendFactor::eZero,  // dstColorBlendFactor
        //     vk::BlendOp::eAdd,	     // colorBlendOp
        //     vk::BlendFactor::eZero,  // srcAlphaBlendFactor
        //     vk::BlendFactor::eZero,  // dstAlphaBlendFactor
        //     vk::BlendOp::eAdd,	     // alphaBlendOp
        //     vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        //     vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA  // colorWriteMask
        // );
        return pipelineState;
    }

	PipelineStateSet& getPipelineStateSet(
		uint32_t optimizeFlags,
		vk::PrimitiveTopology primitiveTopology,
		vk::CullModeFlagBits cullMode,
		vk::FrontFace frontFace,
		uint32_t materialSettings,
		uint32_t attibSetup,
		const decltype(PipelineDescriptor::attribInfo) &attribInfo,
		const std::span<TextureSetup> textureSetup
	) {

		PipelineDescriptor desc{
			.optimizeFlags = optimizeFlags,
			.cullMode = cullMode,
			.frontFace = frontFace,
			.primitiveTopology = primitiveTopology
		};

		if (optimizeFlags & CadPL::ShaderState::OptimizeMaterial.to_ulong()) {
			desc.materialSettings = materialSettings;
		}

		if (optimizeFlags & CadPL::ShaderState::OptimizeAttribs.to_ulong()) {
			uint8_t numAttributes = 0;
			uint8_t numTextureAttributes = 0;
			for (int i = 0; i < attribInfo.size(); ++i) {
				if (attribInfo[i]) {
					numAttributes++;
					if (i >= 4) {
						numTextureAttributes++;
					}
				}
			}
			desc.attribInfo = attribInfo;
			desc.numAttributes = numAttributes;
			desc.attibSetup = attibSetup;

			if (numTextureAttributes == 0) {
				desc.numTextures = 0;
				desc.optimizeFlags |= CadPL::ShaderState::OptimizeTextures.to_ulong();
			}
		}

		if (optimizeFlags & CadPL::ShaderState::OptimizeTextures.to_ulong()) {
			uint8_t numTextures = 0;
			for (int i = 0; i < textureSetup.size(); ++i) {
				if (textureSetup[i]) {
					// std::cout << "  Add texture: " << std::hex << textureSetup[i].settings << std::dec << "\n";
					desc.textureInfo[numTextures] = textureSetup[i].settings;
					numTextures++;
				}
			}
			desc.numTextures = numTextures;
		}

    	const bool none = optimizeFlags == static_cast<uint32_t>(CadPL::ShaderState::OptimizeNone.to_ullong());
    	const bool full = optimizeFlags == static_cast<uint32_t>((
			CadPL::ShaderState::OptimizeAttribs
			| CadPL::ShaderState::OptimizeTextures
			| CadPL::ShaderState::OptimizeMaterial
			).to_ullong());

    	auto &uberSet = getUberPipelineStateSet(UberPipelineDescriptor(desc));

		if (none) {
			desc.materialSettings = materialSettings;

			uint8_t numAttributes = 0;
			for (int i = 0; i < attribInfo.size(); ++i) {
				if (attribInfo[i]) {
					numAttributes++;
				}
			}
			desc.attribInfo = attribInfo;
			desc.numAttributes = numAttributes;
			desc.attibSetup = attibSetup;

			uint8_t numTextures = 0;
			for (int i = 0; i < textureSetup.size(); ++i) {
				if (textureSetup[i]) {
					// std::cout << "  Add texture: " << std::hex << textureSetup[i].settings << std::dec << "\n";
					desc.textureInfo[numTextures] = textureSetup[i].settings;
					numTextures++;
				}
			}
			desc.numTextures = numTextures;
		}

    	CadR::StateSet *parent = {};
    	CadR::StateSet *target = {};
    	auto [it, newRecord] = pipelineStateSets.try_emplace(desc, renderer);
    	if (!full && !none) {
    		parent = &it->second.stateSet;
			if (newRecord) {
				it->second.compile = true;
				uberSet.stateSet.childList.append(*parent);
			}
    		desc.materialSettings = materialSettings;

    		uint8_t numAttributes = 0;
    		for (int i = 0; i < attribInfo.size(); ++i) {
    			if (attribInfo[i]) {
    				numAttributes++;
    			}
    		}
    		desc.attribInfo = attribInfo;
    		desc.numAttributes = numAttributes;
    		desc.attibSetup = attibSetup;

    		uint8_t numTextures = 0;
    		for (int i = 0; i < textureSetup.size(); ++i) {
    			if (textureSetup[i]) {
    				// std::cout << "  Add texture: " << std::hex << textureSetup[i].settings << std::dec << "\n";
    				desc.textureInfo[numTextures] = textureSetup[i].settings;
    				numTextures++;
    			}
    		}
    		desc.numTextures = numTextures;

    		std::tie(it, newRecord) = pipelineStateSets.try_emplace(desc, renderer);
    		target = &it->second.stateSet;
    		if (newRecord) {
    			parent->childList.append(*target);
    		}
    	} else {
			target = &it->second.stateSet;
			if (newRecord) {
				it->second.compile = !none;
				uberSet.stateSet.childList.append(*target);
			}
		}

		if(newRecord) {
			CadR::StateSet& s = it->second.stateSet;
			constexpr auto size = sizeof(desc.materialSettings) + sizeof(desc.attibSetup) + sizeof(desc.attribInfo);
			s.pushConstantData.resize(size);
			uint8_t *data = s.pushConstantData.data();
			*reinterpret_cast<decltype(desc.attribInfo)*>(data) = attribInfo;
			data += sizeof(attribInfo);
			*reinterpret_cast<decltype(desc.attibSetup)*>(data) = attibSetup;
			data += sizeof(attibSetup);
			*reinterpret_cast<decltype(desc.materialSettings)*>(data) = materialSettings;
			// data += sizeof(materialSettings);
		}
		return it->second;
	}

	PipelineStateSet& getUberPipelineStateSet(const UberPipelineDescriptor &desc) {
    	auto [it, newRecord] = uberPipelineStateSets.try_emplace(desc, renderer);
    	if(newRecord) {
    		CadR::StateSet& s = it->second.stateSet;
    		it->second.compile = true;
    		stateSetDrawables.childList.append(s);
    	}
    	return it->second;
    }

	void setPipelines() {
    	for(auto &pss : uberPipelineStateSets) {
    		auto &set = pss.second;
    		const auto &desc = pss.first;

    		CadPL::ShaderState shaderState{
    			.idBuffer = false,
    			.primitiveTopology = desc.primitiveTopology,
    			.projectionHandling = CadPL::ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants,
    			.lightSetup = {},  // no lights; switches between directional light, point light and spotlight
				.numLights = {},
				.textureSetup = {},  // no textures
				.numTextures = {},
				.optimizeFlags = CadPL::ShaderState::OptimizeNone,
			};

    		const auto pipelineState = getPipelineState(
				renderPass,
				window.surfaceExtent(),
				desc.frontFace,
				desc.cullMode
			);

    		std::cout << "PipelineStateSet (uber)" << &pss.second << "  " << shaderState.serialize() << "\n" << shaderState.debugDump() << "\n";
    		std::cout << "  cullMode: " << vk::to_string(desc.cullMode) << "\n";
    		std::cout << "  frontFace: " << vk::to_string(desc.frontFace) << "\n";
    		std::cout << "  primitiveTopology: " << vk::to_string(desc.primitiveTopology) << "\n";

    		if (!set.pipeline.cadrPipeline()) {
    			set.pipeline = pipelineLibrary->getOrCreatePipeline(shaderState, pipelineState);;
    			auto *ptr = set.pipeline.cadrPipeline();
    			assert(ptr && ptr->get() && "pipeline was not compiled");
    			set.stateSet.pipeline = ptr;
    		}
    	}

        for(auto &pss : pipelineStateSets) {
            const auto &d = pss.first;
            auto &set = pss.second;

            CadPL::ShaderState shaderState{
				.idBuffer = false,
				.primitiveTopology = d.primitiveTopology,
				.projectionHandling = CadPL::ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants,
				.lightSetup = {},  // no lights; switches between directional light, point light and spotlight
				.numLights = {},
				.textureSetup = {},  // no textures
				.numTextures = {},
				.optimizeFlags = CadPL::ShaderState::OptimizeNone,
            };

            const auto pipelineState = getPipelineState(
                renderPass,
                window.surfaceExtent(),
                d.frontFace,
                d.cullMode
            );

        	// if (!useUberShader) {
        		shaderState.optimizeFlags = d.optimizeFlags;
        		shaderState.attribAccessInfo = d.attribInfo;
        		shaderState.attribSetup = d.attibSetup;
        		shaderState.textureSetup = d.textureInfo;
        		shaderState.numTextures = d.numTextures;
        		shaderState.materialSetup = d.materialSettings;
        		// uint16_t lightSetup[4];
        	// }

        	if (!set.compile) {
        		std::cout << "(nocompile) PipelineStateSet " << &pss.second.stateSet << "  " << shaderState.serialize() << "\n" << shaderState.debugDump() << "\n";
        		continue;
        	}
			std::cout << "PipelineStateSet " << &pss.second.stateSet << "  " << shaderState.serialize() << "\n" << shaderState.debugDump() << "\n";
        	std::cout << "  vertexSize: " << (d.attibSetup & 0x01fc) << "\n";
        	std::cout << "  cullMode: " << vk::to_string(d.cullMode) << "\n";
        	std::cout << "  frontFace: " << vk::to_string(d.frontFace) << "\n";
        	std::cout << "  primitiveTopology: " << vk::to_string(d.primitiveTopology) << "\n";
        	if (d.attibSetup & 1) {
        		std::cout << "  generateFlatNormals\n";
        	}


        	if(useUberShader) {
        		set.stateSet.pipeline = {};
        	}
        	else {
        		if (!set.pipeline.cadrPipeline()) {
					set.pipeline = pipelineLibrary->getOrCreatePipeline(shaderState, pipelineState);
				}
        		set.stateSet.pipeline = set.pipeline.cadrPipeline();
        	}

        }

    	std::cout << "=== stats: " << uberPipelineStateSets.size() + pipelineStateSets.size() << " stateSets, " << pipelineLibrary->count() << " pipelines, " << shaderLibrary->count() << " shaderModules (" << shaderLibrary->countVertex() << "V, " << shaderLibrary->countGeometry() << "G, " << shaderLibrary->countFragment() << "F)\n";
        printStateSet(stateSetRoot);
    }

	PFN_vkVoidFunction imguiLoadFunction(const char *name) const {
	    auto func = device.getProcAddr(name);
    	if (func) {
    		return func;
    	}
    	return vulkanInstance.getProcAddr(name);
    }

    bool calibratedTimestampsSupported = false;
	bool useUberShader = false;
	bool noPause = true;
    std::chrono::steady_clock::time_point lastDebugTime;
    MedianCounter gpuTimeCounter;
    MedianCounter cpuTimeCounter;

private:

	void setCamera(glm::vec3 position, glm::vec3 target, glm::vec3 up);

	void renderGUI();

};


class ExitWithMessage {
protected:
	int _exitCode;
	string _what;
public:
	ExitWithMessage(int exitCode, const string& msg) : _exitCode(exitCode), _what(msg) {}
	ExitWithMessage(int exitCode, const char* msg) : _exitCode(exitCode), _what(msg) {}
	const char* what() const noexcept  { return _what.c_str(); }
	int exitCode() const noexcept  { return _exitCode; }
};


#ifdef _WIN32
// Do not perform utf16 to utf8 conversion on MSVC using STL.
// std::path::string() throws exception on some characters (❤, ♻) telling us:
// "No mapping for the Unicode character exists in the target multi-byte code page."
// Seen on MSVC 2022 version 17.14.20.
// Do the coversion using the following function:
static string utf16ToUtf8(const wchar_t* ws)
{
	if(ws == nullptr)
		return {};

	// alloc buffer
	unique_ptr<char[]> buffer(new char[wcslen(ws)*3+1]);

	// perform conversion
	size_t pos = 0;
	size_t i = 0;
	char16_t c = ws[0];
	while(c != 0) {
		if((c & 0xfc00) != 0xd800 || (ws[i+1] & 0xfc00) != 0xfc00) {
			if(c < 128)
				// 1 byte utf-8 sequence
				buffer[pos++] = char(c);  // bits 0x7f
			else if(c < 2048) {
				// 2 bytes utf-8 sequence
				buffer[pos++] = 0xc0 | (c >> 6);  // bits 0x07c0
				buffer[pos++] = 0x80 | (c & 0x3f);  // bits 0x3f
			}
			else {
				// 3 bytes utf-8 sequence
				buffer[pos++] = 0xe0 | (c >> 12);  // bits 0xf000
				buffer[pos++] = 0x80 | ((c >> 6) & 0x3f);  // bits 0x0fc0
				buffer[pos++] = 0x80 | (c & 0x3f);  // bits 0x3f
			}
			i++;
			c = ws[i];
		}
		else {
			char16_t cNext = ws[i+1];
			uint32_t codePoint = (uint32_t(c & 0x03ff) << 10) | (cNext & 0x03ff);
			if(codePoint >= 65536) {
				// 4 bytes utf-8 sequence
				buffer[pos++] = 0xf0 | (codePoint >> 18);  // bits 0x1c0000
				buffer[pos++] = 0x80 | ((codePoint >> 12) & 0x3f);  // bits 0x3f000
				buffer[pos++] = 0x80 | ((codePoint >> 6) & 0x3f);  // bits 0x0fc0
				buffer[pos++] = 0x80 | (codePoint & 0x3f);  // bits 0x3f
			}
			else if(codePoint >= 2048) {
				// 3 bytes utf-8 sequence
				buffer[pos++] = 0xe0 | (c >> 12);  // bits 0xf000
				buffer[pos++] = 0x80 | ((c >> 6) & 0x3f);  // bits 0x0fc0
				buffer[pos++] = 0x80 | (c & 0x3f);  // bits 0x3f
			}
			else if(codePoint >= 128) {
				// 2 bytes utf-8 sequence
				buffer[pos++] = 0xc0 | (c >> 6);  // bits 0x07c0
				buffer[pos++] = 0x80 | (c & 0x3f);  // bits 0x3f
			}
			else {
				// 1 byte utf-8 sequence
				buffer[pos++] = char(c);  // bits 0x7f
			}
			i += 2;
			c = ws[i];
		}
	}

	return string(buffer.get(), pos);
}
#endif


/// Construct application object
App::App(int argc, char** argv)
	: sceneDataAllocation(renderer.dataStorage())
	, stateSetRoot(renderer)
	, stateSetDrawables(renderer)
	, stateSetGUI(renderer)
	, defaultSampler(renderer)
	, defaultMaterial(renderer.dataStorage(), CadR::DataAllocation::noHandle)
{
#ifdef _WIN32
	// get wchar_t command line
	LPWSTR commandLine = GetCommandLineW();
	unique_ptr<wchar_t*, void(*)(wchar_t**)> wargv(
		CommandLineToArgvW(commandLine, &argc),
		[](wchar_t** p) { if(LocalFree(p) != 0) assert(0 && "LocalFree() failed."); }
	);
#endif

	// process command-line arguments
	if(argc < 2)
		throw ExitWithMessage(99, "Please, specify glTF file to load.");

#ifdef _WIN32
	filePath = wargv.get()[1];
	utf8FilePath = utf16ToUtf8(filePath.c_str());
	utf8FileName = utf16ToUtf8(filePath.filename().c_str());
#else
	utf8FilePath = argv[1];
	filePath = utf8FilePath;
	utf8FileName = filePath.filename();
#endif
	if(argc > 2) {
		try {
			int i = std::stoi(argv[2]);
			if (i < OptimizeLevels.size()) {
				optimizeFlags = OptimizeLevels[i];
			}
		} catch (...) {
		}
	}
}


App::~App()
{
	if(device) {

		// wait for device idle state
		// (to prevent errors during destruction of Vulkan resources);
		// we ignore any returned error codes here
		// because the device might be in the lost state already, etc.
		device.vkDeviceWaitIdle(device.handle());

		ImGui_ImplVulkan_Shutdown();
#if defined(USE_PLATFORM_WIN32)
		ImGui_ImplWin32_Shutdown();
#elif defined(USE_PLATFORM_SDL3)
		ImGui_ImplSDL3_Shutdown();
#elif defined(USE_PLATFORM_SDL2)
		ImGui_ImplSDL2_Shutdown();
#elif defined(USE_PLATFORM_GLFW)
		ImGui_ImplGlfw_Shutdown();
#endif
		ImGui::DestroyContext();

		device.destroy(imguiPipelineCache);
		device.destroy(imguiDescriptorPool);

		// destroy handles
		// (the handles are destructed in certain (not arbitrary) order)
		pipelineSceneGraph.destroy();
		stateSetRoot.destroy();
		stateSetDrawables.destroy();
		stateSetGUI.destroy();
		for(auto& pss : pipelineStateSets)
		    pss.second.destroy();
		for(auto& pss : uberPipelineStateSets)
			pss.second.destroy();
		textureList.clear();
		imageList.clear();
		samplerList.clear();
		materialList.clear();
		matrixLists.clear();
		defaultSampler.destroy();
		defaultMaterial.free();
		drawableList.clear();
		geometryList.clear();
		sceneDataAllocation.free();
		pipelineLibrary = nullptr;
		shaderLibrary = nullptr;
		device.destroy(commandPool);
		renderer.finalize();
		device.destroy(renderingFinishedFence);
		device.destroy(imageAvailableSemaphore);
		for(auto f : framebuffers)  device.destroy(f);
		for(auto v : swapchainImageViews)  device.destroy(v);
		for(auto s : renderingFinishedSemaphores)  device.destroy(s);
		device.destroy(depthImage);
		device.freeMemory(depthImageMemory);
		device.destroy(depthImageView);
		device.destroy(swapchain);
		device.destroy(renderPass);
		device.destroy();

	}

	window.destroy();
#if defined(USE_PLATFORM_XLIB) || defined(USE_PLATFORM_QT)
	// On Xlib, VulkanWindow::finalize() needs to be called before instance destroy to avoid crash.
	// It is workaround for the known bug in libXext: https://gitlab.freedesktop.org/xorg/lib/libxext/-/issues/3,
	// that crashes the application inside XCloseDisplay(). The problem seems to be present
	// especially on Nvidia drivers (reproduced on versions 470.129.06 and 515.65.01, for example).
	//
	// On Qt, VulkanWindow::finalize() needs to be called not too late after leaving main() because
	// crash might follow. Probably Qt gets partly finalized. Seen on Linux with Qt 5.15.13 and Qt 6.4.2 on 2024-05-03.
	// Calling VulkanWindow::finalize() before leaving main() seems to be a safe solution. For simplicity, we are doing it here.
	VulkanWindow::finalize();
#endif
	vulkanInstance.destroy();
}


void App::init()
{
	// open file
	ifstream f(filePath);
	if(!f.is_open()) {
		string msg("Cannot open file ");
		msg.append(utf8FilePath);
		msg.append(".");
		throw ExitWithMessage(1, msg);
	}
	f.exceptions(ifstream::badbit | ifstream::failbit);

	stateSetRoot.childList.append(stateSetDrawables);
	// stateSetRoot.childList.append(stateSetGUI);

	// init Vulkan and window
	VulkanWindow::init();
	vulkanLib.load(CadR::VulkanLibrary::defaultName());
	vulkanInstance.create(vulkanLib, "glTF reader", 0, "CADR", 0, VK_API_VERSION_1_2,
#ifdef VULKAN_VALIDATION
		{"VK_LAYER_KHRONOS_validation"},
#else
		nullptr,
#endif
	                      VulkanWindow::requiredExtensions());
	window.create(vulkanInstance.handle(), {1024,768}, "glTF reader - " + utf8FileName,
	              vulkanLib.vkGetInstanceProcAddr);

	// init device and renderer
	tuple<vk::PhysicalDevice, uint32_t, uint32_t> deviceAndQueueFamilies =
			vulkanInstance.chooseDevice(
				vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute,  // queueOperations
				window.surface(),  // presentationSurface
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
						features.get<vk::PhysicalDeviceVulkan12Features>().bufferDeviceAddress &&
						features.get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray &&
						features.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingSampledImageUpdateAfterBind &&
						features.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingUpdateUnusedWhilePending &&
						features.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingPartiallyBound &&
						features.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingVariableDescriptorCount &&
						features.get<vk::PhysicalDeviceFeatures2>().features.geometryShader;
				});
	physicalDevice = std::get<0>(deviceAndQueueFamilies);

	std::vector<const char*> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
	for (const auto &ext : vulkanInstance.enumerateDeviceExtensionProperties(physicalDevice)) {
		if (std::strcmp(ext.extensionName, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME) == 0) {
			calibratedTimestampsSupported = true;
			extensions.emplace_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
		}
	}

	if(!physicalDevice)
		throw ExitWithMessage(2, "No compatible Vulkan device found.");
	device.create(
		vulkanInstance, deviceAndQueueFamilies,
#if 1 // enable or disable validation extensions
	  // (0 enables validation extensions and features for debugging purposes)
		extensions,
		[]() {
			CadR::Renderer::RequiredFeaturesStructChain f = CadR::Renderer::requiredFeaturesStructChain();
			f.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy = true;  // required by samplers, or disable it in samplers when the feature is not available
			f.get<vk::PhysicalDeviceFeatures2>().features.geometryShader = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingSampledImageUpdateAfterBind = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingUpdateUnusedWhilePending = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingPartiallyBound = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingVariableDescriptorCount = true;  // required by CadPL
			f.get<vk::PhysicalDeviceFeatures2>().features.geometryShader = true;
			f.get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray = true;

			return f;
		}().get<vk::PhysicalDeviceFeatures2>()
#else
		{"VK_KHR_swapchain", "VK_KHR_shader_non_semantic_info"},
		[]() {
			CadR::Renderer::RequiredFeaturesStructChain f = CadR::Renderer::requiredFeaturesStructChain();
			f.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy = true;  // required by samplers, or disable it in samplers when the feature is not available
			f.get<vk::PhysicalDeviceFeatures2>().features.geometryShader = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().runtimeDescriptorArray = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingSampledImageUpdateAfterBind = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingUpdateUnusedWhilePending = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingPartiallyBound = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().descriptorBindingVariableDescriptorCount = true;  // required by CadPL
			f.get<vk::PhysicalDeviceVulkan12Features>().uniformAndStorageBuffer8BitAccess = true;
			return f;
		}().get<vk::PhysicalDeviceFeatures2>()
#endif
	);
	graphicsQueueFamily = std::get<1>(deviceAndQueueFamilies);
	presentationQueueFamily = std::get<2>(deviceAndQueueFamilies);
	window.setDevice(device.handle(), physicalDevice);
	renderer.init(device, vulkanInstance, physicalDevice, graphicsQueueFamily);
	renderer.setCollectFrameInfo(true, calibratedTimestampsSupported);
	pipelineSceneGraph.init(stateSetDrawables, {
		CadPL::ShaderState::OptimizeNone, CadPL::ShaderState::OptimizeAttribs, CadPL::ShaderState::OptimizeAll,
	}, nullptr, 2048);

	// get queues
	graphicsQueue = device.getQueue(graphicsQueueFamily, 0);
	presentationQueue = device.getQueue(presentationQueueFamily, 0);

	// choose surface format
	surfaceFormat =
		[](vk::PhysicalDevice physicalDevice, CadR::VulkanInstance& vulkanInstance, vk::SurfaceKHR surface)
		{
			constexpr const array candidateFormats{
				vk::SurfaceFormatKHR{ vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear },
				vk::SurfaceFormatKHR{ vk::Format::eR8G8B8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear },
				vk::SurfaceFormatKHR{ vk::Format::eA8B8G8R8SrgbPack32, vk::ColorSpaceKHR::eSrgbNonlinear },
			};
			vector<vk::SurfaceFormatKHR> availableFormats =
				physicalDevice.getSurfaceFormatsKHR(surface, vulkanInstance);
			if(availableFormats.size()==1 && availableFormats[0].format==vk::Format::eUndefined)
				// Vulkan spec allowed single eUndefined value until 1.1.111 (2019-06-10)
				// with the meaning you can use any valid vk::Format value.
				// Now, it is forbidden, but let's handle any old driver.
				return candidateFormats[0];
			else {
				for(vk::SurfaceFormatKHR sf : availableFormats) {
					auto it = std::find(candidateFormats.begin(), candidateFormats.end(), sf);
					if(it != candidateFormats.end())
						return *it;
				}
				if(availableFormats.size() == 0)  // Vulkan must return at least one format (this is mandated since Vulkan 1.0.37 (2016-10-10), but was missing in the spec before probably because of omission)
					throw std::runtime_error("Vulkan error: getSurfaceFormatsKHR() returned empty list.");
				return availableFormats[0];
			}
		}(physicalDevice, vulkanInstance, window.surface());

	// choose depth format
	depthFormat =
		[](vk::PhysicalDevice physicalDevice, CadR::VulkanInstance& vulkanInstance)
		{
			constexpr const array<vk::Format, 3> candidateFormats {
				vk::Format::eD32Sfloat,
				vk::Format::eD32SfloatS8Uint,
				vk::Format::eD24UnormS8Uint,
			};
			for(vk::Format f : candidateFormats) {
				vk::FormatProperties p = physicalDevice.getFormatProperties(f, vulkanInstance);
				if(p.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
					return f;
			}
			throw std::runtime_error("No suitable depth buffer format.");
		}(physicalDevice, vulkanInstance);

	// maxSamplerAnisotropy
	maxSamplerAnisotropy = vulkanInstance.getPhysicalDeviceProperties(physicalDevice).limits.maxSamplerAnisotropy;

	// render pass
	renderPass =
		device.createRenderPass(
			vk::RenderPassCreateInfo(
				vk::RenderPassCreateFlags(),  // flags
				2,                            // attachmentCount
				array<const vk::AttachmentDescription, 2>{  // pAttachments
					vk::AttachmentDescription{  // color attachment
						vk::AttachmentDescriptionFlags(),  // flags
						surfaceFormat.format,              // format
						vk::SampleCountFlagBits::e1,       // samples
						vk::AttachmentLoadOp::eClear,      // loadOp
						vk::AttachmentStoreOp::eStore,     // storeOp
						vk::AttachmentLoadOp::eDontCare,   // stencilLoadOp
						vk::AttachmentStoreOp::eDontCare,  // stencilStoreOp
						vk::ImageLayout::eUndefined,       // initialLayout
						vk::ImageLayout::ePresentSrcKHR    // finalLayout
					},
					vk::AttachmentDescription{  // depth attachment
						vk::AttachmentDescriptionFlags(),  // flags
						depthFormat,                       // format
						vk::SampleCountFlagBits::e1,       // samples
						vk::AttachmentLoadOp::eClear,      // loadOp
						vk::AttachmentStoreOp::eDontCare,  // storeOp
						vk::AttachmentLoadOp::eDontCare,   // stencilLoadOp
						vk::AttachmentStoreOp::eDontCare,  // stencilStoreOp
						vk::ImageLayout::eUndefined,       // initialLayout
						vk::ImageLayout::eDepthStencilAttachmentOptimal  // finalLayout
					},
				}.data(),
				1,  // subpassCount
				&(const vk::SubpassDescription&)vk::SubpassDescription(  // pSubpasses
					vk::SubpassDescriptionFlags(),     // flags
					vk::PipelineBindPoint::eGraphics,  // pipelineBindPoint
					0,        // inputAttachmentCount
					nullptr,  // pInputAttachments
					1,        // colorAttachmentCount
					&(const vk::AttachmentReference&)vk::AttachmentReference(  // pColorAttachments
						0,  // attachment
						vk::ImageLayout::eColorAttachmentOptimal  // layout
					),
					nullptr,  // pResolveAttachments
					&(const vk::AttachmentReference&)vk::AttachmentReference(  // pDepthStencilAttachment
						1,  // attachment
						vk::ImageLayout::eDepthStencilAttachmentOptimal  // layout
					),
					0,        // preserveAttachmentCount
					nullptr   // pPreserveAttachments
				),
				2,  // dependencyCount
				array{  // pDependencies
					vk::SubpassDependency(
						VK_SUBPASS_EXTERNAL,   // srcSubpass
						0,                     // dstSubpass
						vk::PipelineStageFlags(  // srcStageMask
							vk::PipelineStageFlagBits::eColorAttachmentOutput |
							vk::PipelineStageFlagBits::eComputeShader |
							vk::PipelineStageFlagBits::eTransfer),
						vk::PipelineStageFlags(  // dstStageMask
							vk::PipelineStageFlagBits::eDrawIndirect | vk::PipelineStageFlagBits::eVertexInput |
							vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eFragmentShader |
							vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests |
							vk::PipelineStageFlagBits::eColorAttachmentOutput),
						vk::AccessFlags(vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite),  // srcAccessMask
						vk::AccessFlags(  // dstAccessMask
							vk::AccessFlagBits::eIndirectCommandRead | vk::AccessFlagBits::eIndexRead |
							vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eColorAttachmentWrite |
							vk::AccessFlagBits::eDepthStencilAttachmentWrite),
						vk::DependencyFlags()  // dependencyFlags
					),
					vk::SubpassDependency(
						0,                    // srcSubpass
						VK_SUBPASS_EXTERNAL,  // dstSubpass
						vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput),
						vk::PipelineStageFlags(vk::PipelineStageFlagBits::eBottomOfPipe),  // dstStageMask
						vk::AccessFlags(vk::AccessFlagBits::eColorAttachmentWrite),
						vk::AccessFlags(),     // dstAccessMask
						vk::DependencyFlags()  // dependencyFlags
					),
				}.data()

			)
		);

	// rendering semaphore and fence
	imageAvailableSemaphore =
		device.createSemaphore(
			vk::SemaphoreCreateInfo(
				vk::SemaphoreCreateFlags()  // flags
			)
		);
	renderingFinishedFence =
		device.createFence(
			vk::FenceCreateInfo(
				vk::FenceCreateFlagBits::eSignaled  // flags
			)
		);


	// command buffer
	commandPool =
		device.createCommandPool(
			vk::CommandPoolCreateInfo(
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,  // flags
				graphicsQueueFamily  // queueFamilyIndex
			)
		);
	commandBuffer =
		device.allocateCommandBuffers(
			vk::CommandBufferAllocateInfo(
				commandPool,                       // commandPool
				vk::CommandBufferLevel::ePrimary,  // level
				1                                  // commandBufferCount
			)
		)[0];

	// imgui
	{
		ImGui::CreateContext();
#if defined(USE_PLATFORM_WIN32)
		ImGui_ImplWin32_Init(window.handle());
#elif defined(USE_PLATFORM_SDL2) || defined(USE_PLATFORM_SDL3)
		ImGui_ImplSDL2_InitForVulkan(window.handle());
#elif defined(USE_PLATFORM_GLFW)
		ImGui_ImplGlfw_InitForVulkan(window.handle(), true);
#endif

		ImGui_ImplVulkan_LoadFunctions(vulkanInstance.getPhysicalDeviceProperties(physicalDevice).apiVersion, [](const char* name, void* data) {
			return reinterpret_cast<const App*>(data)->imguiLoadFunction(name);
		}, this);

		const auto poolSizes = std::array{
			vk::DescriptorPoolSize{  vk::DescriptorType::eCombinedImageSampler, IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE },
		};
		uint32_t maxSets = 0;
		for (const auto size : poolSizes) {
			maxSets += size.descriptorCount;
		}

		imguiPipelineCache = device.createPipelineCache(
			vk::PipelineCacheCreateInfo(
				vk::PipelineCacheCreateFlags(),  // flags
				0,       // initialDataSize
				nullptr  // pInitialData
			)
		);
		imguiDescriptorPool = device.createDescriptorPool(
			vk::DescriptorPoolCreateInfo(
				vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,  // flags
				maxSets,  // maxSets
				poolSizes.size(),  // poolSizeCount
				poolSizes.data()  // pPoolSizes
			)
		);

		// uint32_t imageCount = device.getSwapchainImagesKHR(swapchain).size(); swapchain not created yet
		ImGui_ImplVulkan_InitInfo init_info = {
			.Instance        = vulkanInstance.handle(),
			.PhysicalDevice  = physicalDevice,
			.Device          = device.handle(),
			.QueueFamily     = graphicsQueueFamily,
			.Queue           = graphicsQueue,
			.DescriptorPool  = imguiDescriptorPool,
			.MinImageCount   = 2,
			.ImageCount      = 2,
			.PipelineCache   = imguiPipelineCache,
			.PipelineInfoMain = ImGui_ImplVulkan_PipelineInfo{
				.RenderPass = renderPass,
				.Subpass = 0,
				.MSAASamples = VK_SAMPLE_COUNT_1_BIT
			},
			.Allocator       = nullptr,
			.CheckVkResultFn = [](VkResult result) {
				if (result != VK_SUCCESS) {
					throw ExitWithMessage(1, "ImGui init error.");
				}
			}
		};

		ImGui_ImplVulkan_Init(&init_info);

		stateSetGUI.setForceRecording(true);
		stateSetGUI.recordCallList.emplace_back([&](CadR::StateSet&, vk::CommandBuffer commandBuffer, vk::PipelineLayout) {
			ImDrawData *data = ImGui::GetDrawData();
			if (data) {
				ImGui_ImplVulkan_RenderDrawData(data, commandBuffer);
			}
		});
	}

	// CadPL
    shaderLibrary = std::make_unique<CadPL::ShaderLibrary>(device, 2048);
    pipelineLibrary = std::make_unique<CadPL::PipelineLibrary>(*shaderLibrary, renderer.pipelineCache());

	// parse json
	cout << "Processing file " << utf8FilePath << "..." << endl;
	json glTF, newGltfItems;
	f >> glTF;
	f.close();

	// read root objects
	// (asset item is mandatory, the rest is optional)
	auto getRootItem =
		[](json& glTF, json& newGltfItems, const string& key) -> json::array_t&
		{
			auto it = glTF.find(key);
			if(it != glTF.end())
				return it->get_ref<json::array_t&>();
			auto& ref = newGltfItems[key];
			if(ref.is_null())
				ref = json::array();
			return ref.get_ref<json::array_t&>();
		};
	auto& asset = glTF.at("asset");
	auto& scenes = getRootItem(glTF, newGltfItems, "scenes");
	auto& nodes = getRootItem(glTF, newGltfItems, "nodes");
	auto& meshes = getRootItem(glTF, newGltfItems, "meshes");
	auto& accessors = getRootItem(glTF, newGltfItems, "accessors");
	auto& buffers = getRootItem(glTF, newGltfItems, "buffers");
	auto& bufferViews = getRootItem(glTF, newGltfItems, "bufferViews");
	auto& materials = getRootItem(glTF, newGltfItems, "materials");
	auto& textures = getRootItem(glTF, newGltfItems, "textures");
	auto& images = getRootItem(glTF, newGltfItems, "images");
	auto& samplers = getRootItem(glTF, newGltfItems, "samplers");

	// print glTF info
	// (version item is mandatory, the rest is optional)
	auto getStringWithDefault =
		[](json& node, const string& key, const string& defaultVal)
		{
			auto it = node.find(key);
			return (it != node.end())
				? it->get_ref<json::string_t&>()
				: defaultVal;
		};
	cout << endl;
	cout << "glTF info:" << endl;
	cout << "   Version:     " << asset.at("version").get_ref<json::string_t&>() << endl;
	cout << "   MinVersion:  " << getStringWithDefault(asset, "minVersion", "< none >") << endl;
	cout << "   Generator:   " << getStringWithDefault(asset, "generator", "< none >") << endl;
	cout << "   Copyright:   " << getStringWithDefault(asset, "copyright", "< none >") << endl;
	cout << endl;

	// print stats
	cout << "Stats:" << endl;
	cout << "   Scenes:    " << scenes.size() << endl;
	cout << "   Nodes:     " << nodes.size() << endl;
	cout << "   Meshes:    " << meshes.size() << endl;
	cout << "   Textures:  " << textures.size() << endl;
	cout << endl;

	// read buffers
	vector<vector<uint8_t>> bufferDataList;
	bufferDataList.reserve(buffers.size());
	for(auto& b : buffers) {

		// get path
		auto uriIt = b.find("uri");
		if(uriIt == b.end())
			throw GltfError("Unsupported functionality: Undefined buffer.uri.");
		const string& s = uriIt->get_ref<json::string_t&>();
		filesystem::path p = u8string_view(reinterpret_cast<const char8_t*>(s.data()), s.size());
		if(p.is_relative())
			p = filePath.parent_path() / p;

		// open file
		cout << "Opening buffer " << s << "..." << endl;
		ifstream f(p, ios::in|ios::binary);
		if(!f.is_open())
			throw GltfError("Error opening file " + p.string() + ".");
		f.exceptions(ifstream::badbit | ifstream::failbit);

		// read content
		size_t size = filesystem::file_size(p);
		auto& bufferData = bufferDataList.emplace_back(size);
		f.read(reinterpret_cast<char*>(bufferData.data()), size);
		f.close();
	}

	// read nodes
	struct Node {
		vector<size_t> children;
		glm::mat4 matrix;
		size_t meshIndex;
	};
	vector<Node> nodeList(nodes.size());
	for(size_t i=0,c=nodes.size(); i<c; i++) {

		// get references
		json::object_t& jobj = nodes[i].get_ref<json::object_t&>();
		Node& node = nodeList[i];

		// matrix
		if(auto it = jobj.find("matrix"); it != jobj.end()) {

			// translation, rotation and scale must not be present
			if(jobj.find("translation") != jobj.end() || jobj.find("rotation") != jobj.end() ||
			   jobj.find("scale") != jobj.end())
				throw GltfError("If matrix is provided for the node, translation, rotation and scale must not be present.");

			// read matrix
			json::array_t& a = it->second.get_ref<json::array_t&>();
			if(a.size() != 16)
				throw GltfError("Node.matrix is not vector of 16 components.");
			float* f = glm::value_ptr(node.matrix);
			for(unsigned j=0; j<16; j++)
				f[j] = float(a[j].get<json::number_float_t>());
		}
		else {

			// read scale
			glm::vec3 scale;
			if(auto it = jobj.find("scale"); it != jobj.end()) {
				json::array_t& a = it->second.get_ref<json::array_t&>();
				if(a.size() != 3)
					throw GltfError("Node.scale is not vector of three components.");
				scale.x = float(a[0].get<json::number_float_t>());
				scale.y = float(a[1].get<json::number_float_t>());
				scale.z = float(a[2].get<json::number_float_t>());
			}
			else
				scale = { 1.f, 1.f, 1.f };

			// read rotation
			if(auto it = jobj.find("rotation"); it != jobj.end()) {
				json::array_t& a = it->second.get_ref<json::array_t&>();
				if(a.size() != 4)
					throw GltfError("Node.rotation is not vector of four components.");
				glm::quat q;
				q.x = float(a[0].get<json::number_float_t>());
				q.y = float(a[1].get<json::number_float_t>());
				q.z = float(a[2].get<json::number_float_t>());
				q.w = float(a[3].get<json::number_float_t>());
				glm::mat3 m = glm::mat3(q);
				node.matrix[0] = glm::vec4(m[0] * scale.x, 0.f);
				node.matrix[1] = glm::vec4(m[1] * scale.y, 0.f);;
				node.matrix[2] = glm::vec4(m[2] * scale.z, 0.f);;
				node.matrix[3] = glm::vec4(0.f, 0.f, 0.f, 1.f);
			}
			else {
				// initialize matrix by scale only
				memset(&node.matrix, 0, sizeof(node.matrix));
				node.matrix[0][0] = scale.x;
				node.matrix[1][1] = scale.y;
				node.matrix[2][2] = scale.z;
				node.matrix[3][3] = 1.f;
			}

			// read translation
			if(auto it = jobj.find("translation"); it != jobj.end()) {
				json::array_t& a = it->second.get_ref<json::array_t&>();
				if(a.size() != 3)
					throw GltfError("Node.translation is not vector of three components.");
				node.matrix[3][0] = float(a[0].get<json::number_float_t>());
				node.matrix[3][1] = float(a[1].get<json::number_float_t>());
				node.matrix[3][2] = float(a[2].get<json::number_float_t>());
			}

		}

		// read children
		if(auto it = jobj.find("children"); it != jobj.end()) {
			json::array_t& a = it->second.get_ref<json::array_t&>();
			size_t size = a.size();
			node.children.resize(size);
			for(unsigned j=0; j<size; j++)
				node.children[j] = size_t(a[j].get_ref<json::number_unsigned_t&>());
		}

		// read mesh index
		if(auto it = jobj.find("mesh"); it != jobj.end())
			node.meshIndex = size_t(it->second.get_ref<json::number_unsigned_t&>());
		else
			node.meshIndex = ~size_t(0);
	}

	// get default scene
	size_t numScenes = scenes.size();
	if(numScenes == 0)
		return;
	size_t defaultSceneIndex = glTF.value<json::number_unsigned_t>("scene", ~size_t(0));
	if(defaultSceneIndex == ~size_t(0)) {
		cout << "There is no default scene in the file. Using the first scene." << endl;
		defaultSceneIndex = 0;
	}
	json& scene = scenes.at(defaultSceneIndex);

	// iterate through root nodes
	// and fill meshMatrixList
	vector<vector<glm::mat4>> meshMatrixList(meshes.size());
	if(auto rootNodesIt=scene.find("nodes"); rootNodesIt!=scene.end()) {
		json& rootNodes = *rootNodesIt;
		for(auto it=rootNodes.begin(); it!= rootNodes.end(); it++) {

			// process node function
			auto processNode =
				[](size_t nodeIndex, const glm::mat4& parentMatrix, vector<Node>& nodeList,
				   vector<vector<glm::mat4>>& meshMatrixList, const auto& processNode) -> void
				{
					// get node
					Node& node = nodeList.at(nodeIndex);

					// compute local matrix
					glm::mat4 m = parentMatrix * node.matrix;

					// assign one more instancing matrix to the mesh
					if(node.meshIndex != ~size_t(0))
						meshMatrixList.at(node.meshIndex).emplace_back(m);

					// process children
					for(size_t i=0,c=node.children.size(); i<c; i++)
						processNode(node.children[i], m, nodeList, meshMatrixList, processNode);
				};

			// get node
			size_t rootNodeIndex = size_t(it->get_ref<json::number_unsigned_t&>());
			Node& node = nodeList.at(rootNodeIndex);

			// compute root matrix
			// (we need to flip y and z axes to get from glTF coordinate system to Vulkan coordinate system)
			glm::mat4 m = node.matrix;
			for(unsigned i=0; i<4; i++) {
				m[i][1] = -m[i][1];
				m[i][2] = -m[i][2];
			}

			// append instancing matrix to the mesh
			if(node.meshIndex != ~size_t(0))
				meshMatrixList.at(node.meshIndex).emplace_back(m);

			// process children
			for(size_t i=0,c=node.children.size(); i<c; i++)
				processNode(node.children[i], m, nodeList, meshMatrixList, processNode);

		}
	}

	// matrixLists
	matrixLists.reserve(meshes.size());
	for(size_t i=0, c=meshes.size(); i<c; i++) {
		CadR::MatrixList& ml = matrixLists.emplace_back(renderer);
		ml.setMatrices(meshMatrixList[i]);
	}

	// process images
	vector<vk::Format> imageFormats;
	if(!images.empty()) {

		// is R8G8B8Srgb format supported?
		vk::FormatProperties fp = vulkanInstance.getPhysicalDeviceFormatProperties(
			physicalDevice, vk::Format::eR8G8B8Srgb);
		bool rgb8srgbSupported =
			(fp.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear) &&
			(fp.optimalTilingFeatures & vk::FormatFeatureFlagBits::eTransferDst);
		fp = vulkanInstance.getPhysicalDeviceFormatProperties(physicalDevice, vk::Format::eR8G8Srgb);
		bool rg8srgbSupported =
			(fp.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear) &&
			(fp.optimalTilingFeatures & vk::FormatFeatureFlagBits::eTransferDst);

		// load images
		size_t c = images.size();
		cout << "Processing images (" << c << " in total)..." << endl;
		imageList.reserve(c);
		imageFormats.resize(c, vk::Format::eUndefined);
		for(size_t i=0; i<c; i++) {
			auto& image = images[i];
			auto uriIt = image.find("uri");
			if(uriIt != image.end()) {

				// image file name
				const string& imageFileName = uriIt->get_ref<json::string_t&>();
				cout << "   " << imageFileName;
				filesystem::path p = u8string_view(reinterpret_cast<const char8_t*>(imageFileName.data()), imageFileName.size());
				if(p.is_relative())
					p = filePath.parent_path() / p;

				// open stream
				ifstream fs(p, ios_base::in | ios_base::binary);
				if(!fs)
					goto failed;
				else {

					// file size
					fs.seekg(0, ios_base::end);
					size_t fileSize = fs.tellg();
					fs.seekg(0, ios_base::beg);

					// read file content
					unique_ptr<unsigned char[]> imgBuffer = make_unique<unsigned char[]>(fileSize);
					fs.read(reinterpret_cast<ifstream::char_type*>(imgBuffer.get()), fileSize);
					if(!fs)
						goto failed;
					fs.close();

					// image info
					int width, height, numComponents;
					if(!stbi_info_from_memory(imgBuffer.get(), int(fileSize), &width, &height, &numComponents))
						goto failed;
					vk::Format format;
					size_t alignment;
					switch(numComponents) {
					case 4: // red, green, blue, alpha
						format = vk::Format::eR8G8B8A8Srgb;
						alignment = 4;
						break;
					case 3: // red, green, blue
						if(rgb8srgbSupported) {
							format = vk::Format::eR8G8B8Srgb;
							alignment = 3;
						}
						else {
							// fallback to alpha always one
							format = vk::Format::eR8G8B8A8Srgb;
							alignment = 4;
							numComponents = 4;
						}
						break;
					case 2: // grey, alpha
						if(rg8srgbSupported) {
							format = vk::Format::eR8G8Srgb;
							alignment = 2;
						}
						else {
							// fallback to expand grey into red+green+blue
							format = vk::Format::eR8G8B8A8Srgb;
							alignment = 4;
							numComponents = 4;
						}
						break;
					default: goto failed;
					}

					// load image
					unique_ptr<stbi_uc[], void(*)(stbi_uc*)> data(
						stbi_load_from_memory(imgBuffer.get(), int(fileSize),
							&width, &height, nullptr, numComponents),
						[](stbi_uc* ptr) { stbi_image_free(ptr); }
					);
					if(data == nullptr)
						goto failed;

					// copy data to staging buffer
					size_t bufferSize = size_t(width) * height * numComponents;
					CadR::StagingBuffer sb(renderer.imageStorage(), bufferSize, alignment);
					memcpy(sb.data(), data.get(), bufferSize);
					data.reset();

					// create ImageAllocation
					CadR::ImageAllocation& a = imageList.emplace_back(renderer.imageStorage());
					imageFormats[imageList.size()-1] = format;
					a.alloc(
						vk::MemoryPropertyFlagBits::eDeviceLocal,  // requiredFlags
						vk::ImageCreateInfo(  // imageCreateInfo
							vk::ImageCreateFlags{},  // flags
							vk::ImageType::e2D,  // imageType
							format,  // format
							vk::Extent3D(width, height, 1),  // extent
							1,  // mipLevels
							1,  // arrayLayers
							vk::SampleCountFlagBits::e1,  // samples
							vk::ImageTiling::eOptimal,  // tiling
							vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,  // usage
							vk::SharingMode::eExclusive,  // sharingMode
							0,  // queueFamilyIndexCount
							nullptr,  // pQueueFamilyIndices
							vk::ImageLayout::eUndefined  // initialLayout
						),
						device  // vulkanDevice
					);
					sb.submit(
						a,  // ImageAllocation
						vk::ImageLayout::eUndefined,  // currentLayout,
						vk::ImageLayout::eTransferDstOptimal,  // copyLayout,
						vk::ImageLayout::eShaderReadOnlyOptimal,  // newLayout,
						vk::PipelineStageFlagBits::eFragmentShader,  // newLayoutBarrierDstStages,
						vk::AccessFlagBits::eShaderRead,  // newLayoutBarrierDstAccessFlags,
						vk::Extent2D(width, height),  // imageExtent
						bufferSize  // dataSize
					);

				}
				goto succeed;
			failed:
				cout << " - failed" << endl;
				throw GltfError("Failed to load texture " + imageFileName + ".");
			succeed:
				cout << endl;
			}
		}
	}

	// process image samplers
	if(!samplers.empty()) {

		// default settings for samplers
		vk::SamplerCreateInfo samplerCreateInfo(
			vk::SamplerCreateFlags(),  // flags
			vk::Filter::eNearest,  // magFilter - will be set later
			vk::Filter::eNearest,  // minFilter - will be set later
			vk::SamplerMipmapMode::eNearest,  // mipmapMode - will be set later
			vk::SamplerAddressMode::eRepeat,  // addressModeU - will be set later
			vk::SamplerAddressMode::eRepeat,  // addressModeV - will be set later
			vk::SamplerAddressMode::eRepeat,  // addressModeW
			0.f,  // mipLodBias
			VK_TRUE,  // anisotropyEnable
			maxSamplerAnisotropy,  // maxAnisotropy
			VK_FALSE,  // compareEnable
			vk::CompareOp::eNever,  // compareOp
			0.f,  // minLod
			0.f,  // maxLod
			vk::BorderColor::eFloatTransparentBlack,  // borderColor
			VK_FALSE  // unnormalizedCoordinates
		);

		// read image samplers
		size_t c = samplers.size();
		samplerList.reserve(c);
		for(size_t i=0; i<c; i++) {
			auto& sampler = samplers[i];

			// magFilter
			auto magFilterIt = sampler.find("magFilter");
			if(magFilterIt != sampler.end()) {
				switch(magFilterIt->get_ref<json::number_unsigned_t&>()) {
				case 9728:  // GL_NEAREST
					samplerCreateInfo.magFilter = vk::Filter::eNearest;
					break;
				case 9729:  // GL_LINEAR
					samplerCreateInfo.magFilter = vk::Filter::eLinear;
					break;
				default:
					throw GltfError("Sampler.magFilter contains invalid value.");
				}
			}
			else {
				// no defaults specified in glTF 2.0 spec
				samplerCreateInfo.magFilter = vk::Filter::eNearest;
			}

			// minFilter
			auto minFilterIt = sampler.find("minFilter");
			if(minFilterIt != sampler.end()) {
				switch(minFilterIt->get_ref<json::number_unsigned_t&>()) {
				case 9728:  // GL_NEAREST
					samplerCreateInfo.minFilter = vk::Filter::eNearest;
					samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;  // probably highest texture detail should be chosen here
					break;
				case 9729:  // GL_LINEAR
					samplerCreateInfo.minFilter = vk::Filter::eLinear;
					samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;  // probably highest texture detail should be chosen here
					break;
				case 9984:  // GL_NEAREST_MIPMAP_NEAREST
					samplerCreateInfo.minFilter = vk::Filter::eNearest;
					samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
					break;
				case 9985:  // GL_LINEAR_MIPMAP_NEAREST
					samplerCreateInfo.minFilter = vk::Filter::eLinear;
					samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
					break;
				case 9986:  // GL_NEAREST_MIPMAP_LINEAR
					samplerCreateInfo.minFilter = vk::Filter::eNearest;
					samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
					break;
				case 9987: // GL_LINEAR_MIPMAP_LINEAR
					samplerCreateInfo.minFilter = vk::Filter::eLinear;
					samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
					break;
				default:
					throw GltfError("Sampler.minFilter contains invalid value.");
				}
			}
			else {
				// no defaults specified in glTF 2.0 spec
				samplerCreateInfo.minFilter = vk::Filter::eNearest;
				samplerCreateInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
			}

			// wrapS
			auto wrapSIt = sampler.find("wrapS");
			if(wrapSIt != sampler.end()) {
				switch(wrapSIt->get_ref<json::number_unsigned_t&>()) {
				case 33071:  // GL_CLAMP_TO_EDGE
					samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
					break;
				case 33648:  // GL_MIRRORED_REPEAT
					samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eMirroredRepeat;
					break;
				case 10497:  // GL_REPEAT
					samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
					break;
				default:
					throw GltfError("Sampler.wrapS contains invalid value.");
				}
			}
			else  // default is GL_REPEAT
				samplerCreateInfo.addressModeU = vk::SamplerAddressMode::eRepeat;

			// wrapT
			auto wrapTIt = sampler.find("wrapT");
			if(wrapTIt != sampler.end()) {
				switch(wrapTIt->get_ref<json::number_unsigned_t&>()) {
				case 33071:  // GL_CLAMP_TO_EDGE
					samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
					break;
				case 33648:  // GL_MIRRORED_REPEAT
					samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eMirroredRepeat;
					break;
				case 10497:  // GL_REPEAT
					samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
					break;
				default:
					throw GltfError("Sampler.wrapT contains invalid value.");
				}
			}
			else  // default is GL_REPEAT
				samplerCreateInfo.addressModeV = vk::SamplerAddressMode::eRepeat;

			// create sampler
			samplerList.emplace_back(renderer, samplerCreateInfo);

		}
	}

	// texture descriptors
	size_t numTextures = textures.size();
	uint32_t numTextureDescriptors = numTextures>0 ? uint32_t(numTextures) : 1;
	layoutOnlyPipeline.init(nullptr, pipelineSceneGraph.pipelineLayout(), nullptr);
	stateSetDrawables.pipeline = &layoutOnlyPipeline;
	stateSetDrawables.allocDescriptorSets(
		vk::DescriptorPoolCreateInfo(
			vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,  // flags
			1,  // maxSets
			1,  // poolSizeCount
			array{  // pPoolSizes
				vk::DescriptorPoolSize(
					vk::DescriptorType::eCombinedImageSampler,  // type
					numTextureDescriptors  // descriptorCount
				),
			}.data()
		),
		pipelineSceneGraph.descriptorSetLayoutList(),
		&(const vk::DescriptorSetVariableDescriptorCountAllocateInfo&)vk::DescriptorSetVariableDescriptorCountAllocateInfo(
			1,  // descriptorSetCount
			&numTextureDescriptors  // pDescriptorCounts
		)
	);

	// process textures
	if(numTextures > 0) {
		textureList.reserve(numTextures);
		for(size_t i=0; i<numTextures; i++) {
			auto& texture = textures[i];
			auto sourceIt = texture.find("source");
			if(sourceIt == texture.end())
				throw GltfError("Unsupported functionality: Texture.source is not defined for the texture.");
			size_t imageIndex = sourceIt->get_ref<json::number_unsigned_t&>();
			auto samplerIt = texture.find("sampler");
			vk::Sampler vkSampler;
			if(samplerIt != texture.end()) {
				size_t samplerIndex = samplerIt->get_ref<json::number_unsigned_t&>();
				vkSampler = samplerList.at(samplerIndex).handle();
			}
			else {
				// create default sampler only if needed
				if(!defaultSampler.handle()) {
					defaultSampler.create(
						vk::SamplerCreateInfo(
							vk::SamplerCreateFlags(),  // flags
							vk::Filter::eNearest,  // magFilter - glTF specifies to use "auto filtering", but what is that?
							vk::Filter::eNearest,  // minFilter - glTF specifies to use "auto filtering", but what is that?
							vk::SamplerMipmapMode::eNearest,  // mipmapMode
							vk::SamplerAddressMode::eRepeat,  // addressModeU
							vk::SamplerAddressMode::eRepeat,  // addressModeV
							vk::SamplerAddressMode::eRepeat,  // addressModeW
							0.f,  // mipLodBias
							VK_TRUE,  // anisotropyEnable
							maxSamplerAnisotropy,  // maxAnisotropy
							VK_FALSE,  // compareEnable
							vk::CompareOp::eNever,  // compareOp
							0.f,  // minLod
							0.f,  // maxLod
							vk::BorderColor::eFloatTransparentBlack,  // borderColor
							VK_FALSE  // unnormalizedCoordinates
						)
					);
				}
				vkSampler = defaultSampler.handle();
			}

			// create texture
			textureList.emplace_back(
				imageList.at(imageIndex), // imageAllocation
				vk::ImageViewCreateInfo(  // imageViewCreateInfo
					vk::ImageViewCreateFlags(),  // flags
					nullptr,  // image - will be filled in later
					vk::ImageViewType::e2D,  // viewType
					imageFormats.at(imageIndex),  // format
					vk::ComponentMapping{  // components
						vk::ComponentSwizzle::eR,
						vk::ComponentSwizzle::eG,
						vk::ComponentSwizzle::eB,
						vk::ComponentSwizzle::eA,
					},
					vk::ImageSubresourceRange{  // subresourceRange
						vk::ImageAspectFlagBits::eColor,  // aspectMask
						0,  // baseMipLevel
						1,  // levelCount
						0,  // baseArrayLayer
						1,  // layerCount
					}
				),
				vkSampler,  // sampler
				device  // device
			);
		}

		// update descriptor sets
		vector<vk::DescriptorImageInfo> imageInfoList(numTextures);
		for(size_t i=0; i<numTextures; i++) {
			const CadR::Texture& t = textureList[i];
			imageInfoList[i] = {
				t.sampler(),  // sampler
				t.imageView(), // imageView
				vk::ImageLayout::eShaderReadOnlyOptimal  // imageLayout
			};
		}
		vk::WriteDescriptorSet writeInfo(
			stateSetDrawables.descriptorSet(0),  // dstSet
			0,  // dstBinding
			0,  // dstArrayElement
			uint32_t(numTextures),  // descriptorCount
			vk::DescriptorType::eCombinedImageSampler,  // descriptorType
			imageInfoList.data(),  // pImageInfo
			nullptr,  // pBufferInfo
			nullptr  // pTexelBufferView
		);
		stateSetDrawables.updateDescriptorSet(1, &writeInfo);
	}

	// create default material
	struct PhongMaterialData {
		glm::vec3 ambient;  // offset 0
		uint32_t padding1;
		glm::vec4 diffuseAndAlpha;  // offset 16
		glm::vec3 specular;  // offset 32
		float shininess;  // offset 44
		glm::vec3 emission;  // offset 48
		float pointSize;  // offset 60
		glm::vec3 reflection;  // offset 64
	};
	static_assert(sizeof(PhongMaterialData) == 76 && "Wrong size of PhongMaterialData structure");
	constexpr size_t phongMaterialDataSizeAligned8 = 80;
	struct TextureData {
		// uint8_t texCoordIndex;
		// uint8_t type;
		// uint16_t settings;
		uint32_t texCoordIndexTypeAndSettings;
		uint32_t textureIndex;
		// float strength;
		// float rs1,rs2,rs3,rs4,t1,t2;  // rotation and scale in 2x2 matrix, translation in vec2
		// float blendR,blendG,blendB;  // texture blend color used in blend texture environment
	};
	// static_assert(sizeof(TextureData) == 48 && "Wrong size of TextureData structure");
	struct StateSetMaterialData {
		std::array<TextureSetup, 6> textureSetup = {};
		uint32_t textureOffset = 0;
		bool doubleSided = false;

	};
	CadR::StagingData sd = defaultMaterial.alloc(sizeof(PhongMaterialData));
	PhongMaterialData* m = sd.data<PhongMaterialData>();
	m->ambient = glm::vec3(1.f, 1.f, 1.f);
	m->diffuseAndAlpha = glm::vec4(1.f, 1.f, 1.f, 1.f);
	m->specular = glm::vec3(0.f, 0.f, 0.f);
	m->shininess = 0.f;
	m->emission = glm::vec3(0.f, 0.f, 0.f);
	m->pointSize = 0.f;
	m->reflection = glm::vec3(0.f, 0.f, 0.f);
	StateSetMaterialData defaultStateSetMaterialData {
		.doubleSided = false
	};


	// process materials
	size_t numMaterials = materials.size();
	materialList.reserve(numMaterials);
	vector<StateSetMaterialData> stateSetMaterialDataList(numMaterials);
	for(size_t materialIndex=0; materialIndex<numMaterials; materialIndex++)
	{
		auto& material = materials.at(materialIndex);

		// StateSetMaterialData
		auto& ssm = stateSetMaterialDataList[materialIndex];
		auto &textureSetup = ssm.textureSetup;

		static constexpr auto TextureAttributeOffset = 4;
		const auto readTexure = [&](json& parent, const std::string_view key, CadPL::TextureType type) {
			if(auto it = parent.find(key); it != parent.end()) {
				auto textureIndex = it->at("index").get_ref<json::number_unsigned_t&>();
				if(textureIndex >= textureList.size())
					throw GltfError(std::string(key) + ".index is out of range. It is not index to a valid texture.");
				auto textureCoord = it->value("texCoord", 0);
				if(textureCoord >= 256)
					throw GltfError(std::string(key) + ".textureCoord is out of range");
				auto &texture = ssm.textureSetup[static_cast<size_t>(type)];
				uint32_t size = 8;
				texture.settings = (size << 26) | (static_cast<uint32_t>(type) << 8) | ((textureCoord + TextureAttributeOffset) & 0xFF);
				texture.index = textureIndex;
				// TODO optional data
			}
		};

		// values to be read from glTF
		bool doubleSided;
		glm::vec4 baseColorFactor;
		float metallicFactor;
		float roughnessFactor;
		glm::vec3 emissiveFactor;

		// material.doubleSided is optional with the default value of false
		doubleSided = material.value<json::boolean_t>("doubleSided", false);
		ssm.doubleSided = doubleSided;

		// read pbr material properties
		if(auto pbrIt = material.find("pbrMetallicRoughness"); pbrIt != material.end()) {

			// read baseColorFactor
			if(auto baseColorFactorIt = pbrIt->find("baseColorFactor"); baseColorFactorIt != pbrIt->end()) {
				json::array_t& a = baseColorFactorIt->get_ref<json::array_t&>();
				if(a.size() != 4)
					throw GltfError("Material.pbrMetallicRoughness.baseColorFactor is not vector of four components.");
				baseColorFactor[0] = float(a[0].get<json::number_float_t>());
				baseColorFactor[1] = float(a[1].get<json::number_float_t>());
				baseColorFactor[2] = float(a[2].get<json::number_float_t>());
				baseColorFactor[3] = float(a[3].get<json::number_float_t>());
			}
			else
				baseColorFactor = glm::vec4(1.f, 1.f, 1.f, 1.f);

			// read properties
			metallicFactor = float(pbrIt->value<json::number_float_t>("metallicFactor", 1.0));
			roughnessFactor = float(pbrIt->value<json::number_float_t>("roughnessFactor", 1.0));

			readTexure(pbrIt.value(), "baseColorTexture", CadPL::TextureType::base);
			readTexure(pbrIt.value(), "metallicRoughnessTexture", CadPL::TextureType::metallicRoughness);

		}
		else
		{
			// default values when pbrMetallicRoughness is not present
			baseColorFactor = glm::vec4(1.f, 1.f, 1.f, 1.f);
			metallicFactor = 1.f;
			roughnessFactor = 1.f;
		}

		// read emissiveFactor
		if(auto emissiveFactorIt = material.find("emissiveFactor"); emissiveFactorIt != material.end()) {
			json::array_t& a = emissiveFactorIt->get_ref<json::array_t&>();
			if(a.size() != 3)
				throw GltfError("Material.emissiveFactor is not vector of three components.");
			emissiveFactor[0] = float(a[0].get<json::number_float_t>());
			emissiveFactor[1] = float(a[1].get<json::number_float_t>());
			emissiveFactor[2] = float(a[2].get<json::number_float_t>());
		}
		else
			emissiveFactor = glm::vec3(0.f, 0.f, 0.f);

		readTexure(material, "normalTexture", CadPL::TextureType::normal);
		readTexure(material, "occlusionTexture", CadPL::TextureType::occlusion);
		readTexure(material, "emissiveTexture", CadPL::TextureType::emissive);
		// not supported material properties
		//
		//				if(material->find("alphaMode") != material->end())
		//					throw GltfError("Unsupported functionality: alpha mode.");
		//				if(material->find("alphaCutoff") != material->end())
		//					throw GltfError("Unsupported functionality: alpha cutoff.");

		// material size
		size_t materialSize = phongMaterialDataSizeAligned8;
		size_t numTextures = 0;
		for (const auto &tex : textureSetup) {
			if (tex) {
				// materialSize += sizeof(TextureInfoBase);
				materialSize += 8;
				++numTextures;
			}
		}
		materialSize += sizeof(uint32_t) * 2;

		materialSize += 64;

		// if (numTextures) {
			ssm.textureOffset = phongMaterialDataSizeAligned8;
		// }

		// material
		CadR::DataAllocation& a = materialList.emplace_back(renderer.dataStorage());
		CadR::StagingData sd = a.alloc(materialSize);
		PhongMaterialData* m = sd.data<PhongMaterialData>();
		m->ambient = glm::vec3(baseColorFactor);
		m->diffuseAndAlpha = baseColorFactor;
		m->specular = baseColorFactor * metallicFactor;  // very vague and imprecise conversion
		m->shininess = (1.f - roughnessFactor) * 128.f;  // very vague and imprecise conversion
		m->emission = emissiveFactor;
		m->pointSize = 0.f;
		m->reflection = glm::vec3(0.f, 0.f, 0.f);

		m->padding1 = 0x33;

		uint8_t* texturePtr = reinterpret_cast<uint8_t*>(sd.data()) + phongMaterialDataSizeAligned8;
		for (size_t i = 0; i < textureSetup.size(); ++i) {
			if (textureSetup[i]) {
				auto* textureData = reinterpret_cast<TextureData*>(texturePtr);
				// *reinterpret_cast<uint32_t*>(texturePtr) = textureSetup[i].settings;

				textureData->texCoordIndexTypeAndSettings = textureSetup[i].settings;
				textureData->textureIndex = textureSetup[i].index;
				texturePtr += 8;
			}
		}
		// terminating record (all zeros)
		*reinterpret_cast<uint64_t*>(texturePtr) = 0;


	}
	assert(materialList.size() == numMaterials && "Not all materials were created.");

	// process meshes
	cout << "Processing meshes..." << endl;
	size_t numMeshes = meshes.size();
	vector<CadR::BoundingSphere> meshBoundingSphereList(numMeshes);
	vector<CadR::BoundingSphere> primitiveSetBSList;
	for(size_t meshIndex=0; meshIndex<numMeshes; meshIndex++) {

		// ignore non-instanced meshes
		if(meshMatrixList[meshIndex].empty())
			continue;

		// get mesh
		auto& mesh = meshes[meshIndex];
		CadR::BoundingBox meshBB = CadR::BoundingBox::empty();

		// process primitives
		// (mesh.primitives are mandatory)
		auto& primitives = mesh.at("primitives");
		primitiveSetBSList.clear();
		primitiveSetBSList.reserve(primitives.size());
		for(auto& primitive : primitives) {

			// mesh.primitive helper functions
			auto getColorFromVec4f =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					glm::vec4 r = *reinterpret_cast<const glm::vec4*>(srcPtr);
					*reinterpret_cast<glm::vec4*>(dstPtr) = glm::clamp(r, 0.f, 1.f);
				};
			auto getColorFromVec3f =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							glm::clamp(*reinterpret_cast<const glm::vec3*>(srcPtr), 0.f, 1.f),
							1.f
						);
				};
			auto getColorFromVec4us =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							float(reinterpret_cast<const uint16_t*>(srcPtr)[0]) / 65535.f,
							float(reinterpret_cast<const uint16_t*>(srcPtr)[1]) / 65535.f,
							float(reinterpret_cast<const uint16_t*>(srcPtr)[2]) / 65535.f,
							float(reinterpret_cast<const uint16_t*>(srcPtr)[3]) / 65535.f
						);
				};
			auto getColorFromVec3us =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							float(reinterpret_cast<const uint16_t*>(srcPtr)[0]) / 65535.f,
							float(reinterpret_cast<const uint16_t*>(srcPtr)[1]) / 65535.f,
							float(reinterpret_cast<const uint16_t*>(srcPtr)[2]) / 65535.f,
							1.f
						);
				};
			auto getColorFromVec4ub =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							float(reinterpret_cast<const uint8_t*>(srcPtr)[0]) / 255.f,
							float(reinterpret_cast<const uint8_t*>(srcPtr)[1]) / 255.f,
							float(reinterpret_cast<const uint8_t*>(srcPtr)[2]) / 255.f,
							float(reinterpret_cast<const uint8_t*>(srcPtr)[3]) / 255.f
						);
				};
			auto getColorFromVec3ub =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							float(reinterpret_cast<const uint8_t*>(srcPtr)[0]) / 255.f,
							float(reinterpret_cast<const uint8_t*>(srcPtr)[1]) / 255.f,
							float(reinterpret_cast<const uint8_t*>(srcPtr)[2]) / 255.f,
							1.f
						);
				};
			auto getPosFromVec3 =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) = glm::vec4(
							*reinterpret_cast<const glm::vec3*>(srcPtr),
							1.f
						);
			};
			auto getVec3 =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec3*>(dstPtr) = *reinterpret_cast<const glm::vec3*>(srcPtr);
			};
			auto getVec2FromVec2f =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec2*>(dstPtr) = *reinterpret_cast<const glm::vec2*>(srcPtr);
			};
			auto getVec2FromVec2us =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec2*>(dstPtr) =
						glm::vec2(
							float(reinterpret_cast<const uint16_t*>(srcPtr)[0]) / 65535.f,
							float(reinterpret_cast<const uint16_t*>(srcPtr)[1]) / 65535.f
						);
			};
			auto getVec2FromVec2ub =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec2*>(dstPtr) =
						glm::vec2(
							float(reinterpret_cast<const uint8_t*>(srcPtr)[0]) / 255.f,
							float(reinterpret_cast<const uint8_t*>(srcPtr)[1]) / 255.f
						);
			};
			auto getVec4FromVec2f =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							*reinterpret_cast<const glm::vec2*>(srcPtr),
							0.f,
							0.f
						);
				};
			auto getVec4FromVec2us =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							float(reinterpret_cast<const uint16_t*>(srcPtr)[0]) / 65535.f,
							float(reinterpret_cast<const uint16_t*>(srcPtr)[1]) / 65535.f,
							0.f,
							0.f
						);
				};
			auto getVec4FromVec2ub =
				[](const uint8_t* srcPtr, uint8_t* dstPtr) {
					*reinterpret_cast<glm::vec4*>(dstPtr) =
						glm::vec4(
							float(reinterpret_cast<const uint8_t*>(srcPtr)[0]) / 255.f,
							float(reinterpret_cast<const uint8_t*>(srcPtr)[1]) / 255.f,
							0.f,
							0.f
						);
				};
			auto updateNumVertices =
				[](json& accessor, size_t& numVertices) -> void
				{
					// get position count (accessor.count is mandatory and >=1)
					json::number_unsigned_t count = accessor.at("count").get_ref<json::number_unsigned_t&>();

					// update numVertices if still set to 0
					if(numVertices != count) {
						if(numVertices == 0) {
							if(count != 0)
								numVertices = count;
							else
								throw GltfError("Accessor's count member must be greater than zero.");
						}
						else
							throw GltfError("Number of elements is not the same for all primitive attributes.");
					}
				};
			auto getDataPointerAndStride =
				[](json& accessor, json::array_t& bufferViews, json::array_t& buffers, vector<vector<uint8_t>>& bufferDataList,
				   size_t numElements, size_t elementSize) -> tuple<void*, unsigned>
				{
					// accessor.sparse is not supported yet
					if(accessor.find("sparse") != accessor.end())
						throw GltfError("Unsupported functionality: Property sparse.");

					// get accessor.bufferView (it is optional)
					auto bufferViewIt = accessor.find("bufferView");
					if(bufferViewIt == accessor.end())
						throw GltfError("Unsupported functionality: Omitted bufferView.");
					auto& bufferView = bufferViews.at(bufferViewIt->get_ref<json::number_unsigned_t&>());

					// bufferView.byteStride (it is optional (but mandatory in some cases), if not provided, data are tightly packed)
					unsigned stride = unsigned(bufferView.value<json::number_unsigned_t>("byteStride", elementSize));
					size_t dataSize = (numElements-1) * stride + elementSize;

					// get accessor.byteOffset (it is optional with default value 0)
					json::number_unsigned_t offset = accessor.value<json::number_unsigned_t>("byteOffset", 0);

					// make sure we not run over bufferView.byteLength (byteLength is mandatory and >=1)
					if(offset + dataSize > bufferView.at("byteLength").get_ref<json::number_unsigned_t&>())
						throw GltfError("Accessor range is not completely inside its BufferView.");

					// append bufferView.byteOffset (byteOffset is optional with default value 0)
					offset += bufferView.value<json::number_unsigned_t>("byteOffset", 0);

					// get bufferView.buffer (buffer is mandatory)
					size_t bufferIndex = bufferView.at("buffer").get_ref<json::number_unsigned_t&>();

					// get buffer
					auto& buffer = buffers.at(bufferIndex);

					// make sure we do not run over buffer.byteLength (byteLength is mandatory)
					if(offset + dataSize > buffer.at("byteLength").get_ref<json::number_unsigned_t&>())
						throw GltfError("BufferView range is not completely inside its Buffer.");

					// return pointer to buffer data and data stride
					auto& bufferData = bufferDataList[bufferIndex];
					if(offset + dataSize > bufferData.size())
						throw GltfError("BufferView range is not completely inside data range.");
					return { bufferData.data() + offset, stride };
				};

			// attributes (mesh.primitive.attributes is mandatory)
			auto& attributes = primitive.at("attributes");
			uint32_t vertexSize = 0;
			size_t numVertices = 0;

			struct Attribute {
				uint8_t *sourceData = nullptr;
				unsigned sourceDataStride = 0;
				unsigned dataSize = 0;
				CadPL::AttributeType type = {};
				void (*getterFunc)(const uint8_t* srcPtr, uint8_t* dstPtr) = nullptr;

				operator bool() const noexcept {
					return sourceData;
				}
			};

			Attribute positionAttribute;
			Attribute normalAttribute;
			Attribute tangentAttribute;
			Attribute colorAttribute;
			std::vector<Attribute> attributeSources;

			CadR::BoundingBox primitiveSetBB;

            if (attributes.size() > 16) {
                throw GltfError("Too many attributes.");
            }
            std::array<uint16_t, 16> attribInfo = {};

			for(auto it = attributes.begin(); it != attributes.end(); it++) {
				if(it.key() == "POSITION") {

					// accessor
					json& accessor = accessors.at(it.value().get_ref<json::number_unsigned_t&>());

					// accessor.type is mandatory and it must be VEC3 for position accessor
					if(accessor.at("type").get_ref<json::string_t&>() != "VEC3")
						throw GltfError("Position attribute is not of VEC3 type.");

					// accessor.componentType is mandatory and it must be FLOAT (5126) for position accessor
					if(accessor.at("componentType").get_ref<json::number_unsigned_t&>() != 5126)
						throw GltfError("Position attribute componentType is not float.");

					// accessor.normalized is optional with default value false; it must be false for float componentType
					if(auto it=accessor.find("normalized"); it!=accessor.end())
						if(it->get_ref<json::boolean_t&>() == true)
							throw GltfError("Position attribute normalized flag is true.");

					// update numVertices
					updateNumVertices(accessor, numVertices);

					positionAttribute.type = CadPL::AttributeType::vec3align16;
					positionAttribute.dataSize = 16;
					positionAttribute.getterFunc = getPosFromVec3;
					// vertex size
					vertexSize += positionAttribute.dataSize;

					// position data and stride
					tie(reinterpret_cast<void*&>(positionAttribute.sourceData), positionAttribute.sourceDataStride) =
						getDataPointerAndStride(accessor, bufferViews, buffers, bufferDataList,
						                        numVertices, sizeof(glm::vec3));

					// get min and max
					if(auto it=accessor.find("min"); it!=accessor.end()) {
						json::array_t& a = it->get_ref<json::array_t&>();
						if(a.size() != 3)
							throw GltfError("Accessor.min is not vector of three components.");
						primitiveSetBB.min.x = float(a[0].get<json::number_float_t>());
						primitiveSetBB.min.y = float(a[1].get<json::number_float_t>());
						primitiveSetBB.min.z = float(a[2].get<json::number_float_t>());
					}
					else
						throw GltfError("Accessor.min be defined for POSITION accessor.");
					if(auto it=accessor.find("max"); it!=accessor.end()) {
						json::array_t& a = it->get_ref<json::array_t&>();
						if(a.size() != 3)
							throw GltfError("Accessor.max is not vector of three components.");
						primitiveSetBB.max.x = float(a[0].get<json::number_float_t>());
						primitiveSetBB.max.y = float(a[1].get<json::number_float_t>());
						primitiveSetBB.max.z = float(a[2].get<json::number_float_t>());
					}
					else
						throw GltfError("Accessor.max be defined for POSITION accessor.");

				}
				else if(it.key() == "NORMAL") {

					// accessor
					json& accessor = accessors.at(it.value().get_ref<json::number_unsigned_t&>());

					// accessor.type is mandatory and it must be VEC3 for normal accessor
					if(accessor.at("type").get_ref<json::string_t&>() != "VEC3")
						throw GltfError("Normal attribute is not of VEC3 type.");

					// accessor.componentType is mandatory and it must be FLOAT (5126) for normal accessor
					if(accessor.at("componentType").get_ref<json::number_unsigned_t&>() != 5126)
						throw GltfError("Normal attribute componentType is not float.");

					// accessor.normalized is optional with default value false; it must be false for float componentType
					if(auto it=accessor.find("normalized"); it!=accessor.end())
						if(it->get_ref<json::boolean_t&>() == true)
							throw GltfError("Normal attribute normalized flag is true.");

					// update numVertices
					updateNumVertices(accessor, numVertices);

					normalAttribute.type = CadPL::AttributeType::vec3align16;
					normalAttribute.dataSize = 16;
					normalAttribute.getterFunc = getVec3;
					// vertex size
					vertexSize += normalAttribute.dataSize;

					// normal data and stride
					tie(reinterpret_cast<void*&>(normalAttribute.sourceData), normalAttribute.sourceDataStride) =
						getDataPointerAndStride(accessor, bufferViews, buffers, bufferDataList,
						                        numVertices, sizeof(glm::vec3));

				}
                else if(it.key() == "TANGENT") {
                    // TODO
                }
				else if(it.key() == "COLOR_0") {

					// accessor
					json& accessor = accessors.at(it.value().get_ref<json::number_unsigned_t&>());

					// accessor.type is mandatory and it must be VEC3 or VEC4 for color accessors
					const json::string_t& t = accessor.at("type").get_ref<json::string_t&>();
					if(t != "VEC3" && t != "VEC4")
						throw GltfError("Color attribute is not of VEC3 or VEC4 type.");

					// accessor.componentType is mandatory and it must be FLOAT (5126),
					// UNSIGNED_BYTE (5121) or UNSIGNED_SHORT (5123) for color accessors
					const json::number_unsigned_t ct = accessor.at("componentType").get_ref<json::number_unsigned_t&>();
					if(ct != 5126 && ct != 5121 && ct != 5123)
						throw GltfError("Color attribute componentType is not float, unsigned byte, or unsigned short.");

					// accessor.normalized is optional with default value false; it must be false for float componentType
					if(auto it=accessor.find("normalized"); it!=accessor.end()) {
						if(it->get_ref<json::boolean_t&>() == false) {
							if(ct == 5121 || ct == 5123)
								throw GltfError("Color attribute component type is set to unsigned byte or unsigned short while normalized flag is not true.");
						}
						else
							if(ct == 5126)
								throw GltfError("Color attribute component type is set to float while normalized flag is true.");
					} else
						if(ct == 5121 || ct == 5123)
							throw GltfError("Color attribute component type is set to unsigned byte or unsigned short while normalized flag is not true.");

					// update numVertices
					updateNumVertices(accessor, numVertices);

					colorAttribute.type = CadPL::AttributeType::vec3align16;
					colorAttribute.dataSize = 16;
					// vertex size
					vertexSize += colorAttribute.dataSize;

					// getColorFunc and elementSize
					size_t elementSize;
					if(t == "VEC4") {
                        switch (ct) {
                            case 5126:
                                colorAttribute.getterFunc = getColorFromVec4f;
                                elementSize = 16;
                                break;
                            case 5121:
                                colorAttribute.getterFunc = getColorFromVec4ub;
                                elementSize = 4;
                                break;
                            case 5123:
                                colorAttribute.getterFunc = getColorFromVec4us;
                                elementSize = 8;
                                break;
                        }
                        colorAttribute.type = CadPL::AttributeType::vec4align16;
                    }
					else { // "VEC3"
                        switch (ct) {
                            case 5126:
                                colorAttribute.getterFunc = getColorFromVec3f;
                                elementSize = 12;
                                break;
                            case 5121:
                                colorAttribute.getterFunc = getColorFromVec3ub;
                                elementSize = 3;
                                break;
                            case 5123:
                                colorAttribute.getterFunc = getColorFromVec3us;
                                elementSize = 6;
                                break;
                        }
                        colorAttribute.type = CadPL::AttributeType::vec3align16;
                    }

					// color data and stride
					tie(reinterpret_cast<void*&>(colorAttribute.sourceData), colorAttribute.sourceDataStride) =
						getDataPointerAndStride(accessor, bufferViews, buffers, bufferDataList,
						                        numVertices, elementSize);

				}
				else if(it.key().starts_with("TEXCOORD_")) {

					// accessor
					json& accessor = accessors.at(it.value().get_ref<json::number_unsigned_t&>());

					// accessor.type is mandatory and it must be VEC2 for texCoord accessors
					const json::string_t& t = accessor.at("type").get_ref<json::string_t&>();
					if(t != "VEC2")
						throw GltfError("TexCoord attribute is not of VEC2 type.");

					// accessor.componentType is mandatory and it must be FLOAT (5126),
					// UNSIGNED_BYTE (5121) or UNSIGNED_SHORT (5123) for color accessors
					const json::number_unsigned_t ct = accessor.at("componentType").get_ref<json::number_unsigned_t&>();
					if(ct != 5126 && ct != 5121 && ct != 5123)
						throw GltfError("TexCoord attribute componentType is not float, unsigned byte, or unsigned short.");

					// accessor.normalized is optional with default value false; it must be false for float componentType
					if(auto it=accessor.find("normalized"); it!=accessor.end()) {
						if(it->get_ref<json::boolean_t&>() == false) {
							if(ct == 5121 || ct == 5123)
								throw GltfError("TexCoord attribute component type is set to unsigned byte or unsigned short while normalized flag is not true.");
						}
						else
							if(ct == 5126)
								throw GltfError("TexCoord attribute component type is set to float while normalized flag is true.");
					} else
						if(ct == 5121 || ct == 5123)
							throw GltfError("TexCoord attribute component type is set to unsigned byte or unsigned short while normalized flag is not true.");

					// update numVertices
					updateNumVertices(accessor, numVertices);

					auto &source = attributeSources.emplace_back();

					source.type = CadPL::AttributeType::vec2align8;
					source.dataSize = 8;
					// vertex size
					vertexSize += source.dataSize;

					// get func and elementSize
					size_t elementSize;
					switch(ct) {
					case 5126: source.getterFunc = getVec2FromVec2f;  elementSize = 8; break;
					case 5121: source.getterFunc = getVec2FromVec2ub; elementSize = 2; break;
					case 5123: source.getterFunc = getVec2FromVec2us; elementSize = 4; break;
					}

					// texCoord data and stride
					tie(reinterpret_cast<void*&>(source.sourceData), source.sourceDataStride) =
						getDataPointerAndStride(accessor, bufferViews, buffers, bufferDataList,
						                        numVertices, elementSize);
				}
				else
					throw GltfError("Unsupported functionality: " + it.key() + " attribute.");
			}

			// align vertexSize to multiples of 16 to avoid issues
			const auto vertexPadding = vertexSize & 15;
			vertexSize = (vertexSize + 15) & ~15;

			// fill attribute information
			{
				uint16_t offset = 0;
				const auto fillAttribute = [&](size_t index, const Attribute &attribute) {
					if (attribute) {
						attribInfo[index] = offset | (static_cast<uint32_t>(attribute.type) << 8);
						offset += attribute.dataSize;
					}
				};
				fillAttribute(0, positionAttribute);
				fillAttribute(1, normalAttribute);
				fillAttribute(3, colorAttribute);
				size_t i = 3;
				for (auto &attribute : attributeSources) {
					fillAttribute(++i, attribute);
				}
			}

			// indices
			// (they are optional)
			size_t numIndices;
			void* indexData;
			unsigned indexComponentType;
			if(auto indicesIt=primitive.find("indices"); indicesIt!=primitive.end()) {

				// accessor
				json& accessor = accessors.at(indicesIt.value().get_ref<json::number_unsigned_t&>());

				// accessor.type is mandatory and it must be SCALAR for index accessors
				const json::string_t& t = accessor.at("type").get_ref<json::string_t&>();
				if(t != "SCALAR")
					throw GltfError("Indices are not of SCALAR type.");

				// accessor.componentType is mandatory and it must be UNSIGNED_INT (5125) for index accessors;
				// unsigned short and unsigned byte component types seems not allowed by the spec
				// but they are used in Khronos sample models, for example Box.gltf
				// (https://github.com/KhronosGroup/glTF-Sample-Models/blob/main/2.0/Box/glTF/Box.gltf)
				indexComponentType = unsigned(accessor.at("componentType").get_ref<json::number_unsigned_t&>());
				if(indexComponentType != 5125 && indexComponentType != 5123 && indexComponentType != 5121)
					throw GltfError("Index componentType is not unsigned int, unsigned short or unsigned byte.");

				// accessor.normalized is optional and must be false for index accessor
				if(auto it=accessor.find("normalized"); it!=accessor.end())
					if(it->get_ref<json::boolean_t&>() == true)
						throw GltfError("Indices cannot have normalized flag set to true.");

				// get index count (accessor.count is mandatory and >=1)
				numIndices = accessor.at("count").get_ref<json::number_unsigned_t&>();
				if(numIndices == 0)
					throw GltfError("Accessor's count member must be greater than zero.");

				// index data
				size_t elementSize;
				switch(indexComponentType) {
				case 5125: elementSize = sizeof(uint32_t); break;
				case 5123: elementSize = sizeof(uint16_t); break;
				case 5121: elementSize = sizeof(uint8_t); break;
				}
				size_t tmp;
				tie(indexData, tmp) =
					getDataPointerAndStride(accessor, bufferViews, buffers, bufferDataList,
					                        numIndices, elementSize);
			}
			else {
				numIndices = numVertices;
				indexData = nullptr;
			}

			// ignore empty primitives
			if(indexData == nullptr && numVertices == 0)
				continue;

			// create Geometry
			std::cout << "Creating geometry of " << mesh.value("name", "{}") << "\n";
			CadR::Geometry& g = geometryList.emplace_back(renderer);

			// update mesh bounds
			meshBB.extendBy(primitiveSetBB);

			// prepare for computing primitiveSet bounding sphere
			CadR::BoundingSphere primitiveSetBS{
				.center = positionAttribute ? primitiveSetBB.getCenter() : glm::vec3(0.f, 0.f, 0.f),
				.radius = 0.f,  // actually, radius^2 is stored here in the following loop as an performance optimization
			};

			// set vertex data
			CadR::StagingData sd = g.createVertexStagingData(numVertices * vertexSize);
			uint8_t* p = sd.data<uint8_t>();
			const auto copyAttribute = [&](Attribute &attrib) {
				attrib.getterFunc(attrib.sourceData, p);
				attrib.sourceData += attrib.sourceDataStride;
				p += attrib.dataSize;
			};

			for(size_t i=0; i<numVertices; i++) {
				if(positionAttribute) {
					glm::vec3 pos = *reinterpret_cast<glm::vec3*>(positionAttribute.sourceData);
					copyAttribute(positionAttribute);
					// update bounding sphere
					// (square of radius is stored in primitiveBS.radius)
					primitiveSetBS.extendRadiusByPointUsingRadius2(pos);
				}
				if(normalAttribute) {
					copyAttribute(normalAttribute);
				}
				if(colorAttribute) {
					copyAttribute(colorAttribute);
				}
				for (auto &src : attributeSources) {
					copyAttribute(src);
				}
				p += vertexPadding;
			}

			// mesh.primitive.mode is optional with default value 4 (TRIANGLES)
			unsigned mode = unsigned(primitive.value<json::number_unsigned_t>("mode", 4));

			// set index data
			if(mode == 4) {  // TRIANGLES
				if(numIndices < 3)
					throw GltfError("Invalid number of indices for TRIANGLES.");
				goto copyIndices;
			}
			else if(mode == 1) {  // LINES
				if(numIndices < 2)
					throw GltfError("Invalid number of indices for LINES.");
				goto copyIndices;
			}
			else if(mode == 0) {  // POINTS
				if(numIndices < 1)
					throw GltfError("Invalid number of indices for POINTS.");

				// POINTS, LINES, TRIANGLES - copy indices directly,
				// while converting uint8_t and uint16_t indices to uint32_t
			copyIndices:
				size_t indexDataSize = numIndices * sizeof(uint32_t);
				sd = g.createIndexStagingData(indexDataSize);
				uint32_t* pi = sd.data<uint32_t>();
				if(indexData)
					switch(indexComponentType) {
					case 5125: memcpy(pi, indexData, indexDataSize); break;
					case 5123: {
						for(size_t i=0; i<numIndices; i++)
							pi[i] = reinterpret_cast<uint16_t*>(indexData)[i];
						break;
					}
					case 5121: {
						for(size_t i=0; i<numIndices; i++)
							pi[i] = reinterpret_cast<uint8_t*>(indexData)[i];
						break;
					}
					}
				else {
					if(numIndices >= size_t((~uint32_t(0))-1)) // value 0xffffffff is forbidden, thus (~0)-1
						throw GltfError("Too large primitive. Index out of 32-bit integer range.");
					for(uint32_t i=0; i<uint32_t(numIndices); i++)
						pi[i] = i;
				}
			}
			else if(mode == 5) {

				// TRIANGLE_STRIP - convert strip indices to indices of separate triangles
				// while considering even and odd triangle ordering
				if(numIndices < 3)
					throw GltfError("Invalid number of indices for TRIANGLE_STRIP.");
				numIndices = (numIndices-2) * 3;
				sd = g.createIndexStagingData(numIndices * sizeof(uint32_t));
				uint32_t* stgIndices = sd.data<uint32_t>();
				if(indexData) {

					// create new indices
					auto createTriangleStripIndices =
						[]<typename T>(uint32_t* dst, void* srcPtr, size_t numIndices) {

							T* src = reinterpret_cast<T*>(srcPtr);
							uint32_t* dstEnd = dst + numIndices;
							uint32_t v1 = *src;
							src++;
							uint32_t v2 = *src;
							src++;
							uint32_t v3 = *src;
							src++;
							while(true) {

								// odd triangle
								*dst = v1;
								dst++;
								*dst = v2;
								dst++;
								*dst = v3;
								dst++;
								if(dst == dstEnd)
									break;
								v1 = v2;
								v2 = v3;
								v3 = *src;
								src++;

								// even triangle
								*dst = v2;
								dst++;
								*dst = v1;
								dst++;
								*dst = v3;
								dst++;
								if(dst == dstEnd)
									break;
								v1 = v2;
								v2 = v3;
								v3 = *src;
								src++;
							}
						};
					switch(indexComponentType) {
					case 5125: createTriangleStripIndices.operator()<uint32_t>(stgIndices, indexData, numIndices); break;
					case 5123: createTriangleStripIndices.operator()<uint16_t>(stgIndices, indexData, numIndices); break;
					case 5121: createTriangleStripIndices.operator()<uint8_t >(stgIndices, indexData, numIndices); break;
					}
				}
				else {

					// generate indices
					if(numIndices >= size_t((~uint32_t(0))-1)) // value 0xffffffff is forbidden, thus (~0)-1
						throw GltfError("Too large primitive. Index out of 32-bit integer range.");

					uint32_t i = 0;
					uint32_t v1 = i;
					i++;
					uint32_t v2 = i;
					i++;
					uint32_t v3 = i;
					i++;
					uint32_t* e = stgIndices + numIndices;
					while(true) {

						// odd triangle
						*stgIndices = v1;
						stgIndices++;
						*stgIndices = v2;
						stgIndices++;
						*stgIndices = v3;
						stgIndices++;
						if(stgIndices == e)
							break;
						v1 = v2;
						v2 = v3;
						v3 = i;
						i++;

						// even triangle
						*stgIndices = v2;
						stgIndices++;
						*stgIndices = v1;
						stgIndices++;
						*stgIndices = v3;
						stgIndices++;
						if(stgIndices == e)
							break;
						v1 = v2;
						v2 = v3;
						v3 = i;
						i++;
					}
				}
			}
			else if(mode == 6) {

				// TRIANGLE_FAN
				if(numIndices < 3)
					throw GltfError("Invalid number of indices for TRIANGLE_FAN.");
				numIndices = (numIndices-2) * 3;
				sd = g.createIndexStagingData(numIndices * sizeof(uint32_t));
				uint32_t* stgIndices = sd.data<uint32_t>();
				if(indexData) {

					// create new indices
					auto createTriangleStripIndices =
						[]<typename T>(uint32_t* dst, void* srcPtr, size_t numIndices) {

							T* src = reinterpret_cast<T*>(srcPtr);
							uint32_t* dstEnd = dst + numIndices;
							uint32_t v1 = *src;
							src++;
							uint32_t v2 = *src;
							src++;
							uint32_t v3 = *src;
							src++;
							while(true) {
								*dst = v1;
								dst++;
								*dst = v2;
								dst++;
								*dst = v3;
								dst++;
								if(dst == dstEnd)
									break;
								v2 = v3;
								v3 = *src;
								src++;
							}
						};
					switch(indexComponentType) {
					case 5125: createTriangleStripIndices.operator()<uint32_t>(stgIndices, indexData, numIndices); break;
					case 5123: createTriangleStripIndices.operator()<uint16_t>(stgIndices, indexData, numIndices); break;
					case 5121: createTriangleStripIndices.operator()<uint8_t >(stgIndices, indexData, numIndices); break;
					}
				}
				else {

					// generate indices
					if(numIndices >= size_t((~uint32_t(0))-1)) // value 0xffffffff is forbidden, thus (~0)-1
						throw GltfError("Too large primitive. Index out of 32-bit integer range.");

					uint32_t i = 0;
					uint32_t v1 = i;
					i++;
					uint32_t v2 = i;
					i++;
					uint32_t v3 = i;
					i++;
					uint32_t* e = stgIndices + numIndices;
					while(true) {
						*stgIndices = v1;
						stgIndices++;
						*stgIndices = v2;
						stgIndices++;
						*stgIndices = v3;
						stgIndices++;
						if(stgIndices == e)
							break;
						v2 = v3;
						v3 = i;
						i++;
					}
				}
			}
			else if(mode == 3) {

				// LINE_STRIP
				if(numIndices < 2)
					throw GltfError("Invalid number of indices for LINE_STRIP.");
				numIndices = (numIndices-1) * 2;
				sd = g.createIndexStagingData(numIndices * sizeof(uint32_t));
				uint32_t* stgIndices = sd.data<uint32_t>();
				if(indexData) {

					// create new indices
					auto createLineStripIndices =
						[]<typename T>(uint32_t* dst, void* srcPtr, size_t numIndices) {
							T* src = reinterpret_cast<T*>(srcPtr);
							*dst = *src;
							uint32_t* dstEnd = dst + (numIndices-1);
							dst++; src++;
							while(dst < dstEnd) {
								*dst = *src;
								dst++;
								*dst = *src;
								dst++; src++;
							}
							*dst = *src;
						};
					switch(indexComponentType) {
					case 5125: createLineStripIndices.operator()<uint32_t>(stgIndices, indexData, numIndices); break;
					case 5123: createLineStripIndices.operator()<uint16_t>(stgIndices, indexData, numIndices); break;
					case 5121: createLineStripIndices.operator()<uint8_t >(stgIndices, indexData, numIndices); break;
					}
				}
				else {

					// generate indices
					if(numIndices >= size_t((~uint32_t(0))-1)) // value 0xffffffff is forbidden, thus (~0)-1
						throw GltfError("Too large primitive. Index out of 32-bit integer range.");
					uint32_t i = 0;
					*stgIndices = i;
					uint32_t* e = stgIndices + (numIndices-1);
					stgIndices++;
					i++;
					while(stgIndices < e) {
						*stgIndices = i;
						stgIndices++;
						*stgIndices = i;
						stgIndices++;
						i++;
					}
					*stgIndices = i;
				}
			}
			else if(mode == 2) {

				// LINE_LOOP
				if(numIndices < 2)
					throw GltfError("Invalid number of indices for LINE_LOOP.");
				numIndices = numIndices * 2;
				sd = g.createIndexStagingData(numIndices * sizeof(uint32_t));
				uint32_t* stgIndices = sd.data<uint32_t>();
				if(indexData) {

					// create new indices
					auto createLineLoopIndices =
						[]<typename T>(uint32_t* dst, void* srcPtr, size_t numIndices) {
							T* src = reinterpret_cast<T*>(srcPtr);
							uint32_t firstValue = *src;
							*dst = *src;
							uint32_t* dstEnd = dst + (numIndices-1);
							dst++; src++;
							while(dst < dstEnd) {
								*dst = *src;
								dst++;
								*dst = *src;
								dst++; src++;
							}
							*dst = firstValue;
						};
					switch(indexComponentType) {
					case 5125: createLineLoopIndices.operator()<uint32_t>(stgIndices, indexData, numIndices); break;
					case 5123: createLineLoopIndices.operator()<uint16_t>(stgIndices, indexData, numIndices); break;
					case 5121: createLineLoopIndices.operator()<uint8_t >(stgIndices, indexData, numIndices); break;
					}
				}
				else {

					// generate indices
					if(numIndices >= size_t((~uint32_t(0))-1)) // value 0xffffffff is forbidden, thus (~0)-1
						throw GltfError("Too large primitive. Index out of 32-bit integer range.");
					uint32_t i = 0;
					*stgIndices = i;
					uint32_t* e = stgIndices + (numIndices-1);
					stgIndices++;
					i++;
					while(stgIndices < e) {
						*stgIndices = i;
						stgIndices++;
						*stgIndices = i;
						stgIndices++;
						i++;
					}
					*stgIndices = 0;
				}
			}
			else
				throw GltfError("Invalid value for mesh.primitive.mode.");

			// set primitiveSet data
			struct PrimitiveSetGpuData {
				uint32_t count;
				uint32_t first;
			};
			sd = g.createPrimitiveSetStagingData(sizeof(PrimitiveSetGpuData));
			PrimitiveSetGpuData* ps = sd.data<PrimitiveSetGpuData>();
			ps->count = uint32_t(numIndices);
			ps->first = 0;

			// primitiveSet bounding sphere list
			// (convert square of radius stored in primitiveBS.radius back to radius)
			primitiveSetBS.radius = sqrt(primitiveSetBS.radius);
			primitiveSetBSList.emplace_back(primitiveSetBS);

			// material
			auto materialIt = primitive.find("material");
			size_t materialIndex =
				(materialIt != primitive.end())
					? materialIt->get_ref<json::number_unsigned_t&>()
					: ~size_t(0);
			StateSetMaterialData& ssMaterialData =
				(materialIt != primitive.end())
				? stateSetMaterialDataList.at(materialIndex)
				: defaultStateSetMaterialData;

			vk::PrimitiveTopology primitiveTopology;
			switch(mode) {
				case 0:  // POINTS
					primitiveTopology = vk::PrimitiveTopology::ePointList;
					break;
				case 1:  // LINES
				case 2:  // LINE_LOOP
				case 3:  // LINE_STRIP
					primitiveTopology = vk::PrimitiveTopology::eLineList;
					break;
				case 4:  // TRIANGLES
				case 5:  // TRIANGLE_STRIP
				case 6:  // TRIANGLE_FAN
					primitiveTopology = vk::PrimitiveTopology::eTriangleList;
					break;
				default:
					throw GltfError("Invalid value for mesh.primitive.mode.");
			}

			uint32_t materialSettings = ssMaterialData.textureOffset;
			materialSettings |= 0x0001;
			if (ssMaterialData.doubleSided) materialSettings |= 0x100;  // two sided lighting
			if (colorAttribute) materialSettings |=  0x0200;  // use color attribute for ambient and diffuse

			uint32_t attribSetup = (normalAttribute ? 0 : 1) |  // generateFlatNormals
									vertexSize;

			CadR::StateSet* ss;
			if (true) {
				// get StateSet
				PipelineStateSet& pipelineStateSet = getPipelineStateSet(
					optimizeFlags,
					primitiveTopology,
					ssMaterialData.doubleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack,
					vk::FrontFace::eCounterClockwise,
					materialSettings,
					attribSetup,
					attribInfo,
					ssMaterialData.textureSetup
				);
				ss = &pipelineStateSet.stateSet;
			}
			else {
				// pipeline
				CadPL::ShaderState shaderState{
					.idBuffer = false,
					.primitiveTopology = primitiveTopology,
					.projectionHandling =
						CadPL::ShaderState::ProjectionHandling::PerspectivePushAndSpecializationConstants,
					.attribAccessInfo = attribInfo,
					.attribSetup = attribSetup,
					.materialSetup = materialSettings,
					.lightSetup = {},  // no lights; switches between directional light, point light and spotlight
					.numLights = {},
					.textureSetup = {},  // no textures
					.numTextures = {},
					.optimizeFlags = CadPL::ShaderState::OptimizeNone,
				};
				CadPL::PipelineState pipelineState{
					.viewportAndScissorHandling = CadPL::PipelineState::ViewportAndScissorHandling::SetFunction,
					.projectionIndex = 0,
					.viewportIndex = 0,
					.scissorIndex = 0,
					.viewport = {},
					.scissor = {},
					.cullMode =
						(ssMaterialData.doubleSided)
							? vk::CullModeFlagBits::eNone   // eNone - nothing is discarded
							: vk::CullModeFlagBits::eBack,  // eBack - back-facing triangles are discarded, eFront - front-facing triangles are discarded
					.frontFace = vk::FrontFace::eCounterClockwise,
					.depthBiasDynamicState = false,
					.depthBiasEnable = false,
					.depthBiasConstantFactor = 0.f,
					.depthBiasClamp = 0.f,
					.depthBiasSlopeFactor = 0.f,
					.lineWidthDynamicState = false,
					.lineWidth = 1.f,
					.rasterizationSamples = vk::SampleCountFlagBits::e1,
					.sampleShadingEnable = false,
					.minSampleShading = 0.f,
					.depthTestEnable = true,
					.depthWriteEnable = true,
					.blendState = { { .blendEnable = false } },
					.renderPass = renderPass,
					.subpass = 0,
				};
				ss = &pipelineSceneGraph.getOrCreateStateSet(shaderState, pipelineState);
			}

			std::cout << "stateSet: " << ss << "\n";
			// drawable
			drawableList.emplace_back(
				g,  // geometry
				0,  // primitiveSetOffset
				matrixLists[meshIndex],  // matrixList
				(materialIndex==~size_t(0))  // drawableData
					? defaultMaterial
					: materialList[materialIndex],
				*ss  // stateSet
			);

			// mesh bounding sphere
			CadR::BoundingSphere meshBS{
				.center = meshBB.getCenter(),
				.radius = 0.f,
			};
			for(size_t i=0,c=primitiveSetBSList.size(); i<c; i++)
				meshBS.extendRadiusBy(primitiveSetBSList[i]);

			// bounding box of all instances of particular mesh
			const vector<glm::mat4>& matrices = meshMatrixList[meshIndex];
			CadR::BoundingBox instancesBB =
				CadR::BoundingBox::createByCenterAndHalfExtents(
					glm::mat3(matrices[0]) * meshBS.center + glm::vec3(matrices[0][3]),  // center
					glm::mat3(matrices[0]) * glm::vec3(meshBS.radius)  // halfExtents
				);
			for(size_t instanceIndex=1, instanceCount=matrices.size();
				instanceIndex<instanceCount; instanceIndex++)
			{
				const glm::mat4& m = matrices[instanceIndex];
				instancesBB.extendBy(
					CadR::BoundingBox::createByCenterAndHalfExtents(
						glm::mat3(m) * meshBS.center + glm::vec3(m[3]),  // center
						glm::mat3(m) * glm::vec3(meshBS.radius)  // radius
					)
				);
			}

			// bounding sphere of all instances of particular mesh
			CadR::BoundingSphere instancesBS{
				.center = instancesBB.getCenter(),
				.radius = 0.f,
			};
			for(size_t instanceIndex=0, instanceCount=matrices.size();
				instanceIndex<instanceCount; instanceIndex++)
			{
				instancesBS.extendRadiusBy(matrices[instanceIndex] * meshBS);
			}
			meshBoundingSphereList[meshIndex] = instancesBS;

		}
	}

	// scene bounding box
	CadR::BoundingBox sceneBB = meshBoundingSphereList[0].getBoundingBox();
	for(size_t i=1, c=meshBoundingSphereList.size(); i<c; i++)
		sceneBB.extendBy(meshBoundingSphereList[i].getBoundingBox());

	// scene bounding sphere
	sceneBoundingSphere.center = sceneBB.getCenter();
	sceneBoundingSphere.radius =
		sqrt(glm::distance2(meshBoundingSphereList[0].center, sceneBoundingSphere.center)) +
		meshBoundingSphereList[0].radius;
	for(size_t i=1, c=meshBoundingSphereList.size(); i<c; i++)
		sceneBoundingSphere.extendRadiusBy(meshBoundingSphereList[i]);
	sceneBoundingSphere.radius *= 1.001f;  // increase radius to accommodate for all floating computations imprecisions

	// initial camera distance
	float fovy2Clamped = glm::clamp(fovy / 2.f, 1.f / 180.f * glm::pi<float>(), 90.f / 180.f * glm::pi<float>());
	cameraDistance = sceneBoundingSphere.radius / sin(fovy2Clamped);

	setCamera(sceneBoundingSphere.center - glm::vec3(sceneBoundingSphere.radius, sceneBoundingSphere.radius, 0), sceneBoundingSphere.center, glm::vec3(0.f, 1.f, 0.f));

#if 0  // show scene bounding sphere
	auto createBoundingSphereVisualization =
		[](const CadR::BoundingSphere bs, App& app) {

			// vertex data
			// (bounding sphere in the form of axis cross)
			CadR::Geometry& g = app.geometryDB.emplace_back(app.renderer);
			CadR::StagingData sd = g.createVertexStagingData(6 * sizeof(glm::vec4));
			glm::vec4* pos = sd.data<glm::vec4>();
			pos[0] = glm::vec4(bs.center.x-bs.radius, bs.center.y, bs.center.z, 1.f);
			pos[1] = glm::vec4(bs.center.x+bs.radius, bs.center.y, bs.center.z, 1.f);
			pos[2] = glm::vec4(bs.center.x, bs.center.y-bs.radius, bs.center.z, 1.f);
			pos[3] = glm::vec4(bs.center.x, bs.center.y+bs.radius, bs.center.z, 1.f);
			pos[4] = glm::vec4(bs.center.x, bs.center.y, bs.center.z-bs.radius, 1.f);
			pos[5] = glm::vec4(bs.center.x, bs.center.y, bs.center.z+bs.radius, 1.f);

			// index data
			sd = g.createIndexStagingData(6 * sizeof(uint32_t));
			uint32_t* indices = sd.data<uint32_t>();
			for(uint32_t i=0; i<6; i++)
				indices[i] = i;

			// primitive set
			struct PrimitiveSetGpuData {
				uint32_t count;
				uint32_t first;
			};
			sd = g.createPrimitiveSetStagingData(sizeof(PrimitiveSetGpuData));
			PrimitiveSetGpuData* ps = sd.data<PrimitiveSetGpuData>();
			ps->count = 6;
			ps->first = 0;

			// state set
			unsigned pipelineIndex =
				PipelineLibrary::getLinePipelineIndex(
					false,  // phong
					false,  // texturing
					false   // perVertexColor
				);
			CadR::StateSet& ss = app.stateSetDB[pipelineIndex];

			// drawable
			app.drawableDB.emplace_back(
				g,  // geometry
				0,  // primitiveSetOffset
				sd,  // shaderStagingData
				64+64,  // shaderDataSize
				1,  // numInstances
				ss);  // stateSet

			// material
			struct MaterialData {
				glm::vec3 ambient;  // offset 0
				uint32_t type;  // offset 12
				glm::vec4 diffuseAndAlpha;  // offset 16
				glm::vec3 specular;  // offset 32
				float shininess;  // offset 44
				glm::vec3 emission;  // offset 48
				float pointSize;  // offset 60
			};
			MaterialData* m = sd.data<MaterialData>();
			m->ambient = glm::vec3(1.f, 1.f, 1.f);
			m->type = 0;
			m->diffuseAndAlpha = glm::vec4(1.f, 1.f, 1.f, 1.f);
			m->specular = glm::vec3(0.f, 0.f, 0.f);
			m->shininess = 0.f;
			m->emission = glm::vec3(0.f, 0.f, 0.f);
			m->pointSize = 0.f;

			// transformation matrices
			glm::mat4* matrices = sd.data<glm::mat4>() + 1;
			matrices[0] = glm::mat4(1.f);
		};

#if 0  // show scene bounding sphere
	createBoundingSphereVisualization(sceneBoundingSphere, *this);
#endif

#if 0  // show bounding sphere of each mesh
	for(size_t i=0, c=meshBoundingSphereList.size(); i<c; i++)
		createBoundingSphereVisualization(meshBoundingSphereList[i], *this);
#endif
#endif

	// upload all staging buffers
	renderer.executeCopyOperations();
}


void App::resize(VulkanWindow& window,
	const vk::SurfaceCapabilitiesKHR& surfaceCapabilities, vk::Extent2D newSurfaceExtent)
{
	// clear resources
	for(auto v : swapchainImageViews)  device.destroy(v);
	swapchainImageViews.clear();
	device.destroy(depthImage);
	device.free(depthImageMemory);
	device.destroy(depthImageView);
	for(auto f : framebuffers)  device.destroy(f);
	framebuffers.clear();

	// print info
	cout << "Recreating swapchain (extent: " << newSurfaceExtent.width << "x" << newSurfaceExtent.height
	     << ", extent by surfaceCapabilities: " << surfaceCapabilities.currentExtent.width << "x"
	     << surfaceCapabilities.currentExtent.height << ", minImageCount: " << surfaceCapabilities.minImageCount
	     << ", maxImageCount: " << surfaceCapabilities.maxImageCount << ")" << endl;

	// create new swapchain
	constexpr const uint32_t requestedImageCount = 2;
	vk::UniqueHandle<vk::SwapchainKHR, CadR::VulkanDevice> newSwapchain =
		device.createSwapchainKHRUnique(
			vk::SwapchainCreateInfoKHR(
				vk::SwapchainCreateFlagsKHR(),  // flags
				window.surface(),               // surface
				surfaceCapabilities.maxImageCount==0  // minImageCount
					? max(requestedImageCount, surfaceCapabilities.minImageCount)
					: clamp(requestedImageCount, surfaceCapabilities.minImageCount, surfaceCapabilities.maxImageCount),
				surfaceFormat.format,           // imageFormat
				surfaceFormat.colorSpace,       // imageColorSpace
				newSurfaceExtent,               // imageExtent
				1,                              // imageArrayLayers
				vk::ImageUsageFlagBits::eColorAttachment,  // imageUsage
				(graphicsQueueFamily==presentationQueueFamily) ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent, // imageSharingMode
				uint32_t(2),  // queueFamilyIndexCount
				array<uint32_t, 2>{graphicsQueueFamily, presentationQueueFamily}.data(),  // pQueueFamilyIndices
				surfaceCapabilities.currentTransform,    // preTransform
				vk::CompositeAlphaFlagBitsKHR::eOpaque,  // compositeAlpha
				vk::PresentModeKHR::eImmediate,  // presentMode
				VK_TRUE,  // clipped
				swapchain  // oldSwapchain
			)
		);
	device.destroy(swapchain);
	swapchain = newSwapchain.release();

	// swapchain images and image views
	vector<vk::Image> swapchainImages = device.getSwapchainImagesKHR(swapchain);
	swapchainImageViews.reserve(swapchainImages.size());
	for(vk::Image image : swapchainImages)
		swapchainImageViews.emplace_back(
			device.createImageView(
				vk::ImageViewCreateInfo(
					vk::ImageViewCreateFlags(),  // flags
					image,                       // image
					vk::ImageViewType::e2D,      // viewType
					surfaceFormat.format,        // format
					vk::ComponentMapping(),      // components
					vk::ImageSubresourceRange(   // subresourceRange
						vk::ImageAspectFlagBits::eColor,  // aspectMask
						0,  // baseMipLevel
						1,  // levelCount
						0,  // baseArrayLayer
						1   // layerCount
					)
				)
			)
		);

	// depth image
	depthImage =
		device.createImage(
			vk::ImageCreateInfo(
				vk::ImageCreateFlags(),  // flags
				vk::ImageType::e2D,      // imageType
				depthFormat,             // format
				vk::Extent3D(newSurfaceExtent.width, newSurfaceExtent.height, 1),  // extent
				1,                       // mipLevels
				1,                       // arrayLayers
				vk::SampleCountFlagBits::e1,  // samples
				vk::ImageTiling::eOptimal,    // tiling
				vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,  // usage
				vk::SharingMode::eExclusive,  // sharingMode
				0,                            // queueFamilyIndexCount
				nullptr,                      // pQueueFamilyIndices
				vk::ImageLayout::eUndefined   // initialLayout
			)
		);

	// memory for images
	vk::PhysicalDeviceMemoryProperties memoryProperties = vulkanInstance.getPhysicalDeviceMemoryProperties(physicalDevice);
	auto allocateMemory =
		[](CadR::VulkanDevice& device, vk::Image image, vk::MemoryPropertyFlags requiredFlags,
		   const vk::PhysicalDeviceMemoryProperties& memoryProperties) -> vk::DeviceMemory
		{
			vk::MemoryRequirements memoryRequirements = device.getImageMemoryRequirements(image);
			for(uint32_t i=0; i<memoryProperties.memoryTypeCount; i++)
				if(memoryRequirements.memoryTypeBits & (1<<i))
					if((memoryProperties.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags)
						return
							device.allocateMemory(
								vk::MemoryAllocateInfo(
									memoryRequirements.size,  // allocationSize
									i                         // memoryTypeIndex
								)
							);
			throw std::runtime_error("No suitable memory type found for the image.");
		};
	depthImageMemory = allocateMemory(device, depthImage, vk::MemoryPropertyFlagBits::eDeviceLocal, memoryProperties);
	device.bindImageMemory(
		depthImage,  // image
		depthImageMemory,  // memory
		0  // memoryOffset
	);

	// image views
	depthImageView =
		device.createImageView(
			vk::ImageViewCreateInfo(
				vk::ImageViewCreateFlags(),  // flags
				depthImage,                  // image
				vk::ImageViewType::e2D,      // viewType
				depthFormat,                 // format
				vk::ComponentMapping(),      // components
				vk::ImageSubresourceRange(   // subresourceRange
					vk::ImageAspectFlagBits::eDepth,  // aspectMask
					0,  // baseMipLevel
					1,  // levelCount
					0,  // baseArrayLayer
					1   // layerCount
				)
			)
		);

	// framebuffers
	framebuffers.reserve(swapchainImages.size());
	for(size_t i=0, c=swapchainImages.size(); i<c; i++)
		framebuffers.emplace_back(
			device.createFramebuffer(
				vk::FramebufferCreateInfo(
					vk::FramebufferCreateFlags(),  // flags
					renderPass,  // renderPass
					2,  // attachmentCount
					array{  // pAttachments
						swapchainImageViews[i],
						depthImageView,
					}.data(),
					newSurfaceExtent.width,  // width
					newSurfaceExtent.height,  // height
					1  // layers
				)
			)
		);

	// rendering finished semaphores
	if(renderingFinishedSemaphores.size() != swapchainImages.size())
	{
		for(auto s : renderingFinishedSemaphores)  device.destroy(s);
		renderingFinishedSemaphores.clear();
		renderingFinishedSemaphores.reserve(swapchainImages.size());
		vk::SemaphoreCreateInfo semaphoreCreateInfo{
			vk::SemaphoreCreateFlags()  // flags
		};
		for(size_t i=0,c=swapchainImages.size(); i<c; i++)
			renderingFinishedSemaphores.emplace_back(
				device.createSemaphore(semaphoreCreateInfo));
	}

	// perspective matrix given by FOV (Field Of View)
	// FOV is given in vertical direction in radians
	// The function perspectiveLH_ZO() produces exactly the matrix we need in Vulkan for right-handed, zero-to-one coordinate system.
	// The coordinate system will have +x in the right direction, +y in down direction (as Vulkan does it), and +z forward into the scene.
	// ZO - Zero to One is output depth range,
	// RH - Right Hand coordinate system, +Y is down, +Z is towards camera
	// LH - LeftHand coordinate system, +Y is down, +Z points into the scene
	constexpr float zNear = 0.5f;
	constexpr float zFar = 100.f;
	glm::mat4 projectionMatrix = glm::perspectiveLH_ZO(fovy, float(newSurfaceExtent.width)/newSurfaceExtent.height, zNear, zFar);

	// resize pipelines
	pipelineSceneGraph.setProjectionViewportAndScissor(
		projectionMatrix,
		vk::Viewport(0.f, 0.f,
					 float(newSurfaceExtent.width), float(newSurfaceExtent.height),
					 0.f, 1.f),
		vk::Rect2D(vk::Offset2D(0, 0), newSurfaceExtent)
	);

	pipelineLibrary->setProjectionViewportAndScissor(
		projectionMatrix,
		vk::Viewport(0.f, 0.f,
					 float(newSurfaceExtent.width), float(newSurfaceExtent.height),
					 0.f, 1.f),
		vk::Rect2D(vk::Offset2D(0, 0), newSurfaceExtent)
	);

	// recreate pipelines
	setPipelines();
}


void App::frame(VulkanWindow&)
{
	// wait for previous frame rendering work
	// if still not finished
	// (we might start copy operations before, but we need to exclude TableHandles that must stay intact
	// until the rendering is finished)
	vk::Result r =
		device.waitForFences(
			renderingFinishedFence,  // fences
			VK_TRUE,  // waitAll
			uint64_t(3e9)  // timeout
		);
	if(r != vk::Result::eSuccess) {
		if(r == vk::Result::eTimeout)
			throw runtime_error("GPU timeout. Task is probably hanging on GPU.");
		throw runtime_error("Vulkan error: vkWaitForFences failed with error " + to_string(r) + ".");
	}
	device.resetFences(renderingFinishedFence);

    // collect previous frame info
    auto info = renderer.getFrameInfo();
    double cpuTime = double(info.cpuEndFrame - info.cpuBeginFrame) * renderer.cpuTimestampPeriod();
    double gpuTime = double(info.gpuEndExecution - info.gpuBeginExecution) * renderer.gpuTimestampPeriod();
    cpuTimeCounter.update(cpuTime);
    gpuTimeCounter.update(gpuTime);

	// _sceneDataAllocation
	uint32_t sceneDataSize = uint32_t(sizeof(SceneGpuData));
	CadR::StagingData sceneStagingData = sceneDataAllocation.alloc(sceneDataSize);
	SceneGpuData* sceneData = sceneStagingData.data<SceneGpuData>();

	float distance;
	if (newCamera) {
		sceneData->viewMatrix = glm::lookAtLH(cameraEye, cameraEye + cameraDirection, cameraUp);
		distance = glm::distance(sceneBoundingSphere.center, cameraEye);
	}
	else {
		sceneData->viewMatrix = glm::lookAtLH(
			sceneBoundingSphere.center + glm::vec3(  // eye
				+cameraDistance*sin(-cameraHeading)*cos(cameraElevation),  // x
				-cameraDistance*sin(cameraElevation),  // y
				-cameraDistance*cos(cameraHeading)*cos(cameraElevation)  // z
			),
			sceneBoundingSphere.center,  // center
			glm::vec3(0.f,1.f,0.f)  // up
		);
		distance = cameraDistance;
	}

	float zFar = fabs(distance) + sceneBoundingSphere.radius;
	float zNear = fabs(distance) - sceneBoundingSphere.radius;
	float minZNear = zFar / maxZNearZFarRatio;
	if(zNear < minZNear)
		zNear = minZNear;
	glm::mat4 projectionMatrix = glm::perspectiveLH_ZO(fovy, float(window.surfaceExtent().width)/window.surfaceExtent().height, zNear, zFar);
	sceneData->projectionMatrix = projectionMatrix;
	sceneData->p11 = projectionMatrix[0][0];
	sceneData->p22 = projectionMatrix[1][1];
	sceneData->p33 = projectionMatrix[2][2];
	sceneData->p43 = projectionMatrix[3][2];
	sceneData->ambientLight = glm::vec3(0.2f, 0.2f, 0.2f);
	sceneData->numLights = 0;
	sceneData->padding = {};
	sceneData->lights[0].eyePositionOrDirection = glm::vec3(0.f, 0.f, 0.f);
	sceneData->lights[0].settings = 2;  // bits 0..1: 1 - directional light, 2 - point light, 3 - spotlight
	sceneData->lights[0].opengl = {
		.ambient = glm::vec3(0.f, 0.f, 0.f),
		.constantAttenuation = 1.f,
		.diffuse = glm::vec3(0.6f, 0.6f, 0.6f),
		.linearAttenuation = 0.f,
		.specular = glm::vec3(0.6f, 0.6f, 0.6f),
		.quadraticAttenuation = 0.f,
	};
	sceneData->lights[0].gltf = {
		.color = glm::vec3(0.8f, 0.8f, 0.8f),
		.intensity = 1.f,
		.range = numeric_limits<float>::infinity(),
	};
	sceneData->lights[0].spotlight = {};
	sceneData->lights[1] = {
		.settings = 0,
	};

	renderGUI();

	// begin the frame
	renderer.beginFrame();

	// submit all copy operations that were not submitted yet
	renderer.executeCopyOperations();

	// acquire image
	uint32_t imageIndex;
	r =
		device.acquireNextImageKHR(
			swapchain,                // swapchain
			uint64_t(3e9),            // timeout (3s)
			imageAvailableSemaphore,  // semaphore to signal
			vk::Fence(nullptr),       // fence to signal
			&imageIndex               // pImageIndex
		);
	if(r != vk::Result::eSuccess) {
		renderer.endFrame();
		if(r == vk::Result::eSuboptimalKHR) {
			window.scheduleResize();
			return;
		} else if(r == vk::Result::eErrorOutOfDateKHR) {
			window.scheduleResize();
			return;
		} else
			throw runtime_error("Vulkan error: vkAcquireNextImageKHR failed with error " + to_string(r) + ".");
	}

	// begin command buffer recording
	renderer.beginRecording(commandBuffer);

	// prepare recording
	size_t numDrawables = renderer.prepareSceneRendering(stateSetRoot);

	// record compute shader preprocessing
	renderer.recordDrawableProcessing(commandBuffer, numDrawables);

	// record scene rendering
	device.cmdPushConstants(
		commandBuffer,  // commandBuffer
		pipelineSceneGraph.pipelineLayout(),  // pipelineLayout
		vk::ShaderStageFlagBits::eAllGraphics,  // stageFlags
		0,  // offset
		sizeof(uint64_t),  // size
		array<uint64_t,1>{  // pValues
			sceneDataAllocation.deviceAddress(),  // sceneDataPtr
		}.data()
	);
	renderer.recordSceneRendering(
		commandBuffer,  // commandBuffer
		stateSetRoot,  // stateSetRoot
		renderPass,  // renderPass
		framebuffers[imageIndex],  // framebuffer
		vk::Rect2D(vk::Offset2D(0, 0), window.surfaceExtent()),  // renderArea
		2,  // clearValueCount
		array<vk::ClearValue,2>{  // pClearValues
			vk::ClearColorValue(array<float,4>{0.f, 0.f, 0.f, 1.f}),
			vk::ClearDepthStencilValue(1.f, 0),
		}.data()
	);

	// end command buffer recording
	renderer.endRecording(commandBuffer);

	// submit all copy operations that were not submitted yet
	renderer.executeCopyOperations();

	// submit frame
	vk::Semaphore renderingFinishedSemaphore = renderingFinishedSemaphores[imageIndex];
	device.queueSubmit(
		graphicsQueue,  // queue
		vk::SubmitInfo(
			1, &imageAvailableSemaphore,  // waitSemaphoreCount + pWaitSemaphores +
			&(const vk::PipelineStageFlags&)vk::PipelineStageFlags(  // pWaitDstStageMask
				vk::PipelineStageFlagBits::eColorAttachmentOutput),
			1, &commandBuffer,  // commandBufferCount + pCommandBuffers
			1, &renderingFinishedSemaphore  // signalSemaphoreCount + pSignalSemaphores
		),
		renderingFinishedFence  // fence
	);

	// present
	r =
		device.presentKHR(
			presentationQueue,  // queue
			&(const vk::PresentInfoKHR&)vk::PresentInfoKHR(  // presentInfo
				1, &renderingFinishedSemaphore,  // waitSemaphoreCount + pWaitSemaphores
				1, &swapchain, &imageIndex,  // swapchainCount + pSwapchains + pImageIndices
				nullptr  // pResults
			)
		);
	if(r != vk::Result::eSuccess) {
		if(r == vk::Result::eSuboptimalKHR) {
			window.scheduleResize();
			cout << "present result: Suboptimal" << endl;
		} else if(r == vk::Result::eErrorOutOfDateKHR) {
			window.scheduleResize();
			cout << "present error: OutOfDate" << endl;
		} else
			throw runtime_error("Vulkan error: vkQueuePresentKHR() failed with error " + to_string(r) + ".");
	}

	// end of the frame
	// (gpu computations might be running asynchronously now
	// and presentation might be waiting for the rendering to finish)
	renderer.endFrame();

	if (noPause) {
		window.scheduleFrame();
	}
}


void App::mouseMove(VulkanWindow& window, const VulkanWindow::MouseState& mouseState)
{
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}
	if(mouseState.buttons[VulkanWindow::MouseButton::Left]) {

		if (newCamera) {
			cameraAngles += glm::vec2{
				(prevMouseX - mouseState.posX) * cameraSensitivity,
				glm::clamp((mouseState.posY - prevMouseY) * cameraSensitivity, -glm::half_pi<float>(), glm::half_pi<float>())
			};
			prevMouseX = mouseState.posX;
			prevMouseY = mouseState.posY;

			const float cosy = cos(cameraAngles.y);
			cameraDirection = glm::normalize(glm::vec3{
				cos(cameraAngles.x) * cosy,
				sin(cameraAngles.y),
				sin(cameraAngles.x) * cosy
			});
			cameraRight = glm::normalize(glm::cross(cameraDirection, cameraUp));
		}
		else {
			cameraHeading = startCameraHeading + (mouseState.posX - startMouseX) * 0.01f;
			cameraElevation = startCameraElevation + (mouseState.posY - startMouseY) * 0.01f;
		}

		window.scheduleFrame();
	}
}


void App::mouseButton(VulkanWindow& window, size_t button, VulkanWindow::ButtonState buttonState,
                      const VulkanWindow::MouseState& mouseState)
{
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}
	if(button == VulkanWindow::MouseButton::Left) {
		if(buttonState == VulkanWindow::ButtonState::Pressed) {
			startMouseX = mouseState.posX;
			startMouseY = mouseState.posY;
			prevMouseX = mouseState.posX;
			prevMouseY = mouseState.posY;

			startCameraHeading = cameraHeading;
			startCameraElevation = cameraElevation;
		}
		else {
			cameraHeading = startCameraHeading + (mouseState.posX - startMouseX) * 0.01f;
			cameraElevation = startCameraElevation + (mouseState.posY - startMouseY) * 0.01f;
			window.scheduleFrame();
		}
	}
}


void App::mouseWheel(VulkanWindow& window, float wheelX, float wheelY, const VulkanWindow::MouseState& mouseState)
{
	if (ImGui::GetIO().WantCaptureMouse) {
		return;
	}
	if (newCamera) {
		fovy -= wheelY * 0.1;
	}
	else {
		cameraDistance *= pow(zoomStepRatio, wheelY);
	}
	window.scheduleFrame();
}

void App::key(VulkanWindow& window, VulkanWindow::KeyState keyState, VulkanWindow::ScanCode scanCode)
{
	if (ImGui::GetIO().WantCaptureKeyboard) {
		return;
	}
    if (keyState == VulkanWindow::KeyState::Pressed) {
    	switch (scanCode) {
    		case VulkanWindow::ScanCode::W:
    			cameraEye += cameraDirection * cameraSpeed;
    			window.scheduleFrame();
    			break;
    		case VulkanWindow::ScanCode::S:
    			cameraEye -= cameraDirection * cameraSpeed;
    			window.scheduleFrame();
    			break;
    		case VulkanWindow::ScanCode::A:
    			cameraEye += cameraRight * cameraSpeed;
    			window.scheduleFrame();
    			break;
    		case VulkanWindow::ScanCode::D:
    			cameraEye -= cameraRight * cameraSpeed;
    			window.scheduleFrame();
    			break;
    		case VulkanWindow::ScanCode::One:
    			useUberShader = true;
    			setPipelines();
    			window.scheduleFrame();
    			std::cout << "Switched to uber shader\n";
    			break;
    		case VulkanWindow::ScanCode::Two:
    			useUberShader = false;
    			setPipelines();
    			window.scheduleFrame();
    			std::cout << "Switched to optimized shader\n";
    			break;
    		case VulkanWindow::ScanCode::C:
    			newCamera = !newCamera;
    			window.scheduleFrame();
    			break;
    		case VulkanWindow::ScanCode::R:
    			break;
    		default:
    			break;
    	}
    }
}

void App::setCamera(glm::vec3 position, glm::vec3 target, glm::vec3 up)
{
	cameraEye = position;
	cameraDirection = glm::normalize(target - cameraEye);
	cameraUp = up;
	cameraRight = glm::normalize(glm::cross(cameraDirection, up));
	cameraAngles = {
		std::atan2(cameraDirection.z, cameraDirection.x),
		std::asin(cameraDirection.y)
	};
}

void App::renderGUI() {
	ImGui_ImplVulkan_NewFrame();
#if defined(USE_PLATFORM_WIN32)
	ImGui_ImplWin32_NewFrame();
#elif defined(USE_PLATFORM_SDL3)
	ImGui_ImplSDL3NewFrame();
#elif defined(USE_PLATFORM_SDL2)
	ImGui_ImplSDL2_NewFrame();
#elif defined(USE_PLATFORM_GLFW)
	ImGui_ImplGlfw_NewFrame();
#endif

	ImGui::NewFrame();

	if (ImGui::Begin("Debug", &imguiPanel, 0)) {
		const auto now = std::chrono::steady_clock::now();
		if (now >= lastDebugTime) {
			lastDebugTime = now + std::chrono::seconds(1);
			double time = gpuTimeCounter.get();
			std::stringstream str;
			if (time > 1.0) {
				str << time << "s";
			}
			else if (time > 1e-3) {
				str << time * 1e3 << "ms";
			}
			else if (time > 1e-6) {
				str << time * 1e6 << "us";
			}
			else if (time > 1e-9) {
				str << time * 1e9 << "ns";
			}
			// if (time > 0) {
			// 	std::cout << ", fps: " << 1.0 / time << "\n";
			// }
			guiFrameTime = std::move(str.str());
		}
		ImGui::Text("Frame time: %s", guiFrameTime.c_str());

		ImGui::End();
	}

	ImGui::EndFrame();

	ImGui::Render();
}

uint32_t getStride(int componentType,const string& type)
{
	uint32_t componentSize;
	switch(componentType) {
	case 5120:                          // BYTE
	case 5121: componentSize=1; break;  // UNSIGNED_BYTE
	case 5122:                          // SHORT
	case 5123: componentSize=2; break;  // UNSIGNED_SHORT
	case 5125:                          // UNSIGNED_INT
	case 5126: componentSize=4; break;  // FLOAT
	default:   throw GltfError("Invalid accessor's componentType value.");
	}
	if(type=="VEC3")  return 3*componentSize;
	else if(type=="VEC4")  return 4*componentSize;
	else if(type=="VEC2")  return 2*componentSize;
	else if(type=="SCALAR")  return componentSize;
	else if(type=="MAT4")  return 16*componentSize;
	else if(type=="MAT3")  return 9*componentSize;
	else if(type=="MAT2")  return 4*componentSize;
	else throw GltfError("Invalid accessor's type.");
}


vk::Format getFormat(int componentType,const string& type,bool normalize,bool wantInt=false)
{
	if(componentType>=5127)
		throw GltfError("Invalid accessor's componentType.");

	// FLOAT component type
	if(componentType==5126) {
		if(normalize)
			throw GltfError("Normalize set while accessor's componentType is FLOAT (5126).");
		if(wantInt)
			throw GltfError("Integer format asked while accessor's componentType is FLOAT (5126).");
		if(type=="VEC3")       return vk::Format::eR32G32B32Sfloat;
		else if(type=="VEC4")  return vk::Format::eR32G32B32A32Sfloat;
		else if(type=="VEC2")  return vk::Format::eR32G32Sfloat;
		else if(type=="SCALAR")  return vk::Format::eR32Sfloat;
		else if(type=="MAT4")  return vk::Format::eR32G32B32A32Sfloat;
		else if(type=="MAT3")  return vk::Format::eR32G32B32Sfloat;
		else if(type=="MAT2")  return vk::Format::eR32G32Sfloat;
		else throw GltfError("Invalid accessor's type.");
	}

	// UNSIGNED_INT component type
	else if(componentType==5125)
		throw GltfError("UNSIGNED_INT componentType shall be used only for accessors containing indices. No attribute format is supported.");

	// INT component type
	else if(componentType==5124)
		throw GltfError("Invalid componentType. INT is not valid value for glTF 2.0.");

	// SHORT and UNSIGNED_SHORT component type
	else if(componentType>=5122) {
		int base;
		if(type=="VEC3")       base=84;  // VK_FORMAT_R16G16B16_UNORM
		else if(type=="VEC4")  base=91;  // VK_FORMAT_R16G16B16A16_UNORM
		else if(type=="VEC2")  base=77;  // VK_FORMAT_R16G16_UNORM
		else if(type=="SCALAR")  base=70;  // VK_FORMAT_R16_UNORM
		else if(type=="MAT4")  base=91;
		else if(type=="MAT3")  base=84;
		else if(type=="MAT2")  base=77;
		else throw GltfError("Invalid accessor's type.");
		if(componentType==5122)  // signed SHORT
			base+=1;  // VK_FORMAT_R16*_S*
		if(wantInt)    return vk::Format(base+4);  // VK_FORMAT_R16*_[U|S]INT
		if(normalize)  return vk::Format(base);    // VK_FORMAT_R16*_[U|S]NORM
		else           return vk::Format(base+2);  // VK_FORMAT_R16*_[U|S]SCALED
	}

	// BYTE and UNSIGNED_BYTE component type
	else if(componentType>=5120) {
	int base;
		if(type=="VEC3")       base=23;  // VK_FORMAT_R8G8B8_UNORM
		else if(type=="VEC4")  base=37;  // VK_FORMAT_R8G8B8A8_UNORM
		else if(type=="VEC2")  base=16;  // VK_FORMAT_R8G8_UNORM
		else if(type=="SCALAR")  base=9;  // VK_FORMAT_R8_UNORM
		else if(type=="MAT4")  base=37;
		else if(type=="MAT3")  base=23;
		else if(type=="MAT2")  base=16;
		else throw GltfError("Invalid accessor's type.");
		if(componentType==5120)  // signed BYTE
			base+=1;  // VK_FORMAT_R8*_S*
		if(wantInt)    return vk::Format(base+4);  // VK_FORMAT_R16*_[U|S]INT
		if(normalize)  return vk::Format(base);    // VK_FORMAT_R16*_[U|S]NORM
		else           return vk::Format(base+2);  // VK_FORMAT_R16*_[U|S]SCALED
	}

	// componentType bellow 5120
	throw GltfError("Invalid accessor's componentType.");
	return vk::Format(0);
}


int main(int argc, char* argv[])
try {

	// set console code page to utf-8 to print non-ASCII characters correctly
#ifdef _WIN32
	if(!SetConsoleOutputCP(CP_UTF8))
		cout << "Failed to set console code page to utf-8." << endl;
#endif

    CadPL::ShaderGenerator::initializeCache();

	// init application
	App app(argc, argv);
	app.init();
	app.window.setResizeCallback(
		bind(
			&App::resize,
			&app,
			placeholders::_1,
			placeholders::_2,
			placeholders::_3
		)
	);
	app.window.setFrameCallback(
		bind(&App::frame, &app, placeholders::_1)
	);
	app.window.setMouseMoveCallback(
		bind(&App::mouseMove, &app, placeholders::_1, placeholders::_2)
	);
	app.window.setMouseButtonCallback(
		bind(&App::mouseButton, &app, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4)
	);
	app.window.setMouseWheelCallback(
		bind(&App::mouseWheel, &app, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4)
	);
    app.window.setKeyCallback(
        bind(&App::key, &app, placeholders::_1, placeholders::_2, placeholders::_3)
    );

	// show window and run main loop
	app.window.show();
	VulkanWindow::mainLoop();

	// finish all pending work on device
	app.device.waitIdle();
	return 0;

// catch exceptions
} catch(CadR::Error &e) {
	cout << "Failed because of CadR exception: " << e.what() << endl;
	return 9;
} catch(vk::Error &e) {
	cout << "Failed because of Vulkan exception: " << e.what() << endl;
	return 9;
} catch(ExitWithMessage &e) {
	cout << e.what() << endl;
	return e.exitCode();
} catch(exception &e) {
	cout << "Failed because of exception: " << e.what() << endl;
	return 9;
} catch(...) {
	cout << "Failed because of unspecified exception." << endl;
	return 9;
}
