#pragma once

#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.h"
#include "util/math/vector2.h"

#include <chrono>

class VulkanRenderer;
class LatteTextureVk;
class LatteTextureViewVk;
class RendererShaderVk;
class VKRObjectTexture;
class VKRObjectTextureView;
class VKRObjectRenderPass;
class VKRObjectFramebuffer;
class VKRObjectSampler;

// Temporal resolve pass for LatteTAA (Vulkan only).
// Ping-pong history buffer; each frame: resolved = clamp-blend(scanout, history),
// rendered into history[dst]. The game's scanout texture is only ever READ:
// games may reuse the scan buffer as an input of their own post-processing the
// next frame, so writing the filtered image back corrupts their effects.
// DrawBackbufferQuad presents the resolved image via GetPresentDescriptorSet.
class VulkanTAAFilter
{
public:
	static VulkanTAAFilter& GetInstance();

	// records the resolve into the current command buffer. Called between the
	// game's last draw and DrawBackbufferQuad. scanoutView must be the exact view
	// the present path would sample (NOT the base texture's default view: the
	// scan buffer can live in a specific slice/mip of an aliased texture, and the
	// base view may hold unrelated intermediate data such as the game's motion
	// blur buffers).
	void Apply(VulkanRenderer* renderer, LatteTextureViewVk* scanoutView);

	// descriptor set (matching the backbuffer blit layout: binding 0, combined
	// image sampler) that samples the latest resolved image. Returns null when
	// there is no valid output of the expected size; the caller then presents
	// the game texture unmodified.
	VkDescriptorSet GetPresentDescriptorSet(VulkanRenderer* renderer, VkDescriptorSetLayout blitLayout,
											sint32 expectedWidth, sint32 expectedHeight);

	void Shutdown(VulkanRenderer* renderer);

private:
	VulkanTAAFilter() = default;

	bool RecreateIfNeeded(VulkanRenderer* renderer, sint32 width, sint32 height, VkFormat format);
	void ReleaseResources(VulkanRenderer* renderer);
	bool CreateShaders();
	bool CreateStaticObjects(VulkanRenderer* renderer);      // sampler, descriptor pool, set layout, pipeline layout
	bool CreateSizedObjects(VulkanRenderer* renderer);       // images, views, renderpass, framebuffers, pipeline
	static uint32 FindMemoryType(VkPhysicalDevice physDev, uint32 typeBits, VkMemoryPropertyFlags properties);

	struct PushConstants
	{
		float blendFactor;
		float historyValid;
		float jitterUVx; // current-frame viewport jitter in UV units, for unjittering
		float jitterUVy;
		float passthrough; // 1.0 = output current frame only (debug)
	};

	// current dimensions/format of the history chain
	sint32 m_width{ 0 };
	sint32 m_height{ 0 };
	VkFormat m_format{ VK_FORMAT_UNDEFINED };

	// history ping-pong resources (raw images owned here, views owned by VKR wrappers)
	VkImage m_image[2]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
	VkDeviceMemory m_memory[2]{ VK_NULL_HANDLE, VK_NULL_HANDLE };
	VKRObjectTexture* m_texObj[2]{};
	VKRObjectTextureView* m_viewObj[2]{};
	VKRObjectFramebuffer* m_framebuffer[2]{};
	VkImageLayout m_layout[2]{ VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED };
	VKRObjectRenderPass* m_renderPass{};
	VkPipeline m_pipeline{ VK_NULL_HANDLE };

	// static objects (size-independent)
	RendererShaderVk* m_vertexShader{};
	RendererShaderVk* m_fragmentShader{};
	VKRObjectSampler* m_sampler{};
	VkDescriptorSetLayout m_descriptorSetLayout{ VK_NULL_HANDLE };
	VkPipelineLayout m_pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorPool m_descriptorPool{ VK_NULL_HANDLE };
	static constexpr uint32 kDescriptorRingSize = 8;
	VkDescriptorSet m_descriptorRing[kDescriptorRingSize]{};
	uint32 m_descriptorRingIndex{ 0 };

	uint32 m_currentHistory{ 0 }; // index holding last frame's resolved image
	uint32 m_lastDrawCallCounter{ 0xFFFFFFFF }; // detects re-presents without new game draws
	bool m_hasValidOutput{ false }; // history[m_currentHistory] holds a presentable resolve

	// low-framerate detection: games with dynamic framerate (30 fps cutscenes /
	// 60 fps gameplay) move twice as far per frame at 30 fps, where the blend
	// without motion vectors ghosts hard; the current-frame weight is raised there
	std::chrono::steady_clock::time_point m_lastResolveTime{};
	uint32 m_lowFpsHold{ 0 };

	// descriptor ring for the present override (allocated against the renderer's
	// backbuffer blit layout on first use)
	VkDescriptorPool m_presentDescriptorPool{ VK_NULL_HANDLE };
	static constexpr uint32 kPresentRingSize = 8;
	VkDescriptorSet m_presentRing[kPresentRingSize]{};
	uint32 m_presentRingIndex{ 0 };

	// deferred destruction of outdated raw images (VKR wrappers handle their own)
	struct PendingDelete
	{
		VkImage image;
		VkDeviceMemory memory;
		VkPipeline pipeline;
		uint32 framesLeft;
	};
	std::vector<PendingDelete> m_pendingDeletes;
	void TickPendingDeletes(VkDevice device);
};
