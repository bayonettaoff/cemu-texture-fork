#pragma once

#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.h"

class VulkanRenderer;
class LatteTextureVk;
class LatteTextureViewVk;
class LatteTexture;
class LatteTextureView;
class VKRObjectTexture;
class VKRObjectTextureView;
class VKRObjectSampler;

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

// DLAA via NVIDIA's real NGX/DLSS runtime (see Core/LatteDLSS.h and VulkanRenderer::InitNGX).
// Unlike VulkanTAAFilter/VulkanSSAOFilter, this filter has no shaders/render pass of its
// own: NGX_VULKAN_EvaluateFeature does the actual work as an opaque GPU command sequence
// recorded into our command buffer. Reuses VulkanTAAFilter's motion vector search pass
// (same block-matching estimator, see LatteTAA.h) instead of duplicating it - real per-game
// motion data isn't available (GX2 constraint, same one that blocks true view-space
// reconstruction everywhere else in this fork).
class VulkanDLSSFilter
{
public:
	static VulkanDLSSFilter& GetInstance();

	// records the DLAA evaluate into the current command buffer. Called after TAA_Apply
	// (needs its motion vector output) and before SSAO_Apply (SSAO should see the AA'd
	// image). scanoutView must be the exact view the present path would sample (see
	// VulkanTAAFilter::Apply's comment - the base texture view can alias unrelated data).
	void Apply(VulkanRenderer* renderer, LatteTextureViewVk* scanoutView);

	// descriptor set (matching the backbuffer blit layout) that samples the latest
	// DLAA output. Returns null when there is no valid output of the expected size.
	VkDescriptorSet GetPresentDescriptorSet(VulkanRenderer* renderer, VkDescriptorSetLayout blitLayout,
											sint32 expectedWidth, sint32 expectedHeight);

	// raw view of the latest DLAA output, for SSAO/SSR chaining after DLAA in the same
	// command buffer - already left in a shader-readable layout, no extra barrier needed.
	VkImageView GetResolvedImageViewIfValid(sint32 expectedWidth, sint32 expectedHeight);

	// called by LatteDLSS::NotifyTextureDeletion (from LatteTexture_Delete) so a cached
	// depth pointer never outlives the texture it points at
	void NotifyTextureDeletion(LatteTexture* texture);

	// called by LatteDLSS::NotifyDepthBind (from LatteMRT) on every depth bind - same
	// cutscene-camera-cut fix as VulkanSSAOFilter's NotifyDepthBind, kept independent
	// (own cache) rather than reading SSAO's, since SSAO can be toggled off while DLAA
	// stays on
	void NotifyDepthBind(LatteTextureView* view);

	void Shutdown(VulkanRenderer* renderer);

private:
	VulkanDLSSFilter() = default;

	bool RecreateIfNeeded(VulkanRenderer* renderer, sint32 width, sint32 height, VkFormat format);
	void ReleaseResources(VulkanRenderer* renderer);
	void TickPendingDeletes(VulkanRenderer* renderer);
	static uint32 FindMemoryType(VkPhysicalDevice physDev, uint32 typeBits, VkMemoryPropertyFlags properties);

	sint32 m_width{ 0 };
	sint32 m_height{ 0 };
	VkFormat m_format{ VK_FORMAT_UNDEFINED };

	// DLAA output (single image, no ping-pong - NGX reads history internally)
	VkImage m_image{ VK_NULL_HANDLE };
	VkDeviceMemory m_memory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_texObj{};
	VKRObjectTextureView* m_viewObj{};
	VkImageLayout m_layout{ VK_IMAGE_LAYOUT_UNDEFINED };

	NVSDK_NGX_Handle* m_dlssHandle{ nullptr };
	NVSDK_NGX_Parameter* m_ngxParams{ nullptr };

	// depth cache - see NotifyDepthBind/NotifyTextureDeletion comments above
	LatteTextureView* m_cachedDepthView{ nullptr };
	LatteTextureView* m_frameDepthView{ nullptr };

	uint32 m_lastDrawCallCounter{ 0xFFFFFFFF };
	uint32 m_lastSceneDrawCounter{ 0xFFFFFFFF };
	uint32 m_consecutiveSceneless{ 0 };
	bool m_hasValidOutput{ false };
	bool m_historyValid{ false }; // false on the first evaluate after (re)creation - tells NGX InReset=1

	// present descriptor ring (allocated against the renderer's backbuffer blit layout
	// on first use), same idiom as VulkanTAAFilter
	VkDescriptorPool m_presentDescriptorPool{ VK_NULL_HANDLE };
	VKRObjectSampler* m_sampler{};
	static constexpr uint32 kPresentRingSize = 8;
	VkDescriptorSet m_presentRing[kPresentRingSize]{};
	uint32 m_presentRingIndex{ 0 };

	struct PendingDelete
	{
		VkImage image;
		VkDeviceMemory memory;
		uint64 safeCmdBufferId;
	};
	std::vector<PendingDelete> m_pendingDeletes;
};
