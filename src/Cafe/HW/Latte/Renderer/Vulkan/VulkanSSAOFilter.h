#pragma once

#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.h"
#include "util/math/vector2.h"

class VulkanRenderer;
class LatteTexture;
class LatteTextureView;
class LatteTextureVk;
class LatteTextureViewVk;
class RendererShaderVk;
class VKRObjectTexture;
class VKRObjectTextureView;
class VKRObjectRenderPass;
class VKRObjectFramebuffer;
class VKRObjectSampler;

// Screen-space effects (HBAO + SSR, each with its own menu toggle), applied as
// one standalone fullscreen pass at scanout time (Vulkan only), same
// architecture as VulkanTAAFilter: reads the game's textures, never writes
// into them. Both effects share this pass because they consume exactly the
// same inputs (post-TAA color + depth) - one pass, one output image. Chained
// after TAA when both are enabled (samples TAA's resolved color if valid,
// otherwise the raw scanout).
// Depth comes from LatteMRT::GetDepthAttachment() at the same call-site timing
// TAA uses for color - the last depth buffer the game bound before the scanout
// swap. As with the original TAA color-aliasing bug, this can be wrong for
// games that rebind/reuse render targets for post-processing after the main
// scene pass; verify AO shape against expected occluders in-game.
class VulkanSSAOFilter
{
public:
	static VulkanSSAOFilter& GetInstance();

	// records the AO pass into the current command buffer. Called after TAA_Apply
	// (if TAA ran) and before DrawBackbufferQuad. scanoutView must be the exact
	// view the present path would sample, same caveat as VulkanTAAFilter::Apply.
	void Apply(VulkanRenderer* renderer, LatteTextureViewVk* scanoutView);

	// descriptor set (matching the backbuffer blit layout) that samples the
	// latest AO'd image. Returns null when there is no valid output of the
	// expected size; the caller then presents the previous stage's image.
	VkDescriptorSet GetPresentDescriptorSet(VulkanRenderer* renderer, VkDescriptorSetLayout blitLayout,
											sint32 expectedWidth, sint32 expectedHeight);

	void Shutdown(VulkanRenderer* renderer);

	// called by LatteSSAO::NotifyTextureDeletion (in turn called from
	// LatteTexture_Delete) so the cached depth fallback below never outlives
	// its texture
	void NotifyTextureDeletion(LatteTexture* texture);

	// called by LatteSSAO::NotifyDepthBind (from LatteMRT) on every depth bind;
	// remembers the last depth view bound this frame whose size matches the
	// scene, which survives cutscene camera cuts that swap in new depth buffers
	void NotifyDepthBind(LatteTextureView* view);

private:
	VulkanSSAOFilter() = default;

	bool RecreateIfNeeded(VulkanRenderer* renderer, sint32 width, sint32 height, VkFormat format);
	void ReleaseResources(VulkanRenderer* renderer);
	bool CreateShaders();
	bool CreateStaticObjects(VulkanRenderer* renderer);
	bool CreateSizedObjects(VulkanRenderer* renderer);
	static uint32 FindMemoryType(VkPhysicalDevice physDev, uint32 typeBits, VkMemoryPropertyFlags properties);

	struct PushConstants
	{
		float radiusU, radiusV;  // sample disk radius in UV units (x/y, aspect-corrected)
		float intensity;
		float bias;
		float debugShowAOOnly;
		float jitterUVx, jitterUVy; // TAA viewport jitter (UV units), to unjitter depth sampling
		float width, height; // render target size in pixels, for the tangent-plane depth prediction
		float falloffRange; // raw-depth window past bias where a sample fades from full occluder to none
		float heightScale; // converts raw-depth deltas into UV-comparable units for the horizon angle
		float ssrStrength; // reflection blend weight (0 = SSR off, the shader skips the march)
		float ssrDepthScale; // raw-depth-to-UV scale of the SSR pseudo view space (normals + ray)
		float ssrThickness; // raw-depth window in which a marched ray counts as hitting a surface
		float ssrDebug; // output only the weighted reflection term
		float aoLumFadeStart; // luminance where AO starts fading out (protects water/glow/light shafts)
		float aoFloor; // AO never darkens below this (avoids out-of-place pure black)
		float ssrRoughness; // reflection cone blur factor
		float csStrength, csDirX, csDirY, csLength; // contact shadows (0 strength = off)
		float giStrength; // SSGI bounce weight (0 = off, the shader skips the color taps)
		float depthLinearK; // shape parameter for approximate view-space depth linearization
		float fovScale; // tan(fovYDegrees/2), precomputed on the CPU side
		float smoothRadiusView; // pass S kernel target reach, in linearized view-space units
	};

	sint32 m_width{ 0 };
	sint32 m_height{ 0 };
	VkFormat m_format{ VK_FORMAT_UNDEFINED };

	// final output image, presented instead of the game's scanout (no history
	// needed, this pass is not temporal)
	VkImage m_image{ VK_NULL_HANDLE };
	VkDeviceMemory m_memory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_texObj{};
	VKRObjectTextureView* m_viewObj{};
	VKRObjectFramebuffer* m_framebuffer{};
	VkImageLayout m_layout{ VK_IMAGE_LAYOUT_UNDEFINED };
	VKRObjectRenderPass* m_renderPass{};
	VkPipeline m_pipeline{ VK_NULL_HANDLE };

	// intermediate target of the effects pass (rgb = SSR-blended color, a = raw
	// AO term). A second depth-aware blur pass reads it and composites into
	// m_image; without it the IGN dither used to break up banding was stamped
	// straight onto the presented frame as a fixed cross-hatch pattern (nothing
	// temporal runs after this filter to average it out)
	VkImage m_aoImage{ VK_NULL_HANDLE };
	VkDeviceMemory m_aoMemory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_aoTexObj{};
	VKRObjectTextureView* m_aoViewObj{};
	VKRObjectFramebuffer* m_aoFramebuffer{}; // MRT: attachment 0 = m_aoImage, 1 = m_giImage
	VkImageLayout m_aoLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
	VKRObjectRenderPass* m_aoRenderPass{};
	VkPipeline m_pipelineBlur{ VK_NULL_HANDLE };

	// second MRT attachment of the effects pass: SSGI bounce radiance, blurred
	// alongside the AO term in pass B and composited additively
	VkImage m_giImage{ VK_NULL_HANDLE };
	VkDeviceMemory m_giMemory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_giTexObj{};
	VKRObjectTextureView* m_giViewObj{};
	VkImageLayout m_giLayout{ VK_IMAGE_LAYOUT_UNDEFINED };

	// normals reconstructed from depth (pass N) and their wide depth/normal-aware
	// smoothing (pass S) - the Launchpad-style "smoothed normals" that fix the
	// faceted/polygonal SSR: depth only contains the real flat triangles, the
	// smoothing recreates the smoothing groups the game applies in shading
	VkImage m_normalImage{ VK_NULL_HANDLE };
	VkDeviceMemory m_normalMemory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_normalTexObj{};
	VKRObjectTextureView* m_normalViewObj{};
	VKRObjectFramebuffer* m_normalFramebuffer{};
	VkImageLayout m_normalLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
	VKRObjectRenderPass* m_normalRenderPass{};
	VkPipeline m_pipelineNormal{ VK_NULL_HANDLE };

	VkImage m_normalSmoothImage{ VK_NULL_HANDLE };
	VkDeviceMemory m_normalSmoothMemory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_normalSmoothTexObj{};
	VKRObjectTextureView* m_normalSmoothViewObj{};
	VKRObjectFramebuffer* m_normalSmoothFramebuffer{};
	VkImageLayout m_normalSmoothLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
	VKRObjectRenderPass* m_normalSmoothRenderPass{};
	VkPipeline m_pipelineSmooth{ VK_NULL_HANDLE };

	RendererShaderVk* m_vertexShader{};
	RendererShaderVk* m_fragmentShader{};
	RendererShaderVk* m_blurShader{};
	RendererShaderVk* m_normalShader{};
	RendererShaderVk* m_smoothShader{};
	VKRObjectSampler* m_sampler{}; // shared by color and depth inputs (both clamp-to-edge)
	// one shared 4-binding layout for every pass (0/2/3 = pass-specific inputs,
	// 1 = depth everywhere; unused slots are filled with the depth view so the
	// set always matches the layout). 4 sets consumed per Apply (one per pass)
	static constexpr uint32 kDescriptorBindings = 4;
	VkDescriptorSetLayout m_descriptorSetLayout{ VK_NULL_HANDLE };
	VkPipelineLayout m_pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorPool m_descriptorPool{ VK_NULL_HANDLE };
	static constexpr uint32 kDescriptorRingSize = 16;
	VkDescriptorSet m_descriptorRing[kDescriptorRingSize]{};
	uint32 m_descriptorRingIndex{ 0 };

	bool m_hasValidOutput{ false };
	// re-presents without new game draws (e.g. 30fps cutscenes shown at 60Hz)
	// carry no new frame; recomputing AO against an unchanged depth buffer using
	// a different jitter offset each redundant present is what caused flicker
	// there. Mirrors VulkanTAAFilter's own drawCallCounter guard
	uint32 m_lastDrawCallCounter{ 0xFFFFFFFF };
	// detects presents whose draws were overlays only (no new scene/depth) via
	// LatteTAA::GetSceneDrawCounter, same rationale as VulkanTAAFilter
	uint32 m_lastSceneDrawCounter{ 0xFFFFFFFF };
	uint32 m_consecutiveSceneless{ 0 }; // presents in a row without scene draws; past ~30 = pure-2D mode (menus), present raw

	// last depth attachment that actually matched the scene's color resolution.
	// The game rebinds render targets after the main scene pass (HUD/UI, letterbox
	// bars, subtitles - none of which use a depth buffer of their own), so at the
	// exact scanout-swap instant GetDepthAttachment() often returns null or a
	// mismatched size even though the real scene depth was rendered fine this same
	// frame. Since it's the same persistent GX2 depth buffer object frame to frame,
	// reusing this cached view when the current one is unusable recovers AO for
	// those frames instead of silently producing nothing. Invalidated via
	// NotifyTextureDeletion so this never outlives the texture it points at
	LatteTextureView* m_cachedDepthView{ nullptr };

	// depth view captured at BIND time during the current frame (via
	// NotifyDepthBind), sized like the last known scene output. More reliable
	// than both the swap-time attachment (usually HUD by then) and
	// m_cachedDepthView (goes stale when cutscene cuts replace the depth
	// buffer): whatever depth the scene pass actually rendered into this frame
	// was necessarily bound during it. Cleared alongside the cache on texture
	// deletion; checked before the older cache in Apply()
	LatteTextureView* m_frameDepthView{ nullptr };

	VkDescriptorPool m_presentDescriptorPool{ VK_NULL_HANDLE };
	static constexpr uint32 kPresentRingSize = 8;
	VkDescriptorSet m_presentRing[kPresentRingSize]{};
	uint32 m_presentRingIndex{ 0 };

	struct PendingDelete
	{
		VkImage image;
		VkDeviceMemory memory;
		VkPipeline pipeline;
		uint64 safeCmdBufferId; // destroy once VulkanRenderer::HasCommandBufferFinished(safeCmdBufferId)
	};
	std::vector<PendingDelete> m_pendingDeletes;
	void TickPendingDeletes(VulkanRenderer* renderer);
};
