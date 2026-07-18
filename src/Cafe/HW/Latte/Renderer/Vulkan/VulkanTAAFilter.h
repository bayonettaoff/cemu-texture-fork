#pragma once

#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanAPI.h"
#include "util/math/vector2.h"

#include <chrono>

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

	// true if the most recent Apply() call did a genuine resolve for a new scene
	// frame, as opposed to an early-return (disabled, re-present with no new
	// draws, overlay-only present). The jitter sequence (LatteTAA::
	// NotifyFramePresented) must only advance on a real resolve - see the call
	// site in LatteRenderTarget_copyToBackbuffer for why: advancing on every
	// copyToBackbuffer call regardless of this would race the jitter index ahead
	// of actual rendered frames on the exact re-present patterns (30fps cutscene
	// presented at 60Hz) these guards exist to detect
	bool DidResolveLastCall() const { return m_resolvedLastCall; }

	// descriptor set (matching the backbuffer blit layout: binding 0, combined
	// image sampler) that samples the latest resolved image. Returns null when
	// there is no valid output of the expected size; the caller then presents
	// the game texture unmodified.
	VkDescriptorSet GetPresentDescriptorSet(VulkanRenderer* renderer, VkDescriptorSetLayout blitLayout,
											sint32 expectedWidth, sint32 expectedHeight);

	// raw view of the latest resolved image, for other filters (SSAO) chaining
	// after TAA in the same command buffer - already left in a shader-readable
	// layout by Apply(), no extra barrier needed. Null when there is no valid
	// output of the expected size.
	VkImageView GetResolvedImageViewIfValid(sint32 expectedWidth, sint32 expectedHeight);

	// raw view of this frame's motion vector search output (RG = UV-space offset,
	// see PushConstants comment above), for DLAA (VulkanDLSSFilter) to reuse instead
	// of running a second block-matching search. Null in FXAA mode (no MV pass runs)
	// or before the first Apply(). outWidth/outHeight receive the (half-res) dimensions;
	// outImage receives the raw VkImage backing the view (NGX needs both).
	VkImageView GetMotionVectorsViewIfValid(sint32& outWidth, sint32& outHeight, VkImage& outImage);

	// called by LatteTAA::NotifyTextureDeletion (from LatteTexture_Delete) so the
	// analytical-MV depth cache below never outlives its texture
	void NotifyTextureDeletion(LatteTexture* texture);

	// called by LatteTAA::NotifyDepthBind (from LatteMRT) on every depth bind - same
	// cutscene-camera-cut fix as VulkanSSAOFilter's NotifyDepthBind, kept independent
	// (own cache) for the same reason SSAO/DLSS keep theirs independent
	void NotifyDepthBind(LatteTextureView* view);

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
		float mvSearchStep; // initial step of the hierarchical motion search (UV units)
		float mvRegularization; // cost penalty per unit of candidate offset (biases ties toward zero motion)
		float useMotionVectors; // 1.0 = resolve reprojects history through texMotionVectors, 0.0 = old static sample
		float mvDebugView; // 1.0 = resolve outputs the motion vector field as color instead of the resolved image
		float ofFlowScaleX; // UV units per pixel (1/mvWidth), only used by m_pipelineOFConvert
		float ofFlowScaleY; // 1/mvHeight
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

	// motion vector target (half resolution of the history chain - see
	// LatteTAA::Config::useMotionVectors). Recomputed fresh every resolve from
	// the current scanout + previous history, no ping-pong needed
	sint32 m_mvWidth{ 0 };
	sint32 m_mvHeight{ 0 };
	VkImage m_mvImage{ VK_NULL_HANDLE };
	VkDeviceMemory m_mvMemory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_mvTexObj{};
	VKRObjectTextureView* m_mvViewObj{};
	VKRObjectFramebuffer* m_mvFramebuffer{};
	VkImageLayout m_mvLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
	VKRObjectRenderPass* m_mvRenderPass{};
	VkPipeline m_pipelineMV{ VK_NULL_HANDLE };
	RendererShaderVk* m_mvShader{};

	// hardware optical flow (VK_NV_optical_flow) - alternative, much higher quality
	// motion source than the block-matching search above: real GPU-accelerated
	// motion estimation (RTX Ampere+), purely from the already-rendered current vs
	// previous frame, no per-game assumptions. Replaces what writes into m_mvImage
	// when available and enabled (see LatteTAA::Config::useOpticalFlow) - everything
	// downstream (this filter's own resolve, DLAA) keeps reading m_mvImage unchanged.
	bool m_ofSessionValid{ false };
	// set once CreateOpticalFlowSession has been tried and failed for a reason that
	// won't change without a driver/queue-architecture fix (currently: Cemu's single
	// graphics queue family lacks VK_QUEUE_OPTICAL_FLOW_BIT_NV on this system/driver -
	// NVOF commands need a dedicated queue family Cemu doesn't create) - stops Apply()
	// from retrying (and re-logging) every single frame once we know it can't work
	bool m_ofUnsupported{ false };
	VkOpticalFlowSessionNV m_ofSession{ VK_NULL_HANDLE };
	VkFormat m_ofInputFormat{ VK_FORMAT_UNDEFINED };
	VkFormat m_ofFlowFormat{ VK_FORMAT_UNDEFINED };
	VkOpticalFlowGridSizeFlagsNV m_ofGridSize{ VK_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_NV };

	// --- cross-queue pipeline state ---
	// The optical-flow-capable queue family (confirmed via VulkanRenderer::
	// GetOpticalFlowQueueFamilyIndex) is narrow on NVIDIA hardware: transfer +
	// optical flow only, no graphics/compute. The "convert scanout to luma" and
	// "convert flow vector to m_mvImage" passes are fragment shader render passes,
	// so they MUST run on the graphics queue - only the bare vkCmdOpticalFlowExecuteNV
	// call runs on the dedicated queue. This means the work is split across two
	// independently-submitted command buffers with a real cross-queue dependency,
	// synchronized here via host-side fence polling (checked once per Apply() call)
	// rather than GPU semaphores, specifically to avoid needing to inject a signal
	// semaphore into Cemu's own (shared, elsewhere-submitted) per-frame command
	// buffer submission - this keeps the whole mechanism self-contained in this
	// filter instead of touching VulkanRenderer's core submission flow. Trade-off:
	// motion vectors lag by a few frames instead of being perfectly current: the
	// variance clipping/regularization safety nets already tolerate that.
	//
	// 2 color slots, ping-ponged but gated: a slot is only overwritten once it is
	// no longer needed by an in-flight (or not-yet-consumed) optical flow execute.
	// With only 2 slots this makes updates bursty (write both, execute, wait for the
	// full round-trip, repeat) rather than continuous - a known, accepted limitation
	// of this first version; a deeper ring would smooth it out at the cost of more
	// image memory and a more involved slot-selection scheme.
	static constexpr uint32 kOFColorSlots = 2;
	VkImage m_ofColorImage[kOFColorSlots]{};
	VkDeviceMemory m_ofColorMemory[kOFColorSlots]{};
	VKRObjectTexture* m_ofColorTexObj[kOFColorSlots]{};
	VKRObjectTextureView* m_ofColorViewObj[kOFColorSlots]{};
	VKRObjectFramebuffer* m_ofColorFramebuffer[kOFColorSlots]{};
	VkImageLayout m_ofColorLayout[kOFColorSlots]{};
	VKRObjectRenderPass* m_ofColorRenderPass{}; // shared by both color slots (same format/size)
	// round-based occupancy: a slot is "occupied" (holds data another stage still
	// needs) from the moment it's written until the execute consuming it has been
	// fully converted into m_mvImage (stage 4) - spanning the ready-wait AND the
	// exec-in-flight period. Both slots of a round free up together in stage 4 and
	// get refilled together, rather than any single slot being eligible for reuse
	// the instant it individually becomes "ready" - an earlier version allowed
	// that and it meant a slot could get overwritten (silently discarding its
	// just-confirmed-ready data) before stage 3 ever got a chance to pair it with
	// its partner, so the "both slots ready" condition was never simultaneously
	// true and no execute ever launched. Round-based occupancy fixes that at the
	// cost of the bursty (not continuous) update cadence already noted above.
	bool m_ofColorOccupied[kOFColorSlots]{};
	bool m_ofColorReady[kOFColorSlots]{}; // true once the graphics-queue write is confirmed done (HasCommandBufferFinished)
	uint64 m_ofColorCmdBufferId[kOFColorSlots]{}; // Cemu command buffer id captured right after recording slot's conversion pass
	uint64 m_ofColorWriteSeq[kOFColorSlots]{}; // monotonic write order, to tell which of the 2 ready slots is INPUT (newer) vs REFERENCE (older)
	uint64 m_ofNextWriteSeq{ 1 };

	VkImage m_ofFlowImage{ VK_NULL_HANDLE };
	VkDeviceMemory m_ofFlowMemory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_ofFlowTexObj{};
	VKRObjectTextureView* m_ofFlowViewObj{};
	VkImageLayout m_ofFlowLayout{ VK_IMAGE_LAYOUT_UNDEFINED };

	// the dedicated optical-flow queue's own command pool/buffer/fence - entirely
	// separate from Cemu's per-frame graphics command buffer
	VkCommandPool m_ofCommandPool{ VK_NULL_HANDLE };
	VkCommandBuffer m_ofCommandBuffer{ VK_NULL_HANDLE };
	VkFence m_ofExecFence{ VK_NULL_HANDLE };
	bool m_ofExecInFlight{ false };
	uint32 m_ofExecInputSlot{ 0 };
	uint32 m_ofExecRefSlot{ 0 };
	// true once the pipeline has produced at least one real result - until then,
	// Apply() also runs the block-matching search so m_mvImage always holds
	// something valid; once true, block-matching stops (optical flow drives
	// m_mvImage exclusively, holding the last result between pipeline updates
	// rather than mixing two different motion estimators frame to frame)
	bool m_ofHasEverProducedResult{ false };

	RendererShaderVk* m_ofInputShader{}; // downscale+convert scanout -> m_ofColorImage (luma broadcast)
	VkPipeline m_pipelineOFInput{ VK_NULL_HANDLE };
	RendererShaderVk* m_ofConvertShader{}; // NVOF flow vector -> m_mvImage (UV-space RG, matching the block-matching output)
	VkPipeline m_pipelineOFConvert{ VK_NULL_HANDLE };
	// VK_FORMAT_R16G16_SFIXED5_NV (the flow vector format) doesn't support linear
	// filtering (confirmed via Vulkan validation: VUID-vkCmdDraw-magFilter-04553) -
	// needs its own NEAREST sampler, can't reuse m_sampler (LINEAR, shared by
	// everything else in this filter). Created once, alongside m_sampler.
	VKRObjectSampler* m_ofFlowSampler{};

	bool CreateOpticalFlowSession(VulkanRenderer* renderer);
	void ReleaseOpticalFlowSession(VulkanRenderer* renderer);
	// runs the cross-queue state machine described above; returns true if
	// m_mvImage was freshly updated with an optical flow result this call
	bool UpdateOpticalFlow(VulkanRenderer* renderer, VkCommandBuffer cmd, VkDevice device,
						   VkImageView scanoutViewRaw, float jitterUVx, float jitterUVy);

	// static objects (size-independent)
	RendererShaderVk* m_vertexShader{};
	RendererShaderVk* m_fxaaShader{};
	VkPipeline m_pipelineFxaa{ VK_NULL_HANDLE };
	RendererShaderVk* m_fragmentShader{};
	VKRObjectSampler* m_sampler{};
	// 3 bindings shared by every pipeline (resolve, FXAA, motion vector search):
	// 0 = current scanout, 1 = history, 2 = motion vectors. A pipeline that
	// doesn't use one of these still gets a descriptor set with all 3 written
	// (unused slot filled with the history view) so every set matches the layout
	static constexpr uint32 kDescriptorBindings = 3;
	VkDescriptorSetLayout m_descriptorSetLayout{ VK_NULL_HANDLE };
	VkPipelineLayout m_pipelineLayout{ VK_NULL_HANDLE };
	VkDescriptorPool m_descriptorPool{ VK_NULL_HANDLE };
	// 2 sets consumed per Apply() now (motion search pass + resolve pass)
	static constexpr uint32 kDescriptorRingSize = 16;
	VkDescriptorSet m_descriptorRing[kDescriptorRingSize]{};
	uint32 m_descriptorRingIndex{ 0 };

	uint32 m_currentHistory{ 0 }; // index holding last frame's resolved image
	uint32 m_lastDrawCallCounter{ 0xFFFFFFFF }; // detects re-presents without new game draws
	uint32 m_lastSceneDrawCounter{ 0xFFFFFFFF }; // detects presents whose draws were overlays only (no new scene)
	uint32 m_consecutiveSceneless{ 0 }; // presents in a row without scene draws; past ~30 = pure-2D mode (menus), present raw
	bool m_hasValidOutput{ false }; // history[m_currentHistory] holds a presentable resolve
	bool m_resolvedLastCall{ false }; // see DidResolveLastCall()

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

	// --- analytical (camera-reprojection) motion vectors, 2026-07-14 - see
	// LatteTAA::Config::useAnalyticalMV. Deliberately kept 100% separate from the
	// block-matching/optical-flow machinery above (own descriptor set layout, own
	// pipeline layout/push constant range, own pipeline) - the only thing shared
	// is m_mvRenderPass/m_mvFramebuffer/m_mvImage (already the right size/format
	// for exactly this purpose) and m_vertexShader (a generic fullscreen
	// triangle, same interface every pipeline in this filter already uses).
	// Writing this analytical result into the SAME m_mvImage the other two
	// sources write to means the resolve pass and DLAA need zero changes to
	// consume it - Apply() just picks which source runs each frame. Static
	// objects (sampler/descriptor set layout/pipeline layout/pool) are created
	// in CreateStaticObjects, the pipeline itself in CreateSizedObjects (needs
	// m_mvRenderPass), same split as the rest of this filter's pipelines.
	// Returns true if the analytical pass ran and wrote m_mvImage this frame
	// (camera locked + depth available) - Apply() falls back to block-matching/
	// optical flow when this returns false
	bool ApplyAnalyticalMV(VulkanRenderer* renderer, VkCommandBuffer cmd);

	struct AnalyticalMVPushConstants
	{
		float invVPCurrent[16]; // clip (current frame) -> world, column-major (matches VulkanRenderer::m_camCachedInvVPColMajor)
		float vpPrevious[16];   // world -> clip (previous frame), column-major (matches m_camPrevVPColMajor)
		// 2026-07-14: added after first in-game test - the game's OWN depth buffer
		// is rendered WITH clip-space jitter (see LatteTAA::Config::clipSpaceJitter),
		// but this pass's own fullscreen triangle is not jittered, so sampling
		// texDepth at this pass's raw passUV reads the WRONG texel of the jittered
		// buffer (off by the sub-pixel jitter offset) - small per-pixel errors
		// invisible to this filter's own resolve (protected by variance clipping)
		// but not to DLAA's NGX evaluate, which has no equivalent safety net (same
		// failure mode already documented for the block-matching search, see
		// mvRegularization's comment). Same unjitter idiom as every other pass
		// in this filter: sample depth at passUV + jitterUV
		float jitterUVx;
		float jitterUVy;
		// hybrid per-pixel selection (2026-07-15): the estimator's vector wins only
		// when it disagrees with the analytical one by more than disagreePx FULL-RES
		// pixels (must sit above the block-matcher's own quantization, ~2.4px at 4K)
		// AND its match cost (B channel) is below estCostMax (a confident track).
		// Everything else - agreement, low-texture ambiguity, estimator noise -
		// resolves to the analytical vector, which is exact for the static world.
		// outputWidth/Height convert the UV-space disagreement to those pixels
		float disagreePx;
		float estCostMax;
		float outputWidth;
		float outputHeight;
	};

	RendererShaderVk* m_analyticalMVShader{};
	VkDescriptorSetLayout m_analyticalMVDescriptorSetLayout{ VK_NULL_HANDLE };
	VkPipelineLayout m_analyticalMVPipelineLayout{ VK_NULL_HANDLE };
	VkPipeline m_pipelineAnalyticalMV{ VK_NULL_HANDLE };
	VKRObjectSampler* m_analyticalMVSampler{};
	VkDescriptorPool m_analyticalMVDescriptorPool{ VK_NULL_HANDLE };
	static constexpr uint32 kAnalyticalMVRingSize = 8;
	VkDescriptorSet m_analyticalMVDescriptorRing[kAnalyticalMVRingSize]{};
	uint32 m_analyticalMVDescriptorRingIndex{ 0 };

	// hybrid output target (2026-07-15, RTX-Remix-style per-pixel MV selection):
	// the estimator (block-matching or optical flow) keeps writing m_mvImage
	// exactly as before - untouched, zero risk - and the analytical pass now READS
	// that result (plus depth) and writes the per-pixel combination here: the
	// camera-reprojection vector (exact, sub-pixel - what RTX Remix computes for
	// non-animated surfaces) wherever the world is static, the estimator's vector
	// wherever it confidently disagrees (animated characters, whose motion camera
	// reprojection cannot know). Same size/format as m_mvImage, same render pass.
	// Consumers (resolve, DLAA) read this image on frames where the hybrid ran
	// (m_mvHybridValidThisFrame), m_mvImage otherwise - see
	// GetMotionVectorsViewIfValid and the resolve's descriptor write
	VkImage m_mvFinalImage{ VK_NULL_HANDLE };
	VkDeviceMemory m_mvFinalMemory{ VK_NULL_HANDLE };
	VKRObjectTexture* m_mvFinalTexObj{};
	VKRObjectTextureView* m_mvFinalViewObj{};
	VKRObjectFramebuffer* m_mvFinalFramebuffer{};
	VkImageLayout m_mvFinalLayout{ VK_IMAGE_LAYOUT_UNDEFINED };
	bool m_mvHybridValidThisFrame{ false };

	// depth cache - same two-tier (frame-bind / cross-frame) fallback pattern as
	// VulkanSSAOFilter/VulkanDLSSFilter's own independent copies, see their
	// comments for the cutscene-camera-cut rationale
	LatteTextureView* m_cachedDepthView{ nullptr };
	LatteTextureView* m_frameDepthView{ nullptr };

	// deferred destruction of outdated raw images (VKR wrappers handle their own)
	struct PendingDelete
	{
		VkImage image;
		VkDeviceMemory memory;
		VkPipeline pipeline;
		uint64 safeCmdBufferId; // destroy once VulkanRenderer::HasCommandBufferFinished(safeCmdBufferId)
		VkOpticalFlowSessionNV ofSession{ VK_NULL_HANDLE }; // appended after safeCmdBufferId so existing 4-arg brace-inits stay valid
	};
	std::vector<PendingDelete> m_pendingDeletes;
	void TickPendingDeletes(VulkanRenderer* renderer);
};
