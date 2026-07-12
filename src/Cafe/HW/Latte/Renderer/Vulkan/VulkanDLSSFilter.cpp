#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanDLSSFilter.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanTAAFilter.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/LatteTextureVk.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/LatteTextureViewVk.h"
#include "Cafe/HW/Latte/Core/LatteTAA.h"
#include "Cafe/HW/Latte/Core/LatteDLSS.h"
#include "Cafe/HW/Latte/Core/LatteCachedFBO.h"
#include "Cafe/HW/Latte/Core/Latte.h"

// only ever compiled on Windows (see src/Cafe/CMakeLists.txt) - the NGX SDK vendored
// under dependencies/DLSS only ships Windows_x86_64 binaries in this integration
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

VulkanDLSSFilter& VulkanDLSSFilter::GetInstance()
{
	static VulkanDLSSFilter s_instance;
	return s_instance;
}

uint32 VulkanDLSSFilter::FindMemoryType(VkPhysicalDevice physDev, uint32 typeBits, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);
	for (uint32 i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeBits & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	cemuLog_log(LogType::Force, "VulkanDLSSFilter: no suitable memory type found");
	return 0;
}

static void _dlssBarrierImage(VkCommandBuffer cmd, VkImage image,
							  VkImageLayout oldLayout, VkImageLayout newLayout,
							  VkAccessFlags srcAccess, VkAccessFlags dstAccess,
							  VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
							  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
							  uint32 baseMip = 0, uint32 baseLayer = 0)
{
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = aspect;
	barrier.subresourceRange.baseMipLevel = baseMip;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = baseLayer;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = srcAccess;
	barrier.dstAccessMask = dstAccess;
	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// GUID-like project id - see VulkanRenderer::InitNGX for why this exists instead of an
// NVIDIA-issued application id. Kept as a single source of truth here too since the
// create-feature call re-validates the NGX session is still the one InitNGX opened.
static const char* kNGXProjectId = "a63f6c8e-2b3d-4e91-9c7a-5f1d8b0e2f4c";

void VulkanDLSSFilter::ReleaseResources(VulkanRenderer* renderer)
{
	m_hasValidOutput = false;
	m_historyValid = false;

	if (m_dlssHandle)
	{
		NVSDK_NGX_VULKAN_ReleaseFeature(m_dlssHandle);
		m_dlssHandle = nullptr;
	}

	uint64 releaseCmdBufferId = renderer->GetCurrentCommandBufferId();
	if (m_image != VK_NULL_HANDLE)
	{
		m_pendingDeletes.push_back({ m_image, m_memory, releaseCmdBufferId });
		m_image = VK_NULL_HANDLE;
		m_memory = VK_NULL_HANDLE;
	}
	if (m_viewObj)
	{
		renderer->ReleaseDestructibleObject(m_viewObj);
		m_viewObj = nullptr;
	}
	if (m_texObj)
	{
		m_texObj->m_image = VK_NULL_HANDLE;
		renderer->ReleaseDestructibleObject(m_texObj);
		m_texObj = nullptr;
	}
	m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanDLSSFilter::TickPendingDeletes(VulkanRenderer* renderer)
{
	VkDevice device = renderer->GetLogicalDevice();
	for (auto it = m_pendingDeletes.begin(); it != m_pendingDeletes.end();)
	{
		if (!renderer->HasCommandBufferFinished(it->safeCmdBufferId))
		{
			++it;
			continue;
		}
		if (it->image != VK_NULL_HANDLE)
			vkDestroyImage(device, it->image, nullptr);
		if (it->memory != VK_NULL_HANDLE)
			vkFreeMemory(device, it->memory, nullptr);
		it = m_pendingDeletes.erase(it);
	}
}

bool VulkanDLSSFilter::RecreateIfNeeded(VulkanRenderer* renderer, sint32 width, sint32 height, VkFormat format)
{
	if (width == m_width && height == m_height && format == m_format && m_image != VK_NULL_HANDLE)
		return true;

	ReleaseResources(renderer);
	m_width = width;
	m_height = height;
	m_format = format;

	VkDevice device = renderer->GetLogicalDevice();

	if (!m_sampler)
	{
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		m_sampler = VKRObjectSampler::GetOrCreateSampler(&samplerInfo);
		if (!m_sampler)
		{
			cemuLog_log(LogType::Force, "VulkanDLSSFilter: failed to create sampler");
			return false;
		}
		m_sampler->incRef();
	}

	if (!m_ngxParams)
	{
		if (NVSDK_NGX_VULKAN_AllocateParameters(&m_ngxParams) != NVSDK_NGX_Result_Success || !m_ngxParams)
		{
			cemuLog_log(LogType::Force, "VulkanDLSSFilter: NVSDK_NGX_VULKAN_AllocateParameters failed");
			m_ngxParams = nullptr;
			m_width = 0; m_height = 0; m_format = VK_FORMAT_UNDEFINED;
			return false;
		}
	}

	// output image: NGX writes to this internally (compute-based), so it needs
	// STORAGE in addition to SAMPLED (for our own later reads - present, SSAO chain)
	VkImageCreateInfo imgInfo{};
	imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imgInfo.imageType = VK_IMAGE_TYPE_2D;
	imgInfo.format = m_format;
	imgInfo.extent = { (uint32)m_width, (uint32)m_height, 1 };
	imgInfo.mipLevels = 1;
	imgInfo.arrayLayers = 1;
	imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	if (vkCreateImage(device, &imgInfo, nullptr, &m_image) != VK_SUCCESS)
	{
		cemuLog_log(LogType::Force, "VulkanDLSSFilter: failed to create output image ({}x{})", width, height);
		m_width = 0; m_height = 0; m_format = VK_FORMAT_UNDEFINED;
		return false;
	}

	VkMemoryRequirements memReq;
	vkGetImageMemoryRequirements(device, m_image, &memReq);
	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memReq.size;
	allocInfo.memoryTypeIndex = FindMemoryType(renderer->GetPhysicalDevice(), memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (vkAllocateMemory(device, &allocInfo, nullptr, &m_memory) != VK_SUCCESS)
	{
		vkDestroyImage(device, m_image, nullptr);
		m_image = VK_NULL_HANDLE;
		m_width = 0; m_height = 0; m_format = VK_FORMAT_UNDEFINED;
		return false;
	}
	vkBindImageMemory(device, m_image, m_memory, 0);

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = m_format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;
	VkImageView rawView;
	if (vkCreateImageView(device, &viewInfo, nullptr, &rawView) != VK_SUCCESS)
	{
		vkDestroyImage(device, m_image, nullptr);
		m_image = VK_NULL_HANDLE;
		m_width = 0; m_height = 0; m_format = VK_FORMAT_UNDEFINED;
		return false;
	}

	m_texObj = new VKRObjectTexture();
	m_texObj->m_image = m_image;
	m_texObj->m_format = m_format;
	m_texObj->m_imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;
	m_viewObj = new VKRObjectTextureView(m_texObj, rawView);
	m_layout = VK_IMAGE_LAYOUT_UNDEFINED;

	return true;
}

void VulkanDLSSFilter::Apply(VulkanRenderer* renderer, LatteTextureViewVk* scanoutView)
{
	auto& config = LatteDLSS::GetConfig();
	if (!config.enabled || !renderer->IsDLAAAvailable())
		return;

	// same anti-reprocessing guards as VulkanTAAFilter::Apply - see its comments.
	// DLAA runs after TAA in the chain (needs its motion vectors), so it shares the
	// exact same scene/draw counters TAA already advanced this frame
	if (LatteGPUState.drawCallCounter == m_lastDrawCallCounter)
		return;
	const uint32 drawsSinceResolve = LatteGPUState.drawCallCounter - m_lastDrawCallCounter;
	const uint32 sceneDrawCounter = LatteTAA::GetSceneDrawCounter();
	if (sceneDrawCounter == m_lastSceneDrawCounter && drawsSinceResolve < 200)
	{
		m_lastDrawCallCounter = LatteGPUState.drawCallCounter;
		m_consecutiveSceneless++;
		if (m_consecutiveSceneless > 30)
			m_hasValidOutput = false;
		return;
	}
	m_consecutiveSceneless = 0;

	LatteTextureVk* scanoutTex = (LatteTextureVk*)scanoutView->baseTexture;
	const uint32 scanMip = (uint32)scanoutView->firstMip;
	const uint32 scanSlice = (uint32)scanoutView->firstSlice;

	sint32 effWidth = 0, effHeight = 0;
	scanoutTex->GetEffectiveSize(effWidth, effHeight, scanoutView->firstMip);
	if (effWidth < 2 || effHeight < 2)
		return;

	// TEMP diagnostic (2026-07-12): confirm which branch Apply() actually takes each
	// state change, so a silent early-return can't masquerade as "DLAA working" just
	// because the fallback (TAA's own resolve) also looks clean. Remove once trusted.
	static const char* s_diagLastState = "";
	auto diagState = [&](const char* state)
	{
		if (state != s_diagLastState)
		{
			cemuLog_log(LogType::Force, "DLAA diag: state -> {}", state);
			s_diagLastState = state;
		}
	};

	// DLAA needs real motion vectors - reuse TAA's block-matching search instead of
	// running a second one. Only produced in temporal mode (see GetMotionVectorsViewIfValid).
	sint32 mvWidth = 0, mvHeight = 0;
	VkImage mvImage = VK_NULL_HANDLE;
	VkImageView mvView = VulkanTAAFilter::GetInstance().GetMotionVectorsViewIfValid(mvWidth, mvHeight, mvImage);
	if (mvView == VK_NULL_HANDLE || mvImage == VK_NULL_HANDLE)
	{
		diagState("no motion vectors (TAA temporal mode not producing them)");
		m_hasValidOutput = false;
		return;
	}

	// depth: same 3-tier fallback as VulkanSSAOFilter (current bind / this-frame bind /
	// cross-frame cache) - kept as an independent cache rather than reading SSAO's, since
	// SSAO can be toggled off while DLAA stays on. See VulkanSSAOFilter::Apply for why
	// this exists (cutscene camera cuts swap in brand-new depth buffers).
	LatteTextureView* depthView = LatteMRT::GetDepthAttachment();
	LatteTextureVk* depthTexVk = nullptr;
	bool haveDepth = false;
	if (depthView && depthView->baseTexture)
	{
		depthTexVk = (LatteTextureVk*)depthView->baseTexture;
		sint32 dw = 0, dh = 0;
		depthTexVk->GetEffectiveSize(dw, dh, depthView->firstMip);
		if (dw == effWidth && dh == effHeight)
		{
			haveDepth = true;
			m_cachedDepthView = depthView;
		}
	}
	if (!haveDepth && m_frameDepthView && m_frameDepthView->baseTexture)
	{
		LatteTextureVk* frameTexVk = (LatteTextureVk*)m_frameDepthView->baseTexture;
		sint32 fw = 0, fh = 0;
		frameTexVk->GetEffectiveSize(fw, fh, m_frameDepthView->firstMip);
		if (fw == effWidth && fh == effHeight)
		{
			depthView = m_frameDepthView;
			depthTexVk = frameTexVk;
			haveDepth = true;
			m_cachedDepthView = depthView;
		}
	}
	if (!haveDepth && m_cachedDepthView && m_cachedDepthView->baseTexture)
	{
		LatteTextureVk* cachedTexVk = (LatteTextureVk*)m_cachedDepthView->baseTexture;
		sint32 cw = 0, ch = 0;
		cachedTexVk->GetEffectiveSize(cw, ch, m_cachedDepthView->firstMip);
		if (cw == effWidth && ch == effHeight)
		{
			depthView = m_cachedDepthView;
			depthTexVk = cachedTexVk;
			haveDepth = true;
		}
	}
	if (!haveDepth)
	{
		diagState("no depth buffer found");
		m_hasValidOutput = false;
		return;
	}
	LatteTextureViewVk* depthViewVk = (LatteTextureViewVk*)depthView;

	if (!RecreateIfNeeded(renderer, effWidth, effHeight, scanoutTex->GetFormat()))
	{
		diagState("RecreateIfNeeded failed (see preceding log line for the reason)");
		m_hasValidOutput = false;
		return;
	}

	VkDevice device = renderer->GetLogicalDevice();
	TickPendingDeletes(renderer);

	// caller (VulkanRenderer::DLAA_Apply) has already closed any pending render pass
	VkCommandBuffer cmd = renderer->DLAA_GetCommandBuffer();

	m_viewObj->flagForCurrentCommandBuffer();
	m_sampler->flagForCurrentCommandBuffer();

	// color input: raw jittered scanout, untouched - NGX de-jitters internally using
	// the jitter offset below, unlike our own hand-rolled resolve which manually shifts
	// the sample position (see VulkanTAAFilter::Apply's jitterUVx/y comment)
	VkImage scanoutImage = scanoutTex->GetImageObj()->m_image;
	VkImageSubresource scanoutSubres{ VK_IMAGE_ASPECT_COLOR_BIT, scanMip, scanSlice };
	VkImageLayout scanoutLayout = scanoutTex->GetImageLayout(scanoutSubres);
	_dlssBarrierImage(cmd, scanoutImage, scanoutLayout, VK_IMAGE_LAYOUT_GENERAL,
					  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					  VK_IMAGE_ASPECT_COLOR_BIT, scanMip, scanSlice);
	VkImageSubresourceRange scanoutRange{ VK_IMAGE_ASPECT_COLOR_BIT, scanMip, 1, scanSlice, 1 };
	scanoutTex->SetImageLayout(scanoutRange, VK_IMAGE_LAYOUT_GENERAL);
	VKRObjectTextureView* scanoutViewObj = scanoutView->GetViewRGBA();
	scanoutViewObj->flagForCurrentCommandBuffer();

	// depth input
	LatteTextureVk* depthTexVkFinal = (LatteTextureVk*)depthViewVk->baseTexture;
	const uint32 depthMip = (uint32)depthViewVk->firstMip;
	const uint32 depthSlice = (uint32)depthViewVk->firstSlice;
	VkImage depthImage = depthTexVkFinal->GetImageObj()->m_image;
	VkImageSubresource depthSubres{ VK_IMAGE_ASPECT_DEPTH_BIT, depthMip, depthSlice };
	VkImageLayout depthLayout = depthTexVkFinal->GetImageLayout(depthSubres);
	_dlssBarrierImage(cmd, depthImage, depthLayout, VK_IMAGE_LAYOUT_GENERAL,
					  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					  VK_IMAGE_ASPECT_DEPTH_BIT, depthMip, depthSlice);
	VkImageSubresourceRange depthRange{ VK_IMAGE_ASPECT_DEPTH_BIT, depthMip, 1, depthSlice, 1 };
	depthTexVkFinal->SetImageLayout(depthRange, VK_IMAGE_LAYOUT_GENERAL);
	VKRObjectTextureView* depthViewObj = depthViewVk->GetViewRGBA();
	depthViewObj->flagForCurrentCommandBuffer();

	// output: about to be fully overwritten, no need to preserve prior contents
	_dlssBarrierImage(cmd, m_image, m_layout, VK_IMAGE_LAYOUT_GENERAL,
					  0, VK_ACCESS_SHADER_WRITE_BIT,
					  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	m_layout = VK_IMAGE_LAYOUT_GENERAL;

	float jitterPxX = 0.0f, jitterPxY = 0.0f;
	LatteTAA::GetCurrentFrameJitter(jitterPxX, jitterPxY);

	// TEMP diagnostic (2026-07-12): user reports "no antialiasing effect at all" with
	// DLAA on - confirm jitter is actually reaching NGX (varying, non-zero) before
	// suspecting motion vector quality/sign convention. Throttled to ~1/sec at 60fps.
	{
		static uint32 s_diagFrameCounter = 0;
		if ((s_diagFrameCounter++ % 60) == 0)
			cemuLog_log(LogType::Force, "DLAA diag: jitter=({:.3f},{:.3f})px reset={} size={}x{}",
				jitterPxX, jitterPxY, m_historyValid ? 0 : 1, effWidth, effHeight);
	}

	// our own dedicated images (output, motion vectors) are always mip0/slice0;
	// the scanout/depth are the game's textures and use their actual subresource
	// (scanoutRange/depthRange, built above) - they can be aliased/non-zero
	VkImageSubresourceRange ownedSubresRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

	NVSDK_NGX_Resource_VK colorRes = NVSDK_NGX_Create_ImageView_Resource_VK(
		scanoutViewObj->m_textureImageView, scanoutImage, scanoutRange, m_format, (uint32)effWidth, (uint32)effHeight, false);
	NVSDK_NGX_Resource_VK depthRes = NVSDK_NGX_Create_ImageView_Resource_VK(
		depthViewObj->m_textureImageView, depthImage, depthRange, depthTexVkFinal->GetFormat(), (uint32)effWidth, (uint32)effHeight, false);
	NVSDK_NGX_Resource_VK mvRes = NVSDK_NGX_Create_ImageView_Resource_VK(
		mvView, mvImage, ownedSubresRange, VK_FORMAT_R16G16B16A16_SFLOAT, (uint32)mvWidth, (uint32)mvHeight, false);
	NVSDK_NGX_Resource_VK outputRes = NVSDK_NGX_Create_ImageView_Resource_VK(
		m_viewObj->m_textureImageView, m_image, ownedSubresRange, m_format, (uint32)effWidth, (uint32)effHeight, true);

	if (!m_dlssHandle)
	{
		NVSDK_NGX_DLSS_Create_Params createParams{};
		createParams.Feature.InWidth = (uint32)effWidth;
		createParams.Feature.InHeight = (uint32)effHeight;
		createParams.Feature.InTargetWidth = (uint32)effWidth;
		createParams.Feature.InTargetHeight = (uint32)effHeight;
		createParams.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
		createParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

		NVSDK_NGX_Result result = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &m_dlssHandle, m_ngxParams, &createParams);
		if (result != NVSDK_NGX_Result_Success || !m_dlssHandle)
		{
			cemuLog_log(LogType::Force, "VulkanDLSSFilter: NGX_VULKAN_CREATE_DLSS_EXT failed (result={:#x})", (uint32)result);
			m_dlssHandle = nullptr;
			m_hasValidOutput = false;
			return;
		}
		m_historyValid = false;
	}

	NVSDK_NGX_VK_DLSS_Eval_Params evalParams{};
	evalParams.Feature.pInColor = &colorRes;
	evalParams.Feature.pInOutput = &outputRes;
	evalParams.pInDepth = &depthRes;
	evalParams.pInMotionVectors = &mvRes;
	evalParams.InJitterOffsetX = jitterPxX;
	evalParams.InJitterOffsetY = jitterPxY;
	evalParams.InRenderSubrectDimensions = { (uint32)effWidth, (uint32)effHeight };
	evalParams.InReset = m_historyValid ? 0 : 1;
	// our motion vectors are UV-space fractional offsets (see VulkanTAAFilter's mv shader);
	// NGX wants pixel space, hence this scale (UV_offset * dimension = pixel_offset)
	evalParams.InMVScaleX = (float)effWidth;
	evalParams.InMVScaleY = (float)effHeight;

	NVSDK_NGX_Result evalResult = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, m_dlssHandle, m_ngxParams, &evalParams);
	if (evalResult != NVSDK_NGX_Result_Success)
	{
		cemuLog_log(LogType::Force, "VulkanDLSSFilter: NGX_VULKAN_EVALUATE_DLSS_EXT failed (result={:#x})", (uint32)evalResult);
		m_hasValidOutput = false;
		return;
	}

	// game's scanout/depth textures are left untouched (read-only, same rule as
	// TAA/SSAO); keep Cemu's layout tracker coherent with the read barriers above
	depthTexVkFinal->SetImageLayout(depthRange, VK_IMAGE_LAYOUT_GENERAL);
	scanoutTex->SetImageLayout(scanoutRange, VK_IMAGE_LAYOUT_GENERAL);

	// make the output visible to the backbuffer blit / SSAO's fragment sampling
	_dlssBarrierImage(cmd, m_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
					  VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	diagState("evaluating (NGX evaluate succeeded, output presented)");
	m_hasValidOutput = true;
	m_historyValid = true;
	m_lastDrawCallCounter = LatteGPUState.drawCallCounter;
	m_lastSceneDrawCounter = sceneDrawCounter;
}

VkDescriptorSet VulkanDLSSFilter::GetPresentDescriptorSet(VulkanRenderer* renderer, VkDescriptorSetLayout blitLayout,
														   sint32 expectedWidth, sint32 expectedHeight)
{
	if (!m_hasValidOutput || expectedWidth != m_width || expectedHeight != m_height)
		return VK_NULL_HANDLE;
	if (!m_sampler || !m_viewObj)
		return VK_NULL_HANDLE;
	VkDevice device = renderer->GetLogicalDevice();

	if (m_presentDescriptorPool == VK_NULL_HANDLE)
	{
		VkDescriptorPoolSize poolSize{};
		poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSize.descriptorCount = kPresentRingSize;
		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.maxSets = kPresentRingSize;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_presentDescriptorPool) != VK_SUCCESS)
			return VK_NULL_HANDLE;
		VkDescriptorSetLayout layouts[kPresentRingSize];
		for (uint32 i = 0; i < kPresentRingSize; i++)
			layouts[i] = blitLayout;
		VkDescriptorSetAllocateInfo dsAlloc{};
		dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		dsAlloc.descriptorPool = m_presentDescriptorPool;
		dsAlloc.descriptorSetCount = kPresentRingSize;
		dsAlloc.pSetLayouts = layouts;
		if (vkAllocateDescriptorSets(device, &dsAlloc, m_presentRing) != VK_SUCCESS)
		{
			vkDestroyDescriptorPool(device, m_presentDescriptorPool, nullptr);
			m_presentDescriptorPool = VK_NULL_HANDLE;
			return VK_NULL_HANDLE;
		}
	}

	m_viewObj->flagForCurrentCommandBuffer();
	m_sampler->flagForCurrentCommandBuffer();

	VkDescriptorSet descSet = m_presentRing[m_presentRingIndex];
	m_presentRingIndex = (m_presentRingIndex + 1) % kPresentRingSize;

	VkDescriptorImageInfo imageInfo{};
	imageInfo.sampler = m_sampler->GetSampler();
	imageInfo.imageView = m_viewObj->m_textureImageView;
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = descSet;
	write.dstBinding = 0;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	write.pImageInfo = &imageInfo;
	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
	return descSet;
}

VkImageView VulkanDLSSFilter::GetResolvedImageViewIfValid(sint32 expectedWidth, sint32 expectedHeight)
{
	if (!m_hasValidOutput || expectedWidth != m_width || expectedHeight != m_height)
		return VK_NULL_HANDLE;
	if (!m_viewObj)
		return VK_NULL_HANDLE;
	m_viewObj->flagForCurrentCommandBuffer();
	return m_viewObj->m_textureImageView;
}

void VulkanDLSSFilter::NotifyTextureDeletion(LatteTexture* texture)
{
	if (m_cachedDepthView && m_cachedDepthView->baseTexture == texture)
		m_cachedDepthView = nullptr;
	if (m_frameDepthView && m_frameDepthView->baseTexture == texture)
		m_frameDepthView = nullptr;
}

void VulkanDLSSFilter::NotifyDepthBind(LatteTextureView* view)
{
	if (!view || !view->baseTexture || m_width <= 0)
		return;
	sint32 w = 0, h = 0;
	view->baseTexture->GetEffectiveSize(w, h, view->firstMip);
	if (w == m_width && h == m_height)
		m_frameDepthView = view;
}

void VulkanDLSSFilter::Shutdown(VulkanRenderer* renderer)
{
	m_cachedDepthView = nullptr;
	m_frameDepthView = nullptr;
	ReleaseResources(renderer);
	VkDevice device = renderer->GetLogicalDevice();
	if (m_ngxParams)
	{
		NVSDK_NGX_VULKAN_DestroyParameters(m_ngxParams);
		m_ngxParams = nullptr;
	}
	for (auto& pd : m_pendingDeletes)
	{
		if (pd.image != VK_NULL_HANDLE)
			vkDestroyImage(device, pd.image, nullptr);
		if (pd.memory != VK_NULL_HANDLE)
			vkFreeMemory(device, pd.memory, nullptr);
	}
	m_pendingDeletes.clear();
	if (m_presentDescriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, m_presentDescriptorPool, nullptr);
		m_presentDescriptorPool = VK_NULL_HANDLE;
	}
	if (m_sampler)
	{
		m_sampler->decRef();
		m_sampler = nullptr;
	}
}

// member of VulkanRenderer so the filter can be driven from common Latte code
// (same idiom as VulkanRenderer::TAA_Apply / SSAO_Apply)
void VulkanRenderer::DLAA_Apply(LatteTextureView* textureView)
{
	if (!textureView || !textureView->baseTexture)
		return;
	LatteTextureVk* texVk = (LatteTextureVk*)textureView->baseTexture;
	if (texVk->isDepth)
		return;
	draw_endRenderPass();
	VulkanDLSSFilter::GetInstance().Apply(this, (LatteTextureViewVk*)textureView);
	m_state.currentPipeline = VK_NULL_HANDLE;
	vkCmdSetViewport(m_state.currentCommandBuffer, 0, 1, &m_state.currentViewport);
	vkCmdSetScissor(m_state.currentCommandBuffer, 0, 1, &m_state.currentScissorRect);
}

VkCommandBuffer VulkanRenderer::DLAA_GetCommandBuffer()
{
	return getCurrentCommandBuffer();
}
