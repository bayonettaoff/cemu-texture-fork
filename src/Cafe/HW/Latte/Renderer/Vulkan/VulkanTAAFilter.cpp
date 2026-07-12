#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanTAAFilter.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/LatteTextureVk.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/LatteTextureViewVk.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/RendererShaderVk.h"
#include "Cafe/HW/Latte/Core/LatteTAA.h"
#include "Cafe/HW/Latte/Core/Latte.h"

VulkanTAAFilter& VulkanTAAFilter::GetInstance()
{
	static VulkanTAAFilter s_instance;
	return s_instance;
}

uint32 VulkanTAAFilter::FindMemoryType(VkPhysicalDevice physDev, uint32 typeBits, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);
	for (uint32 i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeBits & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	cemuLog_log(LogType::Force, "VulkanTAAFilter: no suitable memory type found");
	return 0;
}

static void _taaBarrierImage(VkCommandBuffer cmd, VkImage image,
							 VkImageLayout oldLayout, VkImageLayout newLayout,
							 VkAccessFlags srcAccess, VkAccessFlags dstAccess,
							 VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
							 uint32 baseMip = 0, uint32 baseLayer = 0)
{
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = baseMip;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = baseLayer;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = srcAccess;
	barrier.dstAccessMask = dstAccess;
	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool VulkanTAAFilter::CreateShaders()
{
	const char* vsSrc =
		"#version 450\r\n"
		"layout(location = 0) out vec2 passUV;\r\n"
		"void main(){\r\n"
		"vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));\r\n"
		"passUV = pos;\r\n"
		"gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);\r\n"
		"}\r\n";

	const char* fsSrc =
		"#version 450\r\n"
		"layout(location = 0) in vec2 passUV;\r\n"
		"layout(location = 0) out vec4 outColor;\r\n"
		"layout(binding = 0) uniform sampler2D texCurrent;\r\n"
		"layout(binding = 1) uniform sampler2D texHistory;\r\n"
		"layout(binding = 2) uniform sampler2D texMotionVectors;\r\n" // RG = UV offset from this frame's content to where it was last frame
		"layout(push_constant) uniform pushConstants {\r\n"
		"float blendFactor;\r\n"
		"float historyValid;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float passthrough;\r\n"
		"float mvSearchStep;\r\n"
		"float mvRegularization;\r\n"
		"float useMotionVectors;\r\n"
		"float mvDebugView;\r\n"
		"}uf_pc;\r\n"
		"void main(){\r\n"
		// sample the current frame at the unjittered position: the viewport shifted
		// scene content by +jitter pixels, so the content for this pixel sits at +jitterUV
		"vec2 curUV = passUV + vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"vec4 cur = texture(texCurrent, curUV);\r\n"
		// variance clipping (Salvi): a min/max box over the 3x3 neighborhood is far
		// too wide in high-contrast regions (bloom, specular edges) and lets a
		// misaligned previous frame through as a hard double image. mean +/- one
		// standard deviation stays tight there while still allowing temporal
		// smoothing in flat regions.
		"vec3 m1 = cur.rgb;\r\n"
		"vec3 m2 = cur.rgb * cur.rgb;\r\n"
		"vec3 n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2(-1,-1)).rgb; m1 += n; m2 += n*n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2( 0,-1)).rgb; m1 += n; m2 += n*n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2( 1,-1)).rgb; m1 += n; m2 += n*n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2(-1, 0)).rgb; m1 += n; m2 += n*n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2( 1, 0)).rgb; m1 += n; m2 += n*n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2(-1, 1)).rgb; m1 += n; m2 += n*n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2( 0, 1)).rgb; m1 += n; m2 += n*n;\r\n"
		"n = textureOffset(texCurrent, curUV, ivec2( 1, 1)).rgb; m1 += n; m2 += n*n;\r\n"
		"m1 *= (1.0/9.0);\r\n"
		"m2 *= (1.0/9.0);\r\n"
		"vec3 sigma = sqrt(max(m2 - m1*m1, vec3(0.0)));\r\n"
		"vec3 cmin = m1 - sigma;\r\n"
		"vec3 cmax = m1 + sigma;\r\n"
		// reproject history through the estimated motion vector (see the search
		// pass below) instead of sampling it at the same screen position every
		// frame - real TAA reprojection instead of pure rejection. Falls back to
		// the old static sample when disabled (A/B testing, or a game/scene
		// where the search does more harm than good)
		"vec2 mv = (uf_pc.useMotionVectors > 0.5) ? texture(texMotionVectors, passUV).rg : vec2(0.0);\r\n"
		// visualize the raw search output regardless of useMotionVectors, so this
		// stays useful even while reprojection itself is toggled off. RG: neutral
		// gray (0.5, 0.5) = no motion found here; scaled 10x so typical small
		// UV-space motion is actually visible instead of reading as uniform gray
		// either way. B: the search's own best-match cost (already 0..1-ish, no
		// rescale needed) - if this shows real scene contrast while RG stays flat
		// gray, the pass IS running and computing real per-pixel costs, so the
		// search LOOP itself never improved past the zero-motion seed; if B is
		// ALSO uniformly near-zero everywhere, current and history are already
		// nearly identical at every pixel (a real, if surprising, possibility)
		"if (uf_pc.mvDebugView > 0.5){\r\n"
		"vec3 mvRaw = texture(texMotionVectors, passUV).rgb;\r\n"
		"vec2 mvVis = clamp(mvRaw.xy * 10.0 + vec2(0.5), vec2(0.0), vec2(1.0));\r\n"
		"outColor = vec4(mvVis, clamp(mvRaw.z, 0.0, 1.0), 1.0);\r\n"
		"return;\r\n"
		"}\r\n"
		"vec3 histRaw = texture(texHistory, passUV + mv).rgb;\r\n"
		"vec3 hist = clamp(histRaw, cmin, cmax);\r\n"
		// history far outside the current neighborhood means it is stale
		// (occlusion change, cut, a skipped draw, or the motion search missing) ->
		// converge to current fast. Unchanged by reprojection: still the safety
		// net for wrong/low-confidence motion vectors, same as it always was for
		// a wrong static sample
		"float clampDist = length(histRaw - hist);\r\n"
		"float w = clamp(uf_pc.blendFactor + clampDist * 4.0, 0.0, 1.0);\r\n"
		"vec3 resolved = mix(hist, cur.rgb, w);\r\n"
		"float keepHistory = uf_pc.historyValid * (1.0 - uf_pc.passthrough);\r\n"
		"outColor = vec4(mix(cur.rgb, resolved, keepHistory), cur.a);\r\n"
		"}\r\n";

	// motion vector search: for each half-res output tile, finds the UV offset
	// where texHistory best matches texCurrent at this pixel - a hierarchical
	// block search (the technique video codecs use for motion estimation: an
	// iteratively refined step size instead of one exhaustive search), since
	// GX2 exposes no per-vertex motion data to read directly (same constraint
	// behind every other approximation in this fork). Luma-only cost function
	// on purpose for this first version: no depth plumbing into this filter yet
	// (TAA has never needed depth before this), and regularizing toward zero
	// motion already stabilizes flat/low-texture regions without it
	const char* mvSrc =
		"#version 450\r\n"
		"layout(location = 0) in vec2 passUV;\r\n"
		"layout(location = 0) out vec4 outColor;\r\n"
		"layout(binding = 0) uniform sampler2D texCurrent;\r\n"
		"layout(binding = 1) uniform sampler2D texHistory;\r\n"
		"layout(push_constant) uniform pushConstants {\r\n"
		"float blendFactor;\r\n"
		"float historyValid;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float passthrough;\r\n"
		"float mvSearchStep;\r\n"
		"float mvRegularization;\r\n"
		"float useMotionVectors;\r\n"
		"float mvDebugView;\r\n"
		"}uf_pc;\r\n"
		"const vec3 LUMA = vec3(0.299, 0.587, 0.114);\r\n"
		"void main(){\r\n"
		// same unjitter as the resolve, so both sides of the comparison are in
		// stable content-space (history is already unjittered - it is itself a
		// past resolve's output)
		"vec2 uvC = passUV + vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"float curLuma = dot(texture(texCurrent, uvC).rgb, LUMA);\r\n"
		"vec2 best = vec2(0.0);\r\n"
		"float bestCost = abs(dot(texture(texHistory, passUV).rgb, LUMA) - curLuma);\r\n"
		"float stepSize = uf_pc.mvSearchStep;\r\n"
		// per-pixel rotation of the 8 fixed search directions (interleaved
		// gradient noise, same trick the HBAO/SSR pass in this fork already
		// uses for exactly this reason): without it, every pixel probes the
		// SAME 8 axis-aligned directions, so a smoothly varying luma gradient
		// across a surface "snaps" to a different one of those 8 quantized
		// directions in a spatially regular way - a moire/banding stripe
		// pattern, not real motion (confirmed via the debug view: dense,
		// perfectly regular diagonal stripes, not per-pixel noise)
		"float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));\r\n"
		"float rotation = ign * 6.28318530718;\r\n"
		// 6 iterations of "test 8 neighbors around the current best, keep the
		// winner, halve the step": ~2x the initial step in total reach, same
		// tap cost (48 candidates) regardless of how far the true motion is
		"for (int iter = 0; iter < 6; iter++){\r\n"
		"vec2 newBest = best;\r\n"
		"float newCost = bestCost;\r\n"
		"for (int i = 0; i < 8; i++){\r\n"
		"float ang = float(i) * 0.78539816340 + rotation;\r\n" // PI/4 steps, rotated per pixel
		"vec2 cand = best + stepSize * vec2(cos(ang), sin(ang));\r\n"
		"float hLuma = dot(texture(texHistory, passUV + cand).rgb, LUMA);\r\n"
		// regularization: ties (common in flat regions) resolve toward less
		// motion instead of noise, without needing a real confidence signal
		"float cost = abs(hLuma - curLuma) + uf_pc.mvRegularization * dot(cand, cand);\r\n"
		"if (cost < newCost){ newCost = cost; newBest = cand; }\r\n"
		"}\r\n"
		"best = newBest;\r\n"
		"bestCost = newCost;\r\n"
		"stepSize *= 0.5;\r\n"
		"}\r\n"
		"outColor = vec4(best, bestCost, 1.0);\r\n"
		"}\r\n";

	// single-frame FXAA (3.11-style, quality path): the default AA mode. No
	// history and no jitter, so unlike the temporal resolve it cannot corrupt
	// anything. Reuses the same descriptor layout (bindings 1/2 unused) and
	// push constant block (jitterUVx/y carry 1/width, 1/height here)
	const char* fsFxaaSrc =
		"#version 450\r\n"
		"layout(location = 0) in vec2 passUV;\r\n"
		"layout(location = 0) out vec4 outColor;\r\n"
		"layout(binding = 0) uniform sampler2D texCurrent;\r\n"
		"layout(binding = 1) uniform sampler2D texHistory;\r\n"
		"layout(push_constant) uniform pushConstants {\r\n"
		"float blendFactor;\r\n"
		"float historyValid;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float passthrough;\r\n"
		"float mvSearchStep;\r\n"
		"float mvRegularization;\r\n"
		"float useMotionVectors;\r\n"
		"float mvDebugView;\r\n"
		"}uf_pc;\r\n"
		"void main(){\r\n"
		"vec2 rcpFrame = vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"const vec3 L = vec3(0.299, 0.587, 0.114);\r\n"
		"vec4 colM = texture(texCurrent, passUV);\r\n"
		"if (uf_pc.passthrough > 0.5){ outColor = colM; return; }\r\n"
		"float lumaM = dot(colM.rgb, L);\r\n"
		"float lumaNW = dot(textureOffset(texCurrent, passUV, ivec2(-1,-1)).rgb, L);\r\n"
		"float lumaNE = dot(textureOffset(texCurrent, passUV, ivec2( 1,-1)).rgb, L);\r\n"
		"float lumaSW = dot(textureOffset(texCurrent, passUV, ivec2(-1, 1)).rgb, L);\r\n"
		"float lumaSE = dot(textureOffset(texCurrent, passUV, ivec2( 1, 1)).rgb, L);\r\n"
		"float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));\r\n"
		"float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));\r\n"
		// early out on flat regions: keeps text/UI crisp and saves bandwidth
		"if (lumaMax - lumaMin < max(0.0312, lumaMax * 0.125)){ outColor = colM; return; }\r\n"
		"vec2 dir = vec2(-((lumaNW + lumaNE) - (lumaSW + lumaSE)), ((lumaNW + lumaSW) - (lumaNE + lumaSE)));\r\n"
		"float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * 0.03125, 0.0078125);\r\n"
		"float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);\r\n"
		"dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * rcpFrame;\r\n"
		"vec3 rgbA = 0.5 * (texture(texCurrent, passUV + dir * (1.0/3.0 - 0.5)).rgb + texture(texCurrent, passUV + dir * (2.0/3.0 - 0.5)).rgb);\r\n"
		"vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(texCurrent, passUV + dir * -0.5).rgb + texture(texCurrent, passUV + dir * 0.5).rgb);\r\n"
		"float lumaB = dot(rgbB, L);\r\n"
		"outColor = vec4((lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB, colM.a);\r\n"
		"}\r\n";

	std::string vsStr(vsSrc);
	m_vertexShader = new RendererShaderVk(RendererShader::ShaderType::kVertex, 0, 0, false, false, vsStr);
	m_vertexShader->PreponeCompilation(true);

	std::string fsStr(fsSrc);
	m_fragmentShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, fsStr);
	m_fragmentShader->PreponeCompilation(true);

	std::string fsFxaaStr(fsFxaaSrc);
	m_fxaaShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, fsFxaaStr);
	m_fxaaShader->PreponeCompilation(true);

	std::string mvStr(mvSrc);
	m_mvShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, mvStr);
	m_mvShader->PreponeCompilation(true);

	return m_vertexShader != nullptr && m_fragmentShader != nullptr && m_fxaaShader != nullptr && m_mvShader != nullptr;
}

bool VulkanTAAFilter::CreateStaticObjects(VulkanRenderer* renderer)
{
	VkDevice device = renderer->GetLogicalDevice();

	// sampler (linear so the sub-pixel unjitter offset actually interpolates;
	// history is sampled at texel centers where linear == nearest)
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
		return false;
	m_sampler->incRef();

	// descriptor set layout: 0 = current, 1 = history, 2 = motion vectors
	// (shared by resolve/FXAA/motion-search - see kDescriptorBindings)
	VkDescriptorSetLayoutBinding bindings[kDescriptorBindings]{};
	for (uint32 i = 0; i < kDescriptorBindings; i++)
	{
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dslInfo{};
	dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dslInfo.bindingCount = kDescriptorBindings;
	dslInfo.pBindings = bindings;
	if (vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
		return false;

	// pipeline layout with push constants for the fragment stage
	VkPushConstantRange pcRange{};
	pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	pcRange.offset = 0;
	pcRange.size = sizeof(PushConstants);
	VkPipelineLayoutCreateInfo plInfo{};
	plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plInfo.setLayoutCount = 1;
	plInfo.pSetLayouts = &m_descriptorSetLayout;
	plInfo.pushConstantRangeCount = 1;
	plInfo.pPushConstantRanges = &pcRange;
	if (vkCreatePipelineLayout(device, &plInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
		return false;

	// descriptor pool + ring of sets, updated round-robin each frame (2 sets
	// consumed per Apply() now: the motion search pass and the resolve pass)
	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSize.descriptorCount = kDescriptorRingSize * kDescriptorBindings;
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = kDescriptorRingSize;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
		return false;

	VkDescriptorSetLayout layouts[kDescriptorRingSize];
	for (uint32 i = 0; i < kDescriptorRingSize; i++)
		layouts[i] = m_descriptorSetLayout;
	VkDescriptorSetAllocateInfo dsAlloc{};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descriptorPool;
	dsAlloc.descriptorSetCount = kDescriptorRingSize;
	dsAlloc.pSetLayouts = layouts;
	if (vkAllocateDescriptorSets(device, &dsAlloc, m_descriptorRing) != VK_SUCCESS)
		return false;

	return true;
}

bool VulkanTAAFilter::CreateSizedObjects(VulkanRenderer* renderer)
{
	VkDevice device = renderer->GetLogicalDevice();

	for (uint32 i = 0; i < 2; i++)
	{
		VkImageCreateInfo imgInfo{};
		imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imgInfo.imageType = VK_IMAGE_TYPE_2D;
		imgInfo.format = m_format;
		imgInfo.extent = { (uint32)m_width, (uint32)m_height, 1 };
		imgInfo.mipLevels = 1;
		imgInfo.arrayLayers = 1;
		imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		if (vkCreateImage(device, &imgInfo, nullptr, &m_image[i]) != VK_SUCCESS)
			return false;

		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device, m_image[i], &memReq);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memReq.size;
		allocInfo.memoryTypeIndex = FindMemoryType(renderer->GetPhysicalDevice(), memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &m_memory[i]) != VK_SUCCESS)
			return false;
		vkBindImageMemory(device, m_image[i], m_memory[i], 0);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_image[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = m_format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		VkImageView rawView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &rawView) != VK_SUCCESS)
			return false;

		m_texObj[i] = new VKRObjectTexture();
		m_texObj[i]->m_image = m_image[i]; // handle owned by this filter, wrapper only references it
		m_texObj[i]->m_format = m_format;
		m_texObj[i]->m_imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;
		m_viewObj[i] = new VKRObjectTextureView(m_texObj[i], rawView);
		m_layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	// motion vector target: half resolution (see LatteTAA::Config::useMotionVectors),
	// RGBA16F because Vulkan doesn't reliably support RGB-only color attachments -
	// RG = motion vector (UV units), B = best match cost, A unused
	m_mvWidth = std::max(1, m_width / 2);
	m_mvHeight = std::max(1, m_height / 2);
	const VkFormat mvFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
	{
		VkImageCreateInfo imgInfo{};
		imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imgInfo.imageType = VK_IMAGE_TYPE_2D;
		imgInfo.format = mvFormat;
		imgInfo.extent = { (uint32)m_mvWidth, (uint32)m_mvHeight, 1 };
		imgInfo.mipLevels = 1;
		imgInfo.arrayLayers = 1;
		imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		if (vkCreateImage(device, &imgInfo, nullptr, &m_mvImage) != VK_SUCCESS)
			return false;

		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device, m_mvImage, &memReq);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memReq.size;
		allocInfo.memoryTypeIndex = FindMemoryType(renderer->GetPhysicalDevice(), memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &m_mvMemory) != VK_SUCCESS)
			return false;
		vkBindImageMemory(device, m_mvImage, m_mvMemory, 0);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = m_mvImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = mvFormat;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		VkImageView rawView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &rawView) != VK_SUCCESS)
			return false;

		m_mvTexObj = new VKRObjectTexture();
		m_mvTexObj->m_image = m_mvImage;
		m_mvTexObj->m_format = mvFormat;
		m_mvTexObj->m_imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;
		m_mvViewObj = new VKRObjectTextureView(m_mvTexObj, rawView);
		m_mvLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	}

	// render pass rendering into a single color attachment of the history format
	VKRObjectRenderPass::AttachmentInfo_t attachmentInfo{};
	attachmentInfo.colorAttachment[0].viewObj = m_viewObj[0];
	attachmentInfo.colorAttachment[0].format = m_format;
	m_renderPass = new VKRObjectRenderPass(attachmentInfo, 1);

	for (uint32 i = 0; i < 2; i++)
	{
		std::array<VKRObjectTextureView*, 1> fbAttachments{ m_viewObj[i] };
		m_framebuffer[i] = new VKRObjectFramebuffer(m_renderPass, fbAttachments, Vector2i(m_width, m_height));
	}

	VKRObjectRenderPass::AttachmentInfo_t mvAttachmentInfo{};
	mvAttachmentInfo.colorAttachment[0].viewObj = m_mvViewObj;
	mvAttachmentInfo.colorAttachment[0].format = mvFormat;
	m_mvRenderPass = new VKRObjectRenderPass(mvAttachmentInfo, 1);
	std::array<VKRObjectTextureView*, 1> mvFbAttachments{ m_mvViewObj };
	m_mvFramebuffer = new VKRObjectFramebuffer(m_mvRenderPass, mvFbAttachments, Vector2i(m_mvWidth, m_mvHeight));

	// graphics pipeline (fullscreen triangle, no vertex input, static viewport)
	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = m_vertexShader->GetShaderModule();
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = m_fragmentShader->GetShaderModule();
	stages[1].pName = "main";

	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)m_width;
	viewport.height = (float)m_height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	VkRect2D scissor{};
	scissor.extent = { (uint32)m_width, (uint32)m_height };
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	blendAttachment.blendEnable = VK_FALSE;
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &blendAttachment;

	// dynamic viewport/scissor like the rest of Cemu's pipelines: binding a pipeline
	// with static viewport state would invalidate the dynamic state the game relies on
	VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 2;
	dynamicState.pDynamicStates = dynamicStates;

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = m_pipelineLayout;
	pipelineInfo.renderPass = m_renderPass->m_renderPass;
	pipelineInfo.subpass = 0;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS)
		return false;

	// second pipeline sharing everything but the fragment shader: FXAA mode
	stages[1].module = m_fxaaShader->GetShaderModule();
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipelineFxaa) != VK_SUCCESS)
		return false;

	// third pipeline: motion vector search, targeting the half-res MV render
	// pass instead. Viewport/scissor are dynamic state (set per-draw in Apply),
	// so the pipeline's own viewport/scissor counts (already 1/1 above) don't
	// need to match the half resolution
	stages[1].module = m_mvShader->GetShaderModule();
	pipelineInfo.renderPass = m_mvRenderPass->m_renderPass;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipelineMV) != VK_SUCCESS)
		return false;

	return true;
}

void VulkanTAAFilter::ReleaseResources(VulkanRenderer* renderer)
{
	m_hasValidOutput = false;
	// raw handles are deferred until the GPU has actually finished the command
	// buffer that last used them (not a guessed frame count - a resize burst
	// during a loading-screen stall can easily outlast a fixed guess)
	uint64 releaseCmdBufferId = renderer->GetCurrentCommandBufferId();
	for (uint32 i = 0; i < 2; i++)
	{
		if (m_image[i] != VK_NULL_HANDLE)
		{
			m_pendingDeletes.push_back({ m_image[i], m_memory[i], VK_NULL_HANDLE, releaseCmdBufferId });
			m_image[i] = VK_NULL_HANDLE;
			m_memory[i] = VK_NULL_HANDLE;
		}
		if (m_framebuffer[i])
		{
			renderer->ReleaseDestructibleObject(m_framebuffer[i]);
			m_framebuffer[i] = nullptr;
		}
		if (m_viewObj[i])
		{
			renderer->ReleaseDestructibleObject(m_viewObj[i]); // destroys the VkImageView
			m_viewObj[i] = nullptr;
		}
		if (m_texObj[i])
		{
			m_texObj[i]->m_image = VK_NULL_HANDLE; // image lifetime handled via m_pendingDeletes
			renderer->ReleaseDestructibleObject(m_texObj[i]);
			m_texObj[i] = nullptr;
		}
		m_layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
	}
	if (m_renderPass)
	{
		renderer->ReleaseDestructibleObject(m_renderPass);
		m_renderPass = nullptr;
	}
	if (m_pipeline != VK_NULL_HANDLE)
	{
		m_pendingDeletes.push_back({ VK_NULL_HANDLE, VK_NULL_HANDLE, m_pipeline, releaseCmdBufferId });
		m_pipeline = VK_NULL_HANDLE;
	}
	if (m_pipelineFxaa != VK_NULL_HANDLE)
	{
		m_pendingDeletes.push_back({ VK_NULL_HANDLE, VK_NULL_HANDLE, m_pipelineFxaa, releaseCmdBufferId });
		m_pipelineFxaa = VK_NULL_HANDLE;
	}
	if (m_mvImage != VK_NULL_HANDLE)
	{
		m_pendingDeletes.push_back({ m_mvImage, m_mvMemory, VK_NULL_HANDLE, releaseCmdBufferId });
		m_mvImage = VK_NULL_HANDLE;
		m_mvMemory = VK_NULL_HANDLE;
	}
	if (m_mvFramebuffer)
	{
		renderer->ReleaseDestructibleObject(m_mvFramebuffer);
		m_mvFramebuffer = nullptr;
	}
	if (m_mvViewObj)
	{
		renderer->ReleaseDestructibleObject(m_mvViewObj);
		m_mvViewObj = nullptr;
	}
	if (m_mvTexObj)
	{
		m_mvTexObj->m_image = VK_NULL_HANDLE;
		renderer->ReleaseDestructibleObject(m_mvTexObj);
		m_mvTexObj = nullptr;
	}
	m_mvLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	if (m_mvRenderPass)
	{
		renderer->ReleaseDestructibleObject(m_mvRenderPass);
		m_mvRenderPass = nullptr;
	}
	if (m_pipelineMV != VK_NULL_HANDLE)
	{
		m_pendingDeletes.push_back({ VK_NULL_HANDLE, VK_NULL_HANDLE, m_pipelineMV, releaseCmdBufferId });
		m_pipelineMV = VK_NULL_HANDLE;
	}
}

void VulkanTAAFilter::TickPendingDeletes(VulkanRenderer* renderer)
{
	VkDevice device = renderer->GetLogicalDevice();
	for (auto it = m_pendingDeletes.begin(); it != m_pendingDeletes.end();)
	{
		if (!renderer->HasCommandBufferFinished(it->safeCmdBufferId))
		{
			++it;
			continue;
		}
		if (it->pipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(device, it->pipeline, nullptr);
		if (it->image != VK_NULL_HANDLE)
			vkDestroyImage(device, it->image, nullptr);
		if (it->memory != VK_NULL_HANDLE)
			vkFreeMemory(device, it->memory, nullptr);
		it = m_pendingDeletes.erase(it);
	}
}

bool VulkanTAAFilter::RecreateIfNeeded(VulkanRenderer* renderer, sint32 width, sint32 height, VkFormat format)
{
	if (width == m_width && height == m_height && format == m_format && m_pipeline != VK_NULL_HANDLE)
		return true;

	if (m_vertexShader == nullptr)
	{
		if (!CreateShaders() || !CreateStaticObjects(renderer))
		{
			cemuLog_log(LogType::Force, "VulkanTAAFilter: failed to create static objects");
			return false;
		}
	}

	ReleaseResources(renderer);
	m_width = width;
	m_height = height;
	m_format = format;
	if (!CreateSizedObjects(renderer))
	{
		cemuLog_log(LogType::Force, "VulkanTAAFilter: failed to create sized objects ({}x{})", width, height);
		ReleaseResources(renderer);
		m_width = 0;
		m_height = 0;
		m_format = VK_FORMAT_UNDEFINED;
		return false;
	}
	LatteTAA::InvalidateHistory();
	return true;
}

void VulkanTAAFilter::Apply(VulkanRenderer* renderer, LatteTextureViewVk* scanoutView)
{
	auto& config = LatteTAA::GetConfig();
	if (!config.enabled)
		return;

	// re-presents without new game draws carry no new frame; keep the last resolve
	if (LatteGPUState.drawCallCounter == m_lastDrawCallCounter)
		return;

	// presents whose draws were only overlays (subtitles, letterbox, fades - no
	// full-res scene pass) also carry no new scene: 30fps cutscenes at 60Hz
	// present twice per rendered scene with exactly such draws in between, which
	// defeated the plain drawCallCounter guard above. Re-resolving there
	// unjitters the unchanged scene by the already-advanced jitter index and
	// re-blends it with history; the compounding sub-pixel misalignment was the
	// cutscene smear/moire corruption (confirmed with automated captures:
	// passthrough clean, full TAA smeared). The draw-count valve is a safety
	// net: if a game's scene passes ever elude the heuristic entirely, a burst
	// of draws still forces a resolve instead of freezing the TAA output
	// (temporal mode only: FXAA has no history, re-rendering an overlay-only
	// present is correct and picks up the fresh overlays)
	const uint32 drawsSinceResolve = LatteGPUState.drawCallCounter - m_lastDrawCallCounter;
	const uint32 sceneDrawCounter = LatteTAA::GetSceneDrawCounter();
	if (!config.useFxaa && sceneDrawCounter == m_lastSceneDrawCounter && drawsSinceResolve < 200)
	{
		// measure the next present's draw delta from here, so overlay draws don't
		// accumulate across presents and eventually trip the valve mid-cutscene
		m_lastDrawCallCounter = LatteGPUState.drawCallCounter;
		// a LONG run of scene-less presents is not a cutscene overlay gap but a
		// pure-2D mode (menus, shop, full-screen fades): those redraw fresh 2D
		// content every present, and holding the last resolve would freeze the
		// screen there. Step aside and present the raw scanout until scene
		// geometry returns; half a second of held frame on entering is the
		// trade-off for not shimmering during 30fps cutscenes
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
	// tell the jitter heuristic what the primary scene resolution is
	LatteTAA::SetOutputSize(effWidth, effHeight);

	if (!RecreateIfNeeded(renderer, effWidth, effHeight, scanoutTex->GetFormat()))
		return;

	VkDevice device = renderer->GetLogicalDevice();
	TickPendingDeletes(renderer);

	// caller (VulkanRenderer::TAA_Apply) has already closed any pending render pass
	VkCommandBuffer cmd = renderer->TAA_GetCommandBuffer();

	const uint32 srcIdx = m_currentHistory;
	const uint32 dstIdx = 1 - m_currentHistory;
	VkImage scanoutImage = scanoutTex->GetImageObj()->m_image;

	// keep VKR objects alive as long as this command buffer is in flight
	m_renderPass->flagForCurrentCommandBuffer();
	m_framebuffer[dstIdx]->flagForCurrentCommandBuffer();
	m_viewObj[srcIdx]->flagForCurrentCommandBuffer();
	m_viewObj[dstIdx]->flagForCurrentCommandBuffer();
	m_sampler->flagForCurrentCommandBuffer();

	// Cemu's generic render pass and layout tracking operate in VK_IMAGE_LAYOUT_GENERAL
	// (see VKRObjectRenderPass: initial/final layout = GENERAL). Keep every image in
	// GENERAL and only synchronize access; deviating to *_OPTIMAL breaks the pass.
	VkImageSubresource scanoutSubres{ VK_IMAGE_ASPECT_COLOR_BIT, scanMip, scanSlice };
	VkImageLayout scanoutLayout = scanoutTex->GetImageLayout(scanoutSubres);
	_taaBarrierImage(cmd, scanoutImage, scanoutLayout, VK_IMAGE_LAYOUT_GENERAL,
					 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					 scanMip, scanSlice);

	// history source: keep GENERAL, make prior writes visible to sampling
	_taaBarrierImage(cmd, m_image[srcIdx], m_layout[srcIdx], VK_IMAGE_LAYOUT_GENERAL,
					 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
					 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	m_layout[srcIdx] = VK_IMAGE_LAYOUT_GENERAL;

	// history destination: keep GENERAL, prepare for attachment write
	_taaBarrierImage(cmd, m_image[dstIdx], m_layout[dstIdx], VK_IMAGE_LAYOUT_GENERAL,
					 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
	m_layout[dstIdx] = VK_IMAGE_LAYOUT_GENERAL;

	// sample the exact view the present path would use; the base texture's default
	// view can alias unrelated data (e.g. the game's motion blur intermediates)
	VKRObjectTextureView* scanoutViewObj = scanoutView->GetViewRGBA();
	scanoutViewObj->flagForCurrentCommandBuffer();
	VkImageView scanoutViewRaw = scanoutViewObj->m_textureImageView;

	float jitterPxX = 0.0f, jitterPxY = 0.0f;
	LatteTAA::GetCurrentFrameJitter(jitterPxX, jitterPxY);
	const float jitterUVx = jitterPxX / (float)m_width;
	const float jitterUVy = jitterPxY / (float)m_height;

	// motion vector search pass: only in temporal mode (FXAA has no history to
	// reproject). Always runs when temporal, regardless of the useMotionVectors
	// toggle - that toggle only gates whether the resolve below ACTS on the
	// result, so A/B testing never has to pay for a different barrier/descriptor
	// path and the MV texture is never read while genuinely uninitialized
	if (!config.useFxaa)
	{
		m_mvRenderPass->flagForCurrentCommandBuffer();
		m_mvFramebuffer->flagForCurrentCommandBuffer();
		m_mvViewObj->flagForCurrentCommandBuffer();

		// about to be fully overwritten as an attachment (same idiom as the
		// history "dst" barrier above): no need to preserve prior contents
		_taaBarrierImage(cmd, m_mvImage, m_mvLayout, VK_IMAGE_LAYOUT_GENERAL,
						 0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
						 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
		m_mvLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkDescriptorSet mvDescSet = m_descriptorRing[m_descriptorRingIndex];
		m_descriptorRingIndex = (m_descriptorRingIndex + 1) % kDescriptorRingSize;
		// binding 2 (motion vectors) is unused by this shader but still needs a
		// valid image bound to satisfy the shared layout - reuse the history view
		VkDescriptorImageInfo mvImageInfos[kDescriptorBindings]{};
		mvImageInfos[0] = { m_sampler->GetSampler(), scanoutViewRaw, VK_IMAGE_LAYOUT_GENERAL };
		mvImageInfos[1] = { m_sampler->GetSampler(), m_viewObj[srcIdx]->m_textureImageView, VK_IMAGE_LAYOUT_GENERAL };
		mvImageInfos[2] = { m_sampler->GetSampler(), m_viewObj[srcIdx]->m_textureImageView, VK_IMAGE_LAYOUT_GENERAL };
		VkWriteDescriptorSet mvWrites[kDescriptorBindings]{};
		for (uint32 i = 0; i < kDescriptorBindings; i++)
		{
			mvWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			mvWrites[i].dstSet = mvDescSet;
			mvWrites[i].dstBinding = i;
			mvWrites[i].descriptorCount = 1;
			mvWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			mvWrites[i].pImageInfo = &mvImageInfos[i];
		}
		vkUpdateDescriptorSets(device, kDescriptorBindings, mvWrites, 0, nullptr);

		VkRenderPassBeginInfo mvRpBegin{};
		mvRpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		mvRpBegin.renderPass = m_mvRenderPass->m_renderPass;
		mvRpBegin.framebuffer = m_mvFramebuffer->m_frameBuffer;
		mvRpBegin.renderArea.extent = { (uint32)m_mvWidth, (uint32)m_mvHeight };
		vkCmdBeginRenderPass(cmd, &mvRpBegin, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineMV);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &mvDescSet, 0, nullptr);

		VkViewport mvViewport{ 0.0f, 0.0f, (float)m_mvWidth, (float)m_mvHeight, 0.0f, 1.0f };
		vkCmdSetViewport(cmd, 0, 1, &mvViewport);
		VkRect2D mvScissor{ {0, 0}, { (uint32)m_mvWidth, (uint32)m_mvHeight } };
		vkCmdSetScissor(cmd, 0, 1, &mvScissor);

		PushConstants mvPc{};
		mvPc.jitterUVx = jitterUVx;
		mvPc.jitterUVy = jitterUVy;
		mvPc.mvSearchStep = config.mvSearchStep;
		mvPc.mvRegularization = config.mvRegularization;
		vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(mvPc), &mvPc);
		vkCmdDraw(cmd, 3, 1, 0, 0);
		vkCmdEndRenderPass(cmd);

		_taaBarrierImage(cmd, m_mvImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
						 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
						 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	}

	// update the next descriptor set in the ring
	VkDescriptorSet descSet = m_descriptorRing[m_descriptorRingIndex];
	m_descriptorRingIndex = (m_descriptorRingIndex + 1) % kDescriptorRingSize;

	VkDescriptorImageInfo imageInfos[kDescriptorBindings]{};
	imageInfos[0].sampler = m_sampler->GetSampler();
	imageInfos[0].imageView = scanoutViewRaw;
	imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	imageInfos[1].sampler = m_sampler->GetSampler();
	imageInfos[1].imageView = m_viewObj[srcIdx]->m_textureImageView;
	imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	// always bound (even in FXAA mode, where it just holds stale/unused data -
	// the FXAA shader never samples it and useMotionVectors is forced 0 below)
	imageInfos[2].sampler = m_sampler->GetSampler();
	imageInfos[2].imageView = m_mvViewObj->m_textureImageView;
	imageInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet writes[kDescriptorBindings]{};
	for (uint32 i = 0; i < kDescriptorBindings; i++)
	{
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = descSet;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[i].pImageInfo = &imageInfos[i];
	}
	vkUpdateDescriptorSets(device, kDescriptorBindings, writes, 0, nullptr);

	// resolve pass: render blend(current, clamped history) into history[dst]
	VkRenderPassBeginInfo rpBegin{};
	rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpBegin.renderPass = m_renderPass->m_renderPass;
	rpBegin.framebuffer = m_framebuffer[dstIdx]->m_frameBuffer;
	rpBegin.renderArea.extent = { (uint32)m_width, (uint32)m_height };
	vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, config.useFxaa ? m_pipelineFxaa : m_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &descSet, 0, nullptr);

	VkViewport passViewport{};
	passViewport.x = 0.0f;
	passViewport.y = 0.0f;
	passViewport.width = (float)m_width;
	passViewport.height = (float)m_height;
	passViewport.minDepth = 0.0f;
	passViewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &passViewport);
	VkRect2D passScissor{};
	passScissor.extent = { (uint32)m_width, (uint32)m_height };
	vkCmdSetScissor(cmd, 0, 1, &passScissor);

	// raise the current-frame weight while the game runs slow (30 fps cutscenes):
	// per-frame motion doubles there and the no-motion-vector blend ghosts hard.
	// hold the raised weight for a few frames so 30<->60 transitions don't flicker
	const auto now = std::chrono::steady_clock::now();
	const double frameMs = std::chrono::duration<double, std::milli>(now - m_lastResolveTime).count();
	m_lastResolveTime = now;
	if (frameMs > 25.0)
		m_lowFpsHold = 8;
	else if (m_lowFpsHold > 0)
		m_lowFpsHold--;

	PushConstants pc{};
	if (config.useFxaa)
	{
		// FXAA repurposes jitterUVx/y as the reciprocal frame size
		pc.blendFactor = 0.0f;
		pc.historyValid = 0.0f;
		pc.passthrough = config.debugPassthrough ? 1.0f : 0.0f;
		pc.jitterUVx = 1.0f / (float)m_width;
		pc.jitterUVy = 1.0f / (float)m_height;
		pc.useMotionVectors = 0.0f;
	}
	else
	{
		pc.blendFactor = (m_lowFpsHold > 0) ? std::max(config.blendFactor, 0.5f) : config.blendFactor;
		pc.historyValid = LatteTAA::ConsumeHistoryValidFlag() ? 1.0f : 0.0f;
		pc.passthrough = config.debugPassthrough ? 1.0f : 0.0f;
		pc.jitterUVx = jitterUVx;
		pc.jitterUVy = jitterUVy;
		pc.useMotionVectors = config.useMotionVectors ? 1.0f : 0.0f;
	}
	// meaningful only in temporal mode (FXAA never runs the MV pass, so the
	// texture would show stale/garbage data there)
	pc.mvDebugView = (config.mvDebugView && !config.useFxaa) ? 1.0f : 0.0f;
	vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

	vkCmdDraw(cmd, 3, 1, 0, 0);
	vkCmdEndRenderPass(cmd);

	// the game's scanout texture is left untouched (see class comment); keep the
	// layout tracker coherent with the read barrier issued above
	VkImageSubresourceRange scanoutRange{ VK_IMAGE_ASPECT_COLOR_BIT, scanMip, 1, scanSlice, 1 };
	scanoutTex->SetImageLayout(scanoutRange, VK_IMAGE_LAYOUT_GENERAL);

	// make the resolve visible to the backbuffer blit's fragment sampling
	_taaBarrierImage(cmd, m_image[dstIdx], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
					 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	m_currentHistory = dstIdx;
	m_hasValidOutput = true;
	m_lastDrawCallCounter = LatteGPUState.drawCallCounter;
	m_lastSceneDrawCounter = sceneDrawCounter;
	if (!config.useFxaa)
		LatteTAA::NotifyFramePresented();
}

VkDescriptorSet VulkanTAAFilter::GetPresentDescriptorSet(VulkanRenderer* renderer, VkDescriptorSetLayout blitLayout,
														 sint32 expectedWidth, sint32 expectedHeight)
{
	if (!m_hasValidOutput || expectedWidth != m_width || expectedHeight != m_height)
		return VK_NULL_HANDLE;
	if (!m_sampler || !m_viewObj[m_currentHistory])
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

	// keep the sampled objects alive for this command buffer; also required on
	// frames where the resolve was skipped and only the present samples them
	m_viewObj[m_currentHistory]->flagForCurrentCommandBuffer();
	m_sampler->flagForCurrentCommandBuffer();

	VkDescriptorSet descSet = m_presentRing[m_presentRingIndex];
	m_presentRingIndex = (m_presentRingIndex + 1) % kPresentRingSize;

	VkDescriptorImageInfo imageInfo{};
	imageInfo.sampler = m_sampler->GetSampler();
	imageInfo.imageView = m_viewObj[m_currentHistory]->m_textureImageView;
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

VkImageView VulkanTAAFilter::GetResolvedImageViewIfValid(sint32 expectedWidth, sint32 expectedHeight)
{
	if (!m_hasValidOutput || expectedWidth != m_width || expectedHeight != m_height)
		return VK_NULL_HANDLE;
	if (!m_viewObj[m_currentHistory])
		return VK_NULL_HANDLE;
	m_viewObj[m_currentHistory]->flagForCurrentCommandBuffer();
	return m_viewObj[m_currentHistory]->m_textureImageView;
}

VkImageView VulkanTAAFilter::GetMotionVectorsViewIfValid(sint32& outWidth, sint32& outHeight, VkImage& outImage)
{
	outWidth = 0;
	outHeight = 0;
	outImage = VK_NULL_HANDLE;
	// m_hasValidOutput is only set true at the end of a successful Apply(); the MV
	// pass always runs there in temporal mode (see its comment in Apply()), so this
	// implies fresh MV data too. useFxaa is re-checked since FXAA mode never runs
	// the MV pass at all - m_mvViewObj could be stale from a mode switch.
	if (!m_hasValidOutput || LatteTAA::GetConfig().useFxaa)
		return VK_NULL_HANDLE;
	if (!m_mvViewObj)
		return VK_NULL_HANDLE;
	m_mvViewObj->flagForCurrentCommandBuffer();
	outWidth = m_mvWidth;
	outHeight = m_mvHeight;
	outImage = m_mvImage;
	return m_mvViewObj->m_textureImageView;
}

void VulkanTAAFilter::Shutdown(VulkanRenderer* renderer)
{
	ReleaseResources(renderer);
	VkDevice device = renderer->GetLogicalDevice();
	for (auto& pd : m_pendingDeletes)
	{
		if (pd.pipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(device, pd.pipeline, nullptr);
		if (pd.image != VK_NULL_HANDLE)
			vkDestroyImage(device, pd.image, nullptr);
		if (pd.memory != VK_NULL_HANDLE)
			vkFreeMemory(device, pd.memory, nullptr);
	}
	m_pendingDeletes.clear();
	if (m_descriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
		m_descriptorPool = VK_NULL_HANDLE;
	}
	if (m_presentDescriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, m_presentDescriptorPool, nullptr);
		m_presentDescriptorPool = VK_NULL_HANDLE;
	}
	if (m_pipelineLayout != VK_NULL_HANDLE)
	{
		vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
		m_pipelineLayout = VK_NULL_HANDLE;
	}
	if (m_descriptorSetLayout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
		m_descriptorSetLayout = VK_NULL_HANDLE;
	}
	if (m_sampler)
	{
		m_sampler->decRef();
		m_sampler = nullptr;
	}
}

// member of VulkanRenderer so the filter can be driven from common Latte code
// (declaration added to VulkanRenderer.h by the TAA wiring patch)
void VulkanRenderer::TAA_Apply(LatteTextureView* textureView)
{
	if (!textureView || !textureView->baseTexture)
		return;
	LatteTextureVk* texVk = (LatteTextureVk*)textureView->baseTexture;
	if (texVk->isDepth)
		return;
	// close any pending game render pass before the filter records its own commands
	draw_endRenderPass();
	VulkanTAAFilter::GetInstance().Apply(this, (LatteTextureViewVk*)textureView);
	// the filter bound its own pipeline and dynamic state behind the state
	// tracker's back: force a pipeline rebind on the next game draw and restore
	// the tracked viewport/scissor (same idiom as surfaceCopy_viaDrawcall)
	m_state.currentPipeline = VK_NULL_HANDLE;
	vkCmdSetViewport(m_state.currentCommandBuffer, 0, 1, &m_state.currentViewport);
	vkCmdSetScissor(m_state.currentCommandBuffer, 0, 1, &m_state.currentScissorRect);
}

VkCommandBuffer VulkanRenderer::TAA_GetCommandBuffer()
{
	return getCurrentCommandBuffer();
}
