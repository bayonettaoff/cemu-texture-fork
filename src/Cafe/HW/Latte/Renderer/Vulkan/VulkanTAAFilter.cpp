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
		"layout(push_constant) uniform pushConstants {\r\n"
		"float blendFactor;\r\n"
		"float historyValid;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float passthrough;\r\n"
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
		"vec3 histRaw = texture(texHistory, passUV).rgb;\r\n"
		"vec3 hist = clamp(histRaw, cmin, cmax);\r\n"
		// history far outside the current neighborhood means it is stale
		// (occlusion change, cut, or a skipped draw) -> converge to current fast
		"float clampDist = length(histRaw - hist);\r\n"
		"float w = clamp(uf_pc.blendFactor + clampDist * 4.0, 0.0, 1.0);\r\n"
		"vec3 resolved = mix(hist, cur.rgb, w);\r\n"
		"float keepHistory = uf_pc.historyValid * (1.0 - uf_pc.passthrough);\r\n"
		"outColor = vec4(mix(cur.rgb, resolved, keepHistory), cur.a);\r\n"
		"}\r\n";

	std::string vsStr(vsSrc);
	m_vertexShader = new RendererShaderVk(RendererShader::ShaderType::kVertex, 0, 0, false, false, vsStr);
	m_vertexShader->PreponeCompilation(true);

	std::string fsStr(fsSrc);
	m_fragmentShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, fsStr);
	m_fragmentShader->PreponeCompilation(true);

	return m_vertexShader != nullptr && m_fragmentShader != nullptr;
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

	// descriptor set layout: two combined image samplers (fragment)
	VkDescriptorSetLayoutBinding bindings[2]{};
	for (uint32 i = 0; i < 2; i++)
	{
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dslInfo{};
	dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dslInfo.bindingCount = 2;
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

	// descriptor pool + ring of sets, updated round-robin each frame
	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSize.descriptorCount = kDescriptorRingSize * 2;
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

	return true;
}

void VulkanTAAFilter::ReleaseResources(VulkanRenderer* renderer)
{
	m_hasValidOutput = false;
	// raw handles are deferred a few frames to outlive in-flight command buffers
	for (uint32 i = 0; i < 2; i++)
	{
		if (m_image[i] != VK_NULL_HANDLE)
		{
			m_pendingDeletes.push_back({ m_image[i], m_memory[i], VK_NULL_HANDLE, 8 });
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
		m_pendingDeletes.push_back({ VK_NULL_HANDLE, VK_NULL_HANDLE, m_pipeline, 8 });
		m_pipeline = VK_NULL_HANDLE;
	}
}

void VulkanTAAFilter::TickPendingDeletes(VkDevice device)
{
	for (auto it = m_pendingDeletes.begin(); it != m_pendingDeletes.end();)
	{
		if (it->framesLeft > 0)
		{
			it->framesLeft--;
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
	TickPendingDeletes(device);

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

	// update the next descriptor set in the ring
	VkDescriptorSet descSet = m_descriptorRing[m_descriptorRingIndex];
	m_descriptorRingIndex = (m_descriptorRingIndex + 1) % kDescriptorRingSize;

	// sample the exact view the present path would use; the base texture's default
	// view can alias unrelated data (e.g. the game's motion blur intermediates)
	VKRObjectTextureView* scanoutViewObj = scanoutView->GetViewRGBA();
	scanoutViewObj->flagForCurrentCommandBuffer();
	VkImageView scanoutViewRaw = scanoutViewObj->m_textureImageView;
	VkDescriptorImageInfo imageInfos[2]{};
	imageInfos[0].sampler = m_sampler->GetSampler();
	imageInfos[0].imageView = scanoutViewRaw;
	imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	imageInfos[1].sampler = m_sampler->GetSampler();
	imageInfos[1].imageView = m_viewObj[srcIdx]->m_textureImageView;
	imageInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet writes[2]{};
	for (uint32 i = 0; i < 2; i++)
	{
		writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet = descSet;
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[i].pImageInfo = &imageInfos[i];
	}
	vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

	// resolve pass: render blend(current, clamped history) into history[dst]
	VkRenderPassBeginInfo rpBegin{};
	rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpBegin.renderPass = m_renderPass->m_renderPass;
	rpBegin.framebuffer = m_framebuffer[dstIdx]->m_frameBuffer;
	rpBegin.renderArea.extent = { (uint32)m_width, (uint32)m_height };
	vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
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

	PushConstants pc;
	pc.blendFactor = (m_lowFpsHold > 0) ? std::max(config.blendFactor, 0.5f) : config.blendFactor;
	pc.historyValid = LatteTAA::ConsumeHistoryValidFlag() ? 1.0f : 0.0f;
	pc.passthrough = config.debugPassthrough ? 1.0f : 0.0f;
	float jitterPxX, jitterPxY;
	LatteTAA::GetCurrentFrameJitter(jitterPxX, jitterPxY);
	pc.jitterUVx = jitterPxX / (float)m_width;
	pc.jitterUVy = jitterPxY / (float)m_height;
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
