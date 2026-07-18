#pragma once

class LatteTexture;
class LatteTextureView;

// DLAA via NVIDIA's real NGX/DLSS runtime (dependencies/DLSS, the public SDK from
// github.com/NVIDIA/DLSS) - not a hand-written approximation like the rest of this
// fork's screen-space effects. Requires an NVIDIA RTX GPU (Turing+) with the NGX
// driver component; gracefully unavailable everywhere else (see VulkanRenderer's
// InitNGX, which probes this at device creation time and never blocks startup).

namespace LatteDLSS
{
	struct Config
	{
		bool enabled{ false }; // user/env requested DLAA - independent of whether NGX actually initialized on this system
	};

	Config& GetConfig();

	// extra negative texture LOD bias to stack on top of the game's own per-texture
	// bias while DLAA is actually running (0.0f otherwise) - see LatteDLSS.cpp for
	// the derivation. Read from VulkanRenderer::draw_getOrCreateDescriptorSet's
	// sampler setup, same call site as the existing graphic-pack LOD bias overrides.
	float GetMipmapBias();

	// dispatched from LatteTexture_Delete / LatteMRT::SetDepthAndStencilAttachment,
	// same pattern as LatteSSAO - see VulkanDLSSFilter's NotifyTextureDeletion/NotifyDepthBind
	void NotifyTextureDeletion(LatteTexture* texture);
	void NotifyDepthBind(LatteTextureView* view);
}
