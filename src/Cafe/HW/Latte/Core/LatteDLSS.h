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

	// dispatched from LatteTexture_Delete / LatteMRT::SetDepthAndStencilAttachment,
	// same pattern as LatteSSAO - see VulkanDLSSFilter's NotifyTextureDeletion/NotifyDepthBind
	void NotifyTextureDeletion(LatteTexture* texture);
	void NotifyDepthBind(LatteTextureView* view);
}
