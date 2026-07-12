#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Core/LatteSSAO.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanSSAOFilter.h"

namespace LatteSSAO
{
	static Config s_config;

	Config& GetConfig()
	{
		// environment overrides for automated testing (CEMU_SSAO=1 forces the
		// filter on from startup; CEMU_SSAO_DEBUG shows the raw AO term;
		// CEMU_SSR/CEMU_SSR_DEBUG are the SSR equivalents; CEMU_CS is contact
		// shadows). The single Debug menu checkbox sets all four flags together,
		// so these are the only way to isolate one effect at a time (leave the
		// menu checkbox untouched - clicking it overwrites whatever combination
		// was set here)
		static bool s_envChecked = false;
		if (!s_envChecked)
		{
			s_envChecked = true;
			if (const char* v = std::getenv("CEMU_SSAO"))
				s_config.enabled = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_SSAO_DEBUG"))
				s_config.debugShowAOOnly = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_SSR"))
				s_config.ssrEnabled = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_SSR_DEBUG"))
				s_config.ssrDebugView = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_SSGI"))
				s_config.giEnabled = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_CS"))
				s_config.contactShadowsEnabled = (v[0] == '1');
			// approximate view-space reconstruction constants (see LatteSSAO.h) -
			// first guesses, expect in-game tuning; exposed here so that doesn't
			// require a recompile each round
			if (const char* v = std::getenv("CEMU_SSAO_FOV"))
				s_config.fovYDegrees = (float)std::atof(v);
			if (const char* v = std::getenv("CEMU_SSAO_DEPTHK"))
				s_config.depthLinearK = (float)std::atof(v);
			if (const char* v = std::getenv("CEMU_SSAO_SMOOTHRADIUS"))
				s_config.smoothRadiusView = (float)std::atof(v);
			cemuLog_log(LogType::Force, "SSAO config: enabled={} debugShowAOOnly={} ssrEnabled={} ssrDebugView={} giEnabled={} contactShadowsEnabled={} fovYDegrees={} depthLinearK={} smoothRadiusView={}",
				s_config.enabled, s_config.debugShowAOOnly, s_config.ssrEnabled, s_config.ssrDebugView, s_config.giEnabled, s_config.contactShadowsEnabled,
				s_config.fovYDegrees, s_config.depthLinearK, s_config.smoothRadiusView);
		}
		return s_config;
	}

	bool AnyEffectEnabled()
	{
		auto& config = GetConfig();
		return config.enabled || config.ssrEnabled || config.contactShadowsEnabled || config.giEnabled;
	}

	void NotifyTextureDeletion(LatteTexture* texture)
	{
		if (g_renderer && g_renderer->GetType() == RendererAPI::Vulkan)
			VulkanSSAOFilter::GetInstance().NotifyTextureDeletion(texture);
	}

	void NotifyDepthBind(LatteTextureView* view)
	{
		// hot path (every depth attachment change) - bail as cheaply as possible
		if (!AnyEffectEnabled())
			return;
		if (g_renderer && g_renderer->GetType() == RendererAPI::Vulkan)
			VulkanSSAOFilter::GetInstance().NotifyDepthBind(view);
	}
}
