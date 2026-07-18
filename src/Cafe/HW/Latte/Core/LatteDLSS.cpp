#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Core/LatteDLSS.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#if BOOST_OS_WINDOWS
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanDLSSFilter.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"
#endif

namespace LatteDLSS
{
	static Config s_config;

	Config& GetConfig()
	{
		static bool s_envChecked = false;
		if (!s_envChecked)
		{
			s_envChecked = true;
			if (const char* v = std::getenv("CEMU_DLAA"))
				s_config.enabled = (v[0] == '1');
			cemuLog_log(LogType::Force, "DLAA config: enabled={}", s_config.enabled);
		}
		return s_config;
	}

	float GetMipmapBias()
	{
#if BOOST_OS_WINDOWS
		if (!GetConfig().enabled || !g_renderer || g_renderer->GetType() != RendererAPI::Vulkan)
			return 0.0f;
		if (!VulkanRenderer::GetInstance()->IsDLAAAvailable())
			return 0.0f;
		// NVIDIA's DLSS/DLAA integration guidance: mipBias = log2(renderRes / displayRes) - 1.0,
		// applied on top of the game's own sampler bias so textures stay as sharp under temporal
		// jitter as they'd be without it. DLAA in this fork is always native-resolution (see
		// VulkanDLSSFilter::Apply - InWidth == InTargetWidth), so the ratio is always 1 and the
		// formula collapses to this constant.
		return -1.0f;
#else
		return 0.0f;
#endif
	}

	void NotifyTextureDeletion(LatteTexture* texture)
	{
#if BOOST_OS_WINDOWS
		if (g_renderer && g_renderer->GetType() == RendererAPI::Vulkan)
			VulkanDLSSFilter::GetInstance().NotifyTextureDeletion(texture);
#endif
	}

	void NotifyDepthBind(LatteTextureView* view)
	{
#if BOOST_OS_WINDOWS
		// hot path (every depth attachment change) - bail as cheaply as possible
		if (!GetConfig().enabled)
			return;
		if (g_renderer && g_renderer->GetType() == RendererAPI::Vulkan)
			VulkanDLSSFilter::GetInstance().NotifyDepthBind(view);
#endif
	}
}
