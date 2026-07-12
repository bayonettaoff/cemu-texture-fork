#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Core/LatteDLSS.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#if BOOST_OS_WINDOWS
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanDLSSFilter.h"
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
