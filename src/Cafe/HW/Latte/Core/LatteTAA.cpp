#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Core/LatteTAA.h"
#include "Cafe/HW/Latte/Core/LatteSSAO.h"
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanTAAFilter.h"

namespace LatteTAA
{
	static Config s_config;
	static uint32 s_frameIndex = 0;
	static bool s_historyValid = false;
	static sint32 s_outputWidth = 0;
	static sint32 s_outputHeight = 0;

	// 8-sample Halton(2,3) sequence, offsets in [-0.5, 0.5)
	static const float s_haltonX[8] = { 0.500000f, 0.250000f, 0.750000f, 0.125000f, 0.625000f, 0.375000f, 0.875000f, 0.062500f };
	static const float s_haltonY[8] = { 0.333333f, 0.666667f, 0.111111f, 0.444444f, 0.777778f, 0.222222f, 0.555556f, 0.888889f };

	Config& GetConfig()
	{
		// environment overrides for automated testing (CEMU_TAA=1 forces the
		// filter on from startup; jitter/passthrough analogous). CEMU_TAA_FXAA
		// switches between the two AA modes the single Debug menu checkbox
		// shares - useFxaa defaults to true (single-frame, can't corrupt) and
		// there is no menu item for the temporal resolve path anymore since the
		// menu consolidation, so this is the only way to reach it without
		// editing the default in code
		static bool s_envChecked = false;
		if (!s_envChecked)
		{
			s_envChecked = true;
			if (const char* v = std::getenv("CEMU_TAA"))
				s_config.enabled = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_JITTER"))
				s_config.jitterEnabled = (v[0] == '1');
			// jitter amplitude (1.0 = +-0.5px, the TAA standard). Lowering it trades
			// AA quality for less wobble on content our resolve cannot un-jitter
			// (HUD and the game's screen-anchored post effects are composited into
			// the scanout WITHOUT jitter, so the global un-jitter shifts them by the
			// jitter amount every frame - inherent to intercepting post-composition)
			if (const char* v = std::getenv("CEMU_TAA_JITTER_SCALE"))
				s_config.jitterScale = (float)std::atof(v);
			if (const char* v = std::getenv("CEMU_TAA_PASSTHROUGH"))
				s_config.debugPassthrough = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_FXAA"))
				s_config.useFxaa = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_MV"))
				s_config.useMotionVectors = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_MV_STEP"))
				s_config.mvSearchStep = (float)std::atof(v);
			if (const char* v = std::getenv("CEMU_TAA_MV_REG"))
				s_config.mvRegularization = (float)std::atof(v);
			if (const char* v = std::getenv("CEMU_TAA_MV_DEBUG"))
				s_config.mvDebugView = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_CLIPSPACE_JITTER"))
				s_config.clipSpaceJitter = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_OPTICALFLOW"))
				s_config.useOpticalFlow = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_ANALYTICALMV"))
				s_config.useAnalyticalMV = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_MVH_DISAGREE"))
				s_config.mvHybridDisagreePx = (float)std::atof(v);
			if (const char* v = std::getenv("CEMU_TAA_MVH_COST"))
				s_config.mvHybridCostMax = (float)std::atof(v);
			cemuLog_log(LogType::Force, "TAA config: enabled={} jitter={} passthrough={} useFxaa={} useMotionVectors={} mvSearchStep={} mvRegularization={} mvDebugView={} clipSpaceJitter={} useOpticalFlow={} useAnalyticalMV={}",
				s_config.enabled, s_config.jitterEnabled, s_config.debugPassthrough, s_config.useFxaa,
				s_config.useMotionVectors, s_config.mvSearchStep, s_config.mvRegularization, s_config.mvDebugView, s_config.clipSpaceJitter, s_config.useOpticalFlow, s_config.useAnalyticalMV);
		}
		return s_config;
	}

	void NotifyTextureDeletion(LatteTexture* texture)
	{
		if (g_renderer && g_renderer->GetType() == RendererAPI::Vulkan)
			VulkanTAAFilter::GetInstance().NotifyTextureDeletion(texture);
	}

	void NotifyDepthBind(LatteTextureView* view)
	{
		// hot path (every depth attachment change) - bail as cheaply as possible
		if (!s_config.useAnalyticalMV)
			return;
		if (g_renderer && g_renderer->GetType() == RendererAPI::Vulkan)
			VulkanTAAFilter::GetInstance().NotifyDepthBind(view);
	}

	void NotifyFramePresented()
	{
		s_frameIndex++;
	}

	uint32 GetFrameIndex()
	{
		return s_frameIndex;
	}

	void SetOutputSize(sint32 width, sint32 height)
	{
		s_outputWidth = width;
		s_outputHeight = height;
	}

	static uint32 s_sceneDrawCounter = 0;

	bool GetViewportJitter(bool depthTestEnabled, bool depthWriteEnabled,
						   float vpWidth, float vpHeight,
						   sint32 rtWidth, sint32 rtHeight,
						   float& jitterX, float& jitterY)
	{
		jitterX = 0.0f;
		jitterY = 0.0f;
		// the scene-draw counter also serves the SSAO/SSR pass, so keep
		// classifying draws while TAA itself is off but those effects are on
		if (!s_config.enabled && !LatteSSAO::AnyEffectEnabled())
			return false;
		// scene pass heuristic: geometry that tests AND writes depth. UI has no
		// depth test; post-process quads may test (func=always) but do not write.
		// Depth-only passes are deliberately NOT excluded: camera-space z-prepasses
		// must receive the same jitter as the color passes or the color pass fails
		// the depth test along silhouettes (see the header comment). Shadow maps
		// (also depth-only, must not be jittered) are rejected by the size checks
		// below instead
		if (!depthTestEnabled || !depthWriteEnabled)
			return false;
		// height is negative in Cemu's viewport convention (Y flip)
		vpWidth = std::fabs(vpWidth);
		vpHeight = std::fabs(vpHeight);
		// tiny viewports (probe/effect passes) don't drive the visible image
		if (vpWidth < 320.0f || vpHeight < 180.0f)
			return false;
		// viewport must roughly cover the render target, otherwise this is likely
		// a sub-rect effect pass (blur strips, bloom pyramids, ...)
		if (rtWidth > 0 && rtHeight > 0)
		{
			const float coverW = vpWidth / (float)rtWidth;
			const float coverH = vpHeight / (float)rtHeight;
			if (coverW < 0.9f || coverH < 0.9f)
				return false;
		}
		// only the primary scene resolution gets jitter. Reduced-resolution effect
		// buffers (half/quarter-res particles, DoF, bloom) often keep depth state
		// enabled and fully cover their own small RT; a half-pixel jitter there is
		// several output pixels and shatters the effect when composited over the
		// (differently jittered) scene. Output size is learned at the first resolve,
		// so the very first frame after enabling TAA renders without jitter.
		if (s_outputWidth <= 0 || s_outputHeight <= 0)
			return false;
		if (rtWidth < (sint32)(0.85f * (float)s_outputWidth) || rtHeight < (sint32)(0.85f * (float)s_outputHeight))
			return false;
		// ... and TWO-SIDED: also reject targets meaningfully LARGER than the scene
		// output. Now that depth-only passes are eligible (z-prepass fix, see the
		// header comment), this is what keeps shadow maps out - a 4096x4096 shadow
		// cascade at 4K output passes every check above (bigger than 85% of output,
		// fully covered by its viewport) but its height wildly exceeds the scene's.
		// Camera-space scene passes match the output size exactly, so a modest 1.2x
		// allowance keeps them safely inside
		if (rtWidth > (sint32)(1.2f * (float)s_outputWidth) || rtHeight > (sint32)(1.2f * (float)s_outputHeight))
			return false;
		// this draw is real full-res scene geometry: mark the frame as carrying a
		// new scene for the overlay-only-present detection (TAA resolve and
		// SSAO/SSR pass). Evaluated regardless of the jitter kill-switch
		s_sceneDrawCounter++;
		if (!s_config.enabled || !s_config.jitterEnabled || s_config.useFxaa)
			return false;
		const uint32 idx = s_frameIndex & 7;
		jitterX = (s_haltonX[idx] - 0.5f) * s_config.jitterScale;
		jitterY = (s_haltonY[idx] - 0.5f) * s_config.jitterScale;
		return true;
	}

	static float s_currentDrawJitterClipX = 0.0f;
	static float s_currentDrawJitterClipY = 0.0f;

	void SetCurrentDrawJitterClipSpace(float jitterClipX, float jitterClipY)
	{
		s_currentDrawJitterClipX = jitterClipX;
		s_currentDrawJitterClipY = jitterClipY;
	}

	void GetCurrentDrawJitterClipSpace(float& jitterClipX, float& jitterClipY)
	{
		jitterClipX = s_currentDrawJitterClipX;
		jitterClipY = s_currentDrawJitterClipY;
	}

	void GetCurrentFrameJitter(float& jitterX, float& jitterY)
	{
		jitterX = 0.0f;
		jitterY = 0.0f;
		// must mirror GetViewportJitter's gate exactly: in FXAA mode no scene pass
		// is ever jittered (useFxaa forces GetViewportJitter to return false), so
		// this has to report zero too - otherwise the SSAO/SSR pass "unjitters"
		// depth by an offset that was never actually applied to the scene
		if (!s_config.enabled || !s_config.jitterEnabled || s_config.useFxaa)
			return;
		const uint32 idx = s_frameIndex & 7;
		jitterX = (s_haltonX[idx] - 0.5f) * s_config.jitterScale;
		jitterY = (s_haltonY[idx] - 0.5f) * s_config.jitterScale;
	}

	void InvalidateHistory()
	{
		s_historyValid = false;
	}

	bool ConsumeHistoryValidFlag()
	{
		const bool v = s_historyValid;
		s_historyValid = true;
		return v;
	}

	uint32 GetSceneDrawCounter()
	{
		return s_sceneDrawCounter;
	}
}
