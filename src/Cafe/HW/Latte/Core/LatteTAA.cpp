#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Core/LatteTAA.h"

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
		// filter on from startup; jitter/passthrough analogous)
		static bool s_envChecked = false;
		if (!s_envChecked)
		{
			s_envChecked = true;
			if (const char* v = std::getenv("CEMU_TAA"))
				s_config.enabled = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_JITTER"))
				s_config.jitterEnabled = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_TAA_PASSTHROUGH"))
				s_config.debugPassthrough = (v[0] == '1');
			cemuLog_log(LogType::Force, "TAA config: enabled={} jitter={} passthrough={}", s_config.enabled, s_config.jitterEnabled, s_config.debugPassthrough);
		}
		return s_config;
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

	bool GetViewportJitter(bool depthTestEnabled, bool depthWriteEnabled, bool hasColorBuffer,
						   float vpWidth, float vpHeight,
						   sint32 rtWidth, sint32 rtHeight,
						   float& jitterX, float& jitterY)
	{
		jitterX = 0.0f;
		jitterY = 0.0f;
		if (!s_config.enabled || !s_config.jitterEnabled)
			return false;
		// scene pass heuristic: geometry that tests AND writes depth into a color
		// buffer. Shadow maps are depth-only; UI has no depth test; post-process
		// quads may test (func=always) but do not write.
		if (!depthTestEnabled || !depthWriteEnabled || !hasColorBuffer)
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
		const uint32 idx = s_frameIndex & 7;
		jitterX = (s_haltonX[idx] - 0.5f) * s_config.jitterScale;
		jitterY = (s_haltonY[idx] - 0.5f) * s_config.jitterScale;
		return true;
	}

	void GetCurrentFrameJitter(float& jitterX, float& jitterY)
	{
		jitterX = 0.0f;
		jitterY = 0.0f;
		if (!s_config.enabled || !s_config.jitterEnabled)
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
}
