#pragma once

// Temporal anti-aliasing for Cemu (generic, no per-game knowledge)
// Subpixel viewport jitter on scene passes + temporal resolve with neighborhood
// clamping at scanout time. Without motion vectors the clamp rejects stale history
// in moving regions, so quality degrades to no-AA there instead of ghosting.

namespace LatteTAA
{
	struct Config
	{
		bool enabled{ false };
		bool jitterEnabled{ true }; // separate kill-switch to isolate jitter vs resolve issues
		float jitterScale{ 1.0f };  // jitter amplitude scale (1.0 = +/- half pixel, the TAA standard)
		float blendFactor{ 0.2f }; // weight of current frame in the history blend (no motion vectors -> favor shorter trails over maximum smoothing)
		bool debugPassthrough{ false }; // resolve outputs the current frame only (isolates history-blend issues)
	};

	Config& GetConfig();

	// per-frame jitter sequence (Halton 2/3), advanced once per presented frame
	void NotifyFramePresented();
	uint32 GetFrameIndex();

	// effective scanout size, reported by the resolve pass each frame; jitter is
	// restricted to render targets of (roughly) this size so reduced-resolution
	// effect passes never receive it
	void SetOutputSize(sint32 width, sint32 height);

	// returns true and writes the jitter offset (in render-target pixels) when the
	// pass currently being set up should receive jitter. Caller provides the state
	// this module needs to classify the pass:
	// - depthTestEnabled/depthWriteEnabled: DB_DEPTH_CONTROL bits. Scene geometry
	//   tests AND writes depth; post-process quads often leave the test on with
	//   func=always but never write, and jittering those shatters depth-based
	//   effects (e.g. per-object motion blur)
	// - hasColorBuffer: at least one color attachment bound
	// - vpWidth/vpHeight: viewport size after graphic pack scaling
	// - rtWidth/rtHeight: effective size of the active render target
	bool GetViewportJitter(bool depthTestEnabled, bool depthWriteEnabled, bool hasColorBuffer,
						   float vpWidth, float vpHeight,
						   sint32 rtWidth, sint32 rtHeight,
						   float& jitterX, float& jitterY);

	// jitter offset (render-target pixels) that scene passes used this frame; the
	// resolve pass needs it to sample the current frame at unjittered positions.
	// Returns 0,0 when jitter is disabled. Must be queried before NotifyFramePresented.
	void GetCurrentFrameJitter(float& jitterX, float& jitterY);

	// history handling for the resolve pass
	// invalidate when resolution/output changes or TAA is toggled
	void InvalidateHistory();
	bool ConsumeHistoryValidFlag(); // false on the first resolve after invalidation
}
