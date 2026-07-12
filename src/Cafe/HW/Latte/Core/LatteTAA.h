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
		// default mode: single-frame FXAA (high quality) instead of the temporal
		// resolve. No history, no jitter - cannot corrupt anything by design.
		// ROOT CAUSE of the temporal path's long-standing cutscene corruption
		// ("geometry shredded into directional streaks") confirmed 2026-07-12:
		// NOT the resolve itself, NOT jitter (both were already ruled out, see
		// jitterEnabled below) - it was Bayonetta 2's cutscenes running at
		// native 30fps while presented at 60Hz, which doubles the per-object
		// motion blur's inter-frame delta and (per the 2026-07-11 view-space
		// investigation into unrelated SSR faceting) is exactly the kind of
		// heavy directional blur that defeats 1-sigma variance-clipping TAA:
		// the clamp box inflates along the blur direction and lets misaligned
		// history through as an accepted "double exposure". CONFIRMED FIXED
		// (user report, clean cutscene, temporal resolve + jitter still off)
		// by forcing cutscenes to genuine 60fps via Bayonetta 2's own "60 FPS
		// Cutscenes" graphic pack mod - halves the per-frame motion delta and
		// removes the 30-at-60Hz double-present pattern entirely. This is a
		// per-game mitigation, not a fix in this filter, so the safe FXAA
		// default stays - only flip this for a game/session where cutscenes
		// are confirmed to not double-expose (native 60fps, or a similar mod)
		bool useFxaa{ true };
		// default ON as of 2026-07-12: an earlier (2026-07-10) hypothesis blamed
		// jitter itself for the cutscene shredding and this was kept off as a
		// result - WRONG, ruled out that same session (corruption reproduced
		// with jitter off too). The REAL cause of that bug was the 30fps-
		// cutscene/motion-blur interaction (useFxaa's comment above), fixed
		// separately. Jitter's own long-standing bug (depth-dependent viewport
		// offset causing shimmer/moire even on a static scene, see
		// clipSpaceJitter below) is now fixed too, confirmed clean by the user
		// with clipSpaceJitter - both default true together, since jitter with
		// the OLD viewport-offset mechanism would still corrupt things
		bool jitterEnabled{ true };
		float jitterScale{ 1.0f };  // jitter amplitude scale (1.0 = +/- half pixel, the TAA standard)
		// default ON as of 2026-07-12 (confirmed clean by the user, see below):
		// apply jitter in the vertex shader's clip-space output (scaled by w,
		// before the perspective divide) instead of offsetting the viewport
		// rectangle. The viewport-offset approach this fork used until
		// 2026-07-12 is DEPTH-DEPENDENT - confirmed both by direct observation
		// (shimmer/moire on architecture at varying depths, present in a single
		// passthrough frame with the camera static, so unrelated to motion or
		// our own history blend) and independently by an NVIDIA developer forum
		// report of the exact same viewport-offset approach: "the offset is
		// different at different depths, which doesn't help". Clip-space jitter
		// is what every real engine's TAA does instead, and is depth-independent
		// by construction. Implemented once, centrally, in the Latte shader
		// decompiler's SET_POSITION macro (see LatteDecompilerEmitGLSLHeader.hpp)
		// - applies to every game's vertex/geometry shaders uniformly, not a
		// per-game patch. User-tested clean (static and moving) in Bayonetta 2
		// with this + jitterEnabled + motion vectors + the 60 FPS Cutscenes mod;
		// only lightly tested elsewhere (Mario 3D World, with the OLD viewport
		// jitter, before this fix existed) - watch for regressions in other games
		bool clipSpaceJitter{ true };
		float blendFactor{ 0.2f }; // weight of current frame in the history blend (no motion vectors -> favor shorter trails over maximum smoothing)
		bool debugPassthrough{ false }; // resolve outputs the current frame only (isolates history-blend issues)

		// --- motion vectors (2026-07-12): the resolve above samples history at a
		// FIXED screen position every frame and relies entirely on variance
		// clipping to reject it when wrong - not real reprojection, degrades to
		// no-AA (or worse, under heavy motion blur - see useFxaa's comment) in
		// anything that moves. GX2 never exposes the game's actual per-vertex
		// motion (same constraint that blocks true view-space reconstruction
		// everywhere else in this fork), so this estimates motion the way video
		// codecs do: a small hierarchical block search (iterative, halving step
		// size) finds, for each half-res tile, the screen-space offset where the
		// PREVIOUS frame's content best matches the CURRENT frame's - a real,
		// if approximate, per-pixel motion vector. The resolve then reprojects
		// history through it before the existing variance clip runs (unchanged,
		// still the safety net for wrong/low-confidence vectors)
		bool useMotionVectors{ true };
		// initial search step (UV units) of the hierarchical block search, halved
		// each of 6 iterations - the total reach is roughly 2x this value.
		// Larger catches faster motion, costs nothing extra (same iteration/tap
		// count either way, just wider steps) but a value much bigger than a
		// typical frame's real motion wastes the search's precision on early,
		// mostly-useless wide jumps
		float mvSearchStep{ 0.04f };
		// penalty added to a candidate's cost proportional to how far it is from
		// zero motion, so ties in flat/low-texture regions resolve toward "no
		// motion" instead of a noisy match. Weighed against typical luma-diff
		// costs (0..1 range). Raised from the original 100 untuned guess to 1000
		// (2026-07-12): with DLAA (real NVIDIA NGX) added, sub-pixel noise in the
		// motion vector estimate - invisible to our own resolve's variance clipping
		// safety net - showed up as visible blur in DLSS's temporal accumulation
		// (no equivalent safety net there). User confirmed in-game that 1000 fixes
		// the blur on a near-static scene; not yet re-validated on fast motion.
		float mvRegularization{ 1000.0f };
		// output the motion vector field as color instead of the resolved image
		// (direction -> hue-ish RG split, magnitude -> brightness) - the only way
		// to confirm the search is finding real, spatially coherent motion rather
		// than just returning near-zero everywhere (which would also look
		// "clean, no ghosting" - correct-looking for the wrong reason)
		bool mvDebugView{ false };
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

	// clip-space jitter mode only (Config::clipSpaceJitter): the render-target-pixel
	// jitter offset for the draw currently being set up, so the Vulkan renderer's
	// uniform upload (which runs per-draw, in a different file, right before the
	// actual draw call) can convert it to clip-space units and feed the shader's
	// uf_taaJitter uniform. Set by LatteRenderTarget_updateViewport right after
	// calling GetViewportJitter above - always call Set with (0,0) for draws that
	// aren't jittered so no value leaks from a previous jittered draw
	void SetCurrentDrawJitterPixels(float jitterX, float jitterY);
	void GetCurrentDrawJitterPixels(float& jitterX, float& jitterY);

	// jitter offset (render-target pixels) that scene passes used this frame; the
	// resolve pass needs it to sample the current frame at unjittered positions.
	// Returns 0,0 when jitter is disabled. Must be queried before NotifyFramePresented.
	void GetCurrentFrameJitter(float& jitterX, float& jitterY);

	// history handling for the resolve pass
	// invalidate when resolution/output changes or TAA is toggled
	void InvalidateHistory();
	bool ConsumeHistoryValidFlag(); // false on the first resolve after invalidation

	// increments whenever a draw qualifies as a full-res scene pass (the same
	// classification GetViewportJitter uses, evaluated even with jitter disabled
	// or TAA off, as long as TAA or SSAO/SSR is enabled). The TAA resolve and the
	// SSAO/SSR pass each remember the last value they saw, to tell real new
	// frames from presents that only redrew overlays (30fps cutscenes at 60Hz
	// present twice per rendered scene, with subtitle/letterbox draws in
	// between - those defeated a plain drawCallCounter guard): re-resolving one
	// of those unjitters unchanged scene content by the already-advanced jitter
	// index and re-blends it with history - the compounding sub-pixel
	// misalignment was the cutscene smear/moire corruption (confirmed via
	// automated captures: passthrough clean, full TAA smeared)
	uint32 GetSceneDrawCounter();
}
