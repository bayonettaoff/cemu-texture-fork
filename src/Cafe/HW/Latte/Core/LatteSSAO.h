#pragma once

// Screen-space ambient occlusion for Cemu (generic, no per-game knowledge).
// HBAO (horizon-based AO, Bavoil/Sainz 2008 - the algorithm NVIDIA's HBAO+
// builds on): per pixel, marches a few steps along several screen-space
// directions and accumulates the sine of the highest horizon angle each
// direction's neighbors rise above the local tangent plane. Deliberately does
// NOT reconstruct view-space position/normals, which would require the game's
// projection matrix (not exposed by GX2 as a fixed register, only baked into
// the vertex shader's own constants) - the same per-game calibration problem
// that ruled out ReShade for this fork (see screen-space-effects brief).

class LatteTexture;
class LatteTextureView;

namespace LatteSSAO
{
	struct Config
	{
		bool enabled{ false };
		float radius{ 0.02f };     // horizon march radius, in UV units
		float intensity{ 1.3f };   // occlusion strength multiplier
		float bias{ 0.0025f };     // raw depth bias to suppress self-occlusion acne
		// raw-depth acceptance window past bias: an occluder counts at full
		// strength at diff==0 fading to none at diff>=falloffRange. Raw perspective
		// depth is nonlinear (compressed far from the camera), so this same raw
		// window corresponds to a much larger world-space distance far away than
		// close up - kept small on purpose to avoid a soft dark "glow" reaching
		// past true contact points, at the cost of underestimating AO at range.
		// Value tuned in-game on the previous depth-delta AO (0.02 read as a wide
		// glow, 0.012 was barely visible); the HBAO horizon reuses it unchanged
		// as its disocclusion rejection window
		float falloffRange{ 0.017f };
		// converts a raw-depth delta into units comparable to screen-space UV
		// distance when computing the horizon elevation angle: sin(elev) =
		// h/sqrt(h^2+r^2) with h = diff*heightScale, r = sample distance in UV.
		// Higher values read the same depth step as a steeper (darker) horizon
		float heightScale{ 1.0f };
		// transparencies (water, smoke, light shafts) don't write depth, so AO
		// computed from the opaque geometry behind them showed up on top of them
		// as an out-of-place black stain. Luminance stands in for the material
		// info we don't have: AO fades out from this luminance up to 1.0 (bright
		// pixels = lit water / glow / light shafts, where black AO looks wrong)
		float aoLumFadeStart{ 0.55f };
		// AO never darkens below this floor, so residual mismatches (dark smoke
		// etc.) read as a plausible soft shade instead of pure black
		float aoFloor{ 0.3f };
		bool debugShowAOOnly{ false }; // output the raw AO term instead of color*AO

		// --- approximate view-space reconstruction, shared by every effect below.
		// GX2 never exposes the game's real projection matrix as a fixed register
		// (only baked into the game's own vertex shader constants, different per
		// shader/game - the constraint that ruled out exact view-space math
		// everywhere in this filter). Instead of reading raw depth directly
		// wherever a "how far is this in the real world" question comes up (which
		// is what made distK unreliable for cutscene close-ups framed via FOV/crop
		// rather than physical camera distance - the raw depth value doesn't know
		// the difference), assume a pinhole camera with a tunable FOV and
		// linearize depth through it. This is the same assumption ReShade-family
		// shaders (iMMERSE Launchpad/RTGI) make with their own user-facing FOV
		// slider - not genuine per-game calibration, just a generic guess that
		// makes distance-based math internally consistent regardless of framing.
		// Both fields expect in-game tuning
		float fovYDegrees{ 60.0f };  // assumed vertical FOV
		// shape parameter for depth linearization (linearDepth = 1/(1 - rawD*k)),
		// folding the game's unknown near/far plane pair into one curve (-> 1 as
		// far >> near, the common case). Replaces the old ad-hoc squared-clamped
		// raw-delta heuristic (distK) as the source of every "convert a raw depth
		// delta or UV offset into something world-proportional" computation below
		float depthLinearK{ 0.5f };
		// smoothed-normals kernel (pass S) target reach, in the same made-up
		// view-space units linearDepth produces - converted to texels per-pixel
		// using the reconstructed depth and FOV, so the kernel's SCREEN-SPACE
		// reach grows automatically for a tight FOV-cropped close-up instead of
		// staying a fixed texel count (what silently no-op'd the first fix
		// attempt: distK stayed near baseline for a shot framed via FOV, not
		// camera distance, so a distK-scaled fixed kernel barely moved).
		// 0.045 (first guess) reached noticeably farther than before but still
		// left visible facets in a tight cutscene close-up (user report) -
		// raised ~3x, then confirmed "se ve menos poligonal" (real improvement)
		// with a request for more; raised again alongside another step-count
		// bump so the wider reach doesn't sample more thinly and leave gaps
		float smoothRadiusView{ 0.35f };

		// --- SSR (screen-space reflections), hosted by the same fullscreen pass
		// (it already reads exactly the inputs SSR needs: post-TAA color + depth).
		// Normals are reconstructed from depth derivatives and rays are marched in
		// the same pseudo view space HBAO uses (screen UV x scaled raw depth), so
		// like the AO this needs no projection matrix / per-game calibration.
		// Reflection weight is fresnel-driven (grazing surfaces like floors get
		// strong reflections, camera-facing ones almost none), which doubles as
		// the material mask we don't have without a real G-buffer
		bool ssrEnabled{ false };
		float ssrStrength{ 0.8f };    // reflection blend weight multiplier (applied on top of fresnel)
		// raw-depth-to-UV scale for the SSR pseudo view space (normals + ray).
		// Independent from the AO's heightScale on purpose: raw perspective depth
		// is compressed near 1.0 where most of the scene sits, so SSR needs a much
		// larger scale for reconstructed normals to tilt meaningfully - with the
		// AO's 1.0 everything would read as camera-facing and fresnel would kill
		// all reflections. Untuned first guess, expect in-game iteration
		float ssrDepthScale{ 8.0f };
		// how far behind a surface (raw depth units) the ray may pass and still
		// count as hitting it; larger = more hits but reflections "stick" to
		// foreground objects, smaller = rays slip behind thin geometry and miss
		float ssrThickness{ 0.01f };
		// surface roughness as a reflection cone: the reflected color is blurred
		// with a radius proportional to ray travel distance (radius = roughness
		// x distance, capped in the shader). Deterministic - no noise, works the
		// same with TAA on or off. 0 = mirror-sharp
		float ssrRoughness{ 0.05f };
		bool ssrDebugView{ false };   // output only the weighted reflection term (black = no hit)

		// --- contact shadows: one-directional short-range occlusion march along a
		// fixed screen-space light direction (no light info is available without a
		// G-buffer, so "sun from above" is assumed: up-screen, slightly right).
		// Reuses the HBAO machinery (tangent plane, distance normalization)
		bool contactShadowsEnabled{ false };
		float csStrength{ 0.45f }; // how dark a full contact shadow gets (0.6 read as black blotches)
		float csDirX{ 0.3f };      // screen-space light direction (normalized in shader)
		float csDirY{ -0.85f };    // negative = light from up-screen
		float csLength{ 0.06f };   // march length in UV units (longer than the AO radius)

		// --- SSGI (single-bounce screen-space GI, RTGI-style): during the HBAO
		// march, geometry that occludes a pixel also bounces its own lit color
		// onto it, weighted by the same horizon factor as its occlusion. The
		// bounce buffer is denoised by the shared bilateral blur pass and added
		// on top of the AO-darkened image (colored light bleeding into corners)
		bool giEnabled{ false };
		float giStrength{ 0.5f }; // bounce intensity multiplier (0.8 added a smoky glow)
	};

	Config& GetConfig();

	// true when the fullscreen pass has any work to do (AO and/or SSR) - the
	// call sites that gate Apply()/present on the pass check this, not .enabled
	bool AnyEffectEnabled();

	// forwards to VulkanSSAOFilter's cached-depth-fallback invalidation (Vulkan
	// only; a no-op otherwise). Called from LatteTexture_Delete
	void NotifyTextureDeletion(LatteTexture* texture);

	// called from LatteMRT::SetDepthAndStencilAttachment on every depth bind, so
	// the filter can capture the scene's depth buffer DURING the frame instead of
	// hoping it is still bound at scanout-swap time. Cutscene camera cuts swap in
	// freshly created depth buffers; the old swap-time-only + cached-view scheme
	// lost track of depth across those cuts and the effects visibly dropped out
	void NotifyDepthBind(LatteTextureView* view);
}
