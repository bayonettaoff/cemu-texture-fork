#pragma once

// Geometry amplification for Cemu (generic, no per-game knowledge).
// 4K texture replacement exposes the low poly-count of 2009-2014 character
// meshes - visible as a faceted (prism-like) silhouette, confirmed on
// Bayonetta 2's arms (not a shading bug - the geometry itself is coarse).
// See Downloads/cemu-geometry-amplification-brief.md for the full design.
//
// Approach: an injected geometry shader subdivides each eligible triangle
// 1->4 (edge midpoints) and displaces the 3 new midpoint vertices along a
// Phong-tessellation-style projection (Boubekeur & Alexa 2008) to round the
// silhouette instead of just adding flat sub-triangles. Original corner
// vertices are never moved, so amplified draws still line up exactly with
// anything unamplified sharing those vertices.
//
// Phong tessellation needs a per-vertex normal, but GX2 shaders have no fixed
// "this varying is the normal" semantic - which one it is differs per game
// and per shader. Rather than requiring the user to hand-identify it (the
// brief's original "manifest" idea), the generated shader tries every
// candidate passthrough varying at runtime and picks whichever one is
// closest to unit length when read as a direction - same "no per-game
// calibration" philosophy already used by LatteSSAO's HBAO/SSR. If nothing
// scores close enough to be plausible, displacement is skipped (flat
// subdivision only - safe no-op, more triangles but same silhouette).
namespace LatteGeometryAmp
{
	struct Config
	{
		bool  enabled = false;    // master switch, Debug menu + env var CEMU_GEOAMP
		// displacement strength as a 0..1 blend between the flat midpoint and
		// the full Phong-projected point. 0 = pure flat subdivision (topology
		// only, silhouette unchanged - safe no-op, useful as an A/B baseline).
		// 2026-07-14: first live test at 0.35 (applied globally, every
		// triangle draw) produced catastrophic whole-scene geometry
		// destruction, not a subtle silhouette change. Defaulted to 0 to
		// isolate the cause - and it turned out to persist even at 0
		// (topology-only subdivision, no displacement math involved at all),
		// which proved the bug wasn't the Phong projection - it was
		// VulkanPipelineCompiler.cpp's gaPhongMid dividing by w and
		// re-homogenizing UNCONDITIONALLY, which only equals the correct
		// clip-space linear midpoint when wa==wb. Small/uniform-depth
		// triangles (character meshes) hid the error; large environment
		// surfaces spanning a big depth range did not. Fixed: gaPhongMid now
		// always uses the correct (Pa+Pb)*0.5 clip-space midpoint as its
		// base, confirmed safe in-game at factor=0 (map no longer breaks).
		// The perspective-divided projection math is now only used to
		// compute a small ADDITIVE delta on top of that base when actually
		// displacing. Raising factor above 0 is safe from THIS bug, but the
		// ORIGINAL open question - whether the auto-detected "normal" is
		// even in a usable coordinate space for the projection to look
		// geometrically correct (vs. just bounded-but-arbitrary) - is still
		// untested. Start any re-test with a small value (~0.1-0.2)
		float factor = 0.0f;
		// runtime auto-detect accepts a candidate varying as "the normal" only
		// if its average length across the triangle's 3 corners falls within
		// this distance of 1.0. Too loose -> displaces using garbage data
		// (UVs, colors) and warps the mesh; too tight -> real normals that
		// aren't perfectly unit-length (compression, skinning) get rejected
		float normalDetectTolerance = 0.35f;
	};

	Config& GetConfig();
}
