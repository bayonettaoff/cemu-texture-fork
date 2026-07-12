#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanSSAOFilter.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanTAAFilter.h"
#if BOOST_OS_WINDOWS
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanDLSSFilter.h"
#endif
#include "Cafe/HW/Latte/Renderer/Vulkan/LatteTextureVk.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/LatteTextureViewVk.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/RendererShaderVk.h"
#include "Cafe/HW/Latte/Core/LatteSSAO.h"
#include "Cafe/HW/Latte/Core/LatteTAA.h"
#include "Cafe/HW/Latte/Core/LatteCachedFBO.h"
#include "Cafe/HW/Latte/Core/Latte.h"

VulkanSSAOFilter& VulkanSSAOFilter::GetInstance()
{
	static VulkanSSAOFilter s_instance;
	return s_instance;
}

uint32 VulkanSSAOFilter::FindMemoryType(VkPhysicalDevice physDev, uint32 typeBits, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);
	for (uint32 i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeBits & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	cemuLog_log(LogType::Force, "VulkanSSAOFilter: no suitable memory type found");
	return 0;
}

static void _ssaoBarrierImage(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
							  VkImageLayout oldLayout, VkImageLayout newLayout,
							  VkAccessFlags srcAccess, VkAccessFlags dstAccess,
							  VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
							  uint32 baseMip = 0, uint32 baseLayer = 0)
{
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = aspect;
	barrier.subresourceRange.baseMipLevel = baseMip;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = baseLayer;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = srcAccess;
	barrier.dstAccessMask = dstAccess;
	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

bool VulkanSSAOFilter::CreateShaders()
{
	const char* vsSrc =
		"#version 450\r\n"
		"layout(location = 0) out vec2 passUV;\r\n"
		"void main(){\r\n"
		"vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));\r\n"
		"passUV = pos;\r\n"
		"gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);\r\n"
		"}\r\n";

	// HBAO (horizon-based AO, Bavoil/Sainz 2008 - the algorithm NVIDIA's HBAO+
	// builds on; the HBAO+ library itself is a closed D3D/GL-only binary, so the
	// algorithm is reimplemented here). Instead of counting how many disk samples
	// are closer than expected (the old Crytek-style depth-delta AO), each pixel
	// marches a few steps along several screen-space directions and finds, per
	// direction, the highest "horizon" - the largest elevation angle a neighbor
	// rises above the local tangent plane. Occlusion is the mean of sin(horizon)
	// over all directions, which approximates how much of the hemisphere the
	// surrounding geometry blocks. This grades occlusion continuously with the
	// occluder's height instead of the old binary near/far vote, so contact
	// shadows darken progressively toward corners rather than saturating.
	// Still no view-space reconstruction: GX2 never exposes the projection
	// matrix as a fixed register (it's baked into the game's own vertex shader
	// constants), so "height" is the raw-depth delta above the tangent plane and
	// the march radius is constant in UV space - same known simplification as
	// before, no per-game calibration needed.
	const char* fsSrc =
		"#version 450\r\n"
		"layout(location = 0) in vec2 passUV;\r\n"
		"layout(location = 0) out vec4 outColor;\r\n"  // rgb = SSR-blended color, a = raw AO
		"layout(location = 1) out vec4 outGI;\r\n"     // rgb = SSGI bounce radiance
		"layout(binding = 0) uniform sampler2D texColor;\r\n"
		"layout(binding = 1) uniform sampler2D texDepth;\r\n"
		"layout(binding = 2) uniform sampler2D texNormal;\r\n" // smoothed normals (pass S output)
		"layout(push_constant) uniform pushConstants {\r\n"
		"float radiusU;\r\n"
		"float radiusV;\r\n"
		"float intensity;\r\n"
		"float bias;\r\n"
		"float debugShowAOOnly;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float width;\r\n"
		"float height;\r\n"
		"float falloffRange;\r\n"
		"float heightScale;\r\n"
		"float ssrStrength;\r\n"
		"float ssrDepthScale;\r\n"
		"float ssrThickness;\r\n"
		"float ssrDebug;\r\n"
		"float aoLumFadeStart;\r\n"
		"float aoFloor;\r\n"
		"float ssrRoughness;\r\n"
		"float csStrength;\r\n"
		"float csDirX;\r\n"
		"float csDirY;\r\n"
		"float csLength;\r\n"
		"float giStrength;\r\n"
		"float depthLinearK;\r\n"
		"float fovScale;\r\n"
		"float smoothRadiusView;\r\n"
		"}uf_pc;\r\n"
		"const float PI2 = 6.28318530718;\r\n"
		"const int NUM_DIRS = 6;\r\n"
		"const int NUM_STEPS = 4;\r\n"
		"const int NUM_SSR_STEPS = 16;\r\n"
		"const int NUM_SSR_REFINE = 4;\r\n"
		"const float SSR_MAX_DIST = 0.75;\r\n"  // max ray travel in UV units
		"const float SSR_EDGE_FADE = 0.08;\r\n" // fade band at screen borders (UV units)
		"void main(){\r\n"
		// color is sampled at passUV unchanged (it's either TAA's already-stable
		// resolved output, or the raw scanout exactly as it would be presented
		// without AO - neither should be re-jittered here). Depth is different:
		// it's always the raw, still-jittered game depth, and this pass's fixed
		// per-pixel sample pattern needs a stable reference frame or the AO term
		// flickers at every depth discontinuity as the jitter shifts geometry -
		// unjitter it the same way VulkanTAAFilter's resolve does for color
		"vec4 color = texture(texColor, passUV);\r\n"
		"vec2 uvC = passUV + vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"float centerDepth = texture(texDepth, uvC).r;\r\n"
		// far plane / cleared background (skybox, empty depth): nothing to occlude
		// or reflect off of. Keep the debug views readable there too: SSR debug
		// shows black (no reflection), AO debug shows white (fully unoccluded)
		"if (centerDepth >= 0.9999){\r\n"
		"outGI = vec4(0.0);\r\n" // MRT outputs must be written on every path
		"if (uf_pc.ssrDebug > 0.5) outColor = vec4(0.0, 0.0, 0.0, 1.0);\r\n"
		"else if (uf_pc.debugShowAOOnly > 0.5) outColor = vec4(1.0);\r\n"
		"else outColor = vec4(color.rgb, 1.0);\r\n" // alpha carries the AO term now: sky = unoccluded
		"return; }\r\n"
		// local depth slope (raw depth change per UV unit): the tangent plane every
		// sample is measured against. Referencing the tangent plane (instead of
		// flat centerDepth) makes a smoothly curved surface predict itself and
		// stop reading as its own occluder (the "cylinder halo" fix), and doubles
		// as HBAO's tangent angle - elevations are measured relative to it, so
		// the horizon starts at the surface's own slope, not at screen-plane zero.
		// Computed from 4 explicit depth taps instead of dFdx/dFdy: quad
		// derivatives are constant across each 2x2 pixel block, which made the
		// SSR normals (and thus the reflections) look faceted/"polygonal" - taps
		// give a genuinely per-pixel slope. Of the two one-sided differences per
		// axis, keep the smaller magnitude so a silhouette edge on one side
		// doesn't skew the slope of the surface the pixel actually belongs to
		// 2-texel baseline (with the linear sampler averaging neighbors) low-passes
		// the slope estimate: a 1-texel baseline tracked every polygon facet edge
		// exactly, which the SSR normals turned into visibly faceted reflections
		"float texelU = 1.0 / uf_pc.width;\r\n"
		"float texelV = 1.0 / uf_pc.height;\r\n"
		"float dL = texture(texDepth, uvC - vec2(2.0 * texelU, 0.0)).r;\r\n"
		"float dR = texture(texDepth, uvC + vec2(2.0 * texelU, 0.0)).r;\r\n"
		"float dU = texture(texDepth, uvC - vec2(0.0, 2.0 * texelV)).r;\r\n"
		"float dD = texture(texDepth, uvC + vec2(0.0, 2.0 * texelV)).r;\r\n"
		"float dduP = dR - centerDepth;\r\n"
		"float dduM = centerDepth - dL;\r\n"
		"float ddu = (abs(dduM) < abs(dduP) ? dduM : dduP) * uf_pc.width * 0.5;\r\n"
		"float ddvP = dD - centerDepth;\r\n"
		"float ddvM = centerDepth - dU;\r\n"
		"float ddv = (abs(ddvM) < abs(ddvP) ? ddvM : ddvP) * uf_pc.height * 0.5;\r\n"
		// distance normalization: raw perspective depth compresses with distance
		// (a fixed world-space step produces a raw delta that shrinks roughly with
		// (1-d)^2), so fixed raw thresholds made both effects fade out away from
		// the camera - visible only in cutscene close-ups, nearly absent in
		// gameplay's wider camera. Dividing every depth delta by this factor makes
		// bias/falloff/thickness behave world-proportionally with distance.
		// Reference (distK=1, where the in-game-tuned constants keep their exact
		// meaning) sits at d=0.85: perspective depth is heavily front-loaded, most
		// scene content lives at high raw d - a first version referenced d=0.5 and
		// amplified typical-content deltas ~100x, blowing past the tuned acceptance
		// windows (AO vanished) and flattening SSR normals sideways (fresnel ~1
		// everywhere = full-cost march on every pixel, the reported perf drop).
		// The clamp bounds the correction to ~6x either way: full proportionality
		// mid-range, graceful degradation at the extremes instead of runaway
		// amplification of far-plane precision noise
		// approximate view-space depth linearization (assumed FOV, see
		// LatteSSAO::Config::fovYDegrees/depthLinearK): linearDepth is
		// proportional to real distance from the camera, unlike raw Vulkan
		// depth which compresses nonlinearly. distK is its local derivative
		// w.r.t. raw depth (how much real distance one raw-depth unit
		// represents here) - dividing a raw depth delta by distK converts it
		// into the same world-proportional units every bias/falloff/thickness
		// constant below is tuned in, exactly like the old ad-hoc squared-
		// clamped heuristic did, but grounded in an actual depth curve instead
		// of a guess referenced against raw depth alone (which carries no
		// information about camera FOV/framing - a cutscene close-up shot via
		// a tight FOV crop has perfectly normal raw depth values despite
		// needing much wider on-screen kernels, which silently defeated the
		// old formula)
		"float linZ = 1.0 / max(1.0 - centerDepth * uf_pc.depthLinearK, 1e-4);\r\n"
		"float distK = 1.0 / max(uf_pc.depthLinearK * linZ * linZ, 1e-4);\r\n"
		// smoothed normal and its tangent-plane slope (raw depth per UV) drive
		// the expected-depth prediction of EVERY effect now, not just the SSR:
		// with the raw per-facet slopes, the AO/GI/contact terms were constant
		// inside each polygon facet and jumped at facet edges, which the blur
		// pass then polished into hard glossy patches - the reported "crystal"
		// look on faces and clothing. The smoothed plane varies continuously
		// across facets, so the effect terms do too
		"vec3 nS = normalize(texture(texNormal, passUV).xyz * 2.0 - 1.0);\r\n"
		"vec2 slopeS = -nS.xy / max(nS.z, 0.05) * (distK / uf_pc.ssrDepthScale);\r\n"
		// interleaved gradient noise (Jimenez): per-pixel rotation of the whole
		// direction set plus a per-pixel offset of the step positions along each
		// ray, trading banding/ring artifacts for high-frequency noise without a
		// separate noise texture (HBAO+ does the same with an interleaved pattern)
		"float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));\r\n"
		"float rotation = ign * PI2;\r\n"
		"float cs = cos(rotation);\r\n"
		"float sn = sin(rotation);\r\n"
		"float stepJitter = fract(ign * 7.545);\r\n"
		// ---- SSR (screen-space reflections) ----
		// pseudo view space (screen UV x raw depth scaled into comparable units,
		// same trick as the HBAO below but with its own scale): normal comes from
		// the depth derivatives, the reflection of the nominal view ray (0,0,1)
		// is marched through the depth buffer, and on a hit the color there is
		// blended in weighted by fresnel - grazing surfaces (floors seen at an
		// angle) reflect strongly, camera-facing ones barely at all, which also
		// stands in for the material mask a real G-buffer would provide
		"vec3 finalRGB = color.rgb;\r\n"
		"if (uf_pc.ssrStrength > 0.0){\r\n"
		"float S = uf_pc.ssrDepthScale;\r\n"
		// second, wider slope estimate just for the reflection normal: central
		// difference over a wide baseline (widened from 8 texels - too narrow
		// for a cutscene close-up's on-screen facet size, same lesson as the
		// pass S kernel above). Min-magnitude combine with the near-tap slope
		// smooths polygon facet creases (the central difference averages the
		// two faces, so the normal field stops jumping at every facet edge -
		// the remaining "polygons instead of reflections" look) while staying
		// safe at silhouettes, where the wide difference is huge and the
		// near-tap slope wins
		"float dL4 = texture(texDepth, uvC - vec2(12.0 * texelU, 0.0)).r;\r\n"
		"float dR4 = texture(texDepth, uvC + vec2(12.0 * texelU, 0.0)).r;\r\n"
		"float dU4 = texture(texDepth, uvC - vec2(0.0, 12.0 * texelV)).r;\r\n"
		"float dD4 = texture(texDepth, uvC + vec2(0.0, 12.0 * texelV)).r;\r\n"
		"float dduW = (dR4 - dL4) * uf_pc.width * (1.0 / 24.0);\r\n"
		"float ddvW = (dD4 - dU4) * uf_pc.height * (1.0 / 24.0);\r\n"
		// reflection normal = the smoothed normal (Launchpad style): depth only
		// contains the game's real flat triangles, the smoothing pass
		// reconstructs the lost smoothing groups so the reflection direction
		// varies continuously instead of jumping at every polygon edge
		"vec3 n = nS;\r\n"
		// curvature mask: where the raw facet normal disagrees with the smoothed
		// one, this is low-poly curved geometry (characters, props) whose
		// depth-derived reflections can only ever look like glass shards - fade
		// them out entirely. On true planes (floors, walls) both estimators
		// agree and the mask passes
		"vec3 nRaw = normalize(vec3(-ddu * S / distK, -ddv * S / distK, 1.0));\r\n"
		"float curv = smoothstep(0.3, 0.8, dot(nRaw, n));\r\n"
		// planarity mask: only locally planar surfaces (floors, walls) get
		// reflections. On curved/low-poly meshes the depth-reconstructed normal
		// is piecewise-constant per polygon facet, so the reflection "mirrors off
		// the mesh" facet by facet (as reported). A plane measures the same slope
		// at any baseline, so disagreement between the 2-texel and 8-texel
		// estimates (in the same tilt units as the normal) marks non-planar
		// geometry and fades its reflection out entirely
		"float planarDx = abs(dduW - ddu) * S / distK;\r\n"
		"float planarDy = abs(ddvW - ddv) * S / distK;\r\n"
		// pure fresnel again: the 0.2 floor tried to make SSR more visible but
		// turned dark clothing into mirrors ("crystal" look). Reflections belong
		// on planes seen at an angle - planarity + curvature masks enforce that
		"float planar = 1.0 - clamp((planarDx + planarDy) * 2.0, 0.0, 1.0);\r\n"
		"float fres = clamp(1.0 - n.z, 0.0, 1.0) * planar * curv;\r\n"
		"vec3 R = reflect(vec3(0.0, 0.0, 1.0), n);\r\n"
		"float rxyLen = length(R.xy);\r\n"
		"vec3 ssrDebugColor = vec3(0.0);\r\n"
		"if (fres > 0.001 && rxyLen > 0.001){\r\n"
		// normalize so t advances in screen-space UV units; jittered start point
		// dithers the stair-stepping of the coarse march into noise
		"vec3 rayDir = R / rxyLen;\r\n"
		"vec3 rayOrigin = vec3(uvC, centerDepth * S);\r\n"
		"float stepLen = SSR_MAX_DIST / float(NUM_SSR_STEPS);\r\n"
		"float t = stepLen * (0.25 + 0.75 * stepJitter);\r\n"
		"float tPrev = 0.0;\r\n"
		"float hitT = -1.0;\r\n"
		"for (int i = 0; i < NUM_SSR_STEPS; i++){\r\n"
		"vec3 p = rayOrigin + rayDir * t;\r\n"
		"if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z < 0.0) break;\r\n"
		"float sceneRaw = texture(texDepth, p.xy).r;\r\n"
		// sky/cleared depth can't be a reflected surface; and a ray that passed
		// farther behind a surface than ssrThickness is a disocclusion (it slipped
		// behind a foreground object), keep marching instead of latching onto it
		"if (sceneRaw < 0.9999){\r\n"
		"float delta = p.z - sceneRaw * S;\r\n"
		// thickness window distance-normalized like everything else, so far
		// surfaces accept hits with the same world-proportional tolerance
		"if (delta > 0.0 && delta / distK < uf_pc.ssrThickness * S){ hitT = t; break; }\r\n"
		"}\r\n"
		"tPrev = t;\r\n"
		"t += stepLen;\r\n"
		"}\r\n"
		"if (hitT > 0.0){\r\n"
		// binary refinement between the last miss and the hit for a crisp contact
		"float lo = tPrev;\r\n"
		"float hi = hitT;\r\n"
		"for (int i = 0; i < NUM_SSR_REFINE; i++){\r\n"
		"float mid = 0.5 * (lo + hi);\r\n"
		"vec3 p = rayOrigin + rayDir * mid;\r\n"
		"float sceneRaw = texture(texDepth, p.xy).r;\r\n"
		"if (p.z - sceneRaw * S > 0.0) hi = mid; else lo = mid;\r\n"
		"}\r\n"
		"vec3 pHit = rayOrigin + rayDir * hi;\r\n"
		// ray marched in (jittered) depth space; color lives in unjittered space
		"vec2 hitUV = pHit.xy - vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"vec2 edge = min(hitUV, vec2(1.0) - hitUV);\r\n"
		"float edgeFade = clamp(min(edge.x, edge.y) / SSR_EDGE_FADE, 0.0, 1.0);\r\n"
		"float distFade = 1.0 - clamp(hi / SSR_MAX_DIST, 0.0, 1.0);\r\n"
		"float w = clamp(uf_pc.ssrStrength * fres * edgeFade * distFade, 0.0, 1.0);\r\n"
		// cone-approximated roughness: blur the reflected color with a radius
		// proportional to how far the ray travelled, like a rough surface's
		// widening reflection cone - deterministic and noise-free. (A previous
		// attempt jittered the reflection direction per pixel expecting TAA to
		// average it, but this pass runs AFTER TAA's resolve and the jitter
		// pattern is static per pixel, so it just stamped fixed noise onto the
		// reflection - visible with TAA on or off, as reported)
		"float blurR = min(uf_pc.ssrRoughness * hi, 0.02);\r\n"
		"vec3 reflColor = texture(texColor, hitUV).rgb * 0.2;\r\n"
		"reflColor += texture(texColor, hitUV + vec2( blurR,  blurR)).rgb * 0.2;\r\n"
		"reflColor += texture(texColor, hitUV + vec2(-blurR,  blurR)).rgb * 0.2;\r\n"
		"reflColor += texture(texColor, hitUV + vec2( blurR, -blurR)).rgb * 0.2;\r\n"
		"reflColor += texture(texColor, hitUV + vec2(-blurR, -blurR)).rgb * 0.2;\r\n"
		"finalRGB = mix(finalRGB, reflColor, w);\r\n"
		"ssrDebugColor = reflColor * w;\r\n"
		"}\r\n"
		"}\r\n"
		"if (uf_pc.ssrDebug > 0.5){ outColor = vec4(ssrDebugColor, 1.0); outGI = vec4(0.0); return; }\r\n"
		"}\r\n"
		// ---- HBAO + SSGI (shared march) ----
		// SSGI piggybacks on the exact same depth taps: geometry that rises above
		// the tangent plane doesn't just occlude the sky - it also bounces its own
		// lit color onto this pixel (RTGI-style single-bounce, using the screen as
		// the light source). One extra color tap per accepted occluder sample
		"float occlusion = 0.0;\r\n"
		"vec3 giSum = vec3(0.0);\r\n"
		"if (uf_pc.intensity > 0.0 || uf_pc.giStrength > 0.0){\r\n"
		"for (int d = 0; d < NUM_DIRS; d++){\r\n"
		"float baseAngle = float(d) * (PI2 / float(NUM_DIRS));\r\n"
		"vec2 dir = vec2(cos(baseAngle), sin(baseAngle));\r\n"
		"vec2 rdir = vec2(dir.x*cs - dir.y*sn, dir.x*sn + dir.y*cs);\r\n"
		"float horizonSin = 0.0;\r\n"
		"for (int s = 0; s < NUM_STEPS; s++){\r\n"
		// march outward; rFrac in (0..1] of the full radius. A jittered first step
		// near the pixel captures fine contact shadows, later steps the wide ones
		"float rFrac = (float(s) + stepJitter) / float(NUM_STEPS);\r\n"
		"vec2 offset = rdir * rFrac * vec2(uf_pc.radiusU, uf_pc.radiusV);\r\n"
		"float sampleDepth = texture(texDepth, uvC + offset).r;\r\n"
		// elevation above the SMOOTHED tangent plane. Smaller raw depth = nearer
		// camera (Vulkan depth range 0=near..1=far, no reversed-Z in Cemu's
		// Latte->Vulkan mapping), so positive diff = sample rises toward the
		// camera = occluder
		"float expectedDepth = centerDepth + slopeS.x * offset.x + slopeS.y * offset.y;\r\n"
		// distance-normalized so bias/falloff/height mean the same at any range
		"float diff = (expectedDepth - sampleDepth) / distK;\r\n"
		"if (diff > uf_pc.bias){\r\n"
		// raw-depth acceptance window (halo guard): raw depth is nonlinear and the
		// march radius isn't world-space, so a large raw delta is almost always a
		// disocclusion (unrelated foreground object), not a contact occluder -
		// fade it out entirely past falloffRange, exactly like the previous AO did
		"float depthFalloff = 1.0 - clamp(diff / uf_pc.falloffRange, 0.0, 1.0);\r\n"
		// sine of the elevation angle in the pseudo view space (screen UV plane x
		// raw depth scaled by heightScale into comparable units) - HBAO's
		// sin(horizon)-sin(tangent), with sin(tangent)=0 by construction because
		// diff is already measured relative to the tangent plane
		"float h = diff * uf_pc.heightScale;\r\n"
		"float rScreen = rFrac * uf_pc.radiusV;\r\n"
		"float sinElev = h / sqrt(h*h + rScreen*rScreen);\r\n"
		// HBAO's radial attenuation W(r) = 1 - (r/R)^2: near occluders matter more
		"float atten = 1.0 - rFrac * rFrac;\r\n"
		"float contrib = sinElev * atten * depthFalloff;\r\n"
		"horizonSin = max(horizonSin, contrib);\r\n"
		// SSGI: the occluder's lit color arrives as bounce light, weighted by the
		// same solid-angle-ish factor as its occlusion. Color sampled in unjittered
		// space (same frame the AO's color input lives in)
		"if (uf_pc.giStrength > 0.0) giSum += texture(texColor, passUV + offset).rgb * contrib;\r\n"
		"}\r\n"
		"}\r\n"
		"occlusion += horizonSin;\r\n"
		"}\r\n"
		"occlusion = occlusion / float(NUM_DIRS);\r\n"
		"}\r\n"
		"float ao = clamp(1.0 - occlusion * uf_pc.intensity, 0.0, 1.0);\r\n"
		// ---- contact shadows ----
		// one-directional occlusion march along an assumed screen-space light
		// direction: geometry rising above the tangent plane toward the light,
		// within csLength, casts a soft short-range shadow onto this pixel.
		// Same primitives as the HBAO (tangent plane, distK); folded into ao so
		// the luminance protection and floor below also bound the shadows
		"if (uf_pc.csStrength > 0.0){\r\n"
		"vec2 csDir = normalize(vec2(uf_pc.csDirX, uf_pc.csDirY));\r\n"
		"csDir.x *= uf_pc.height / uf_pc.width;\r\n" // aspect-correct like radiusU
		"float cs = 0.0;\r\n"
		"for (int s = 0; s < 8; s++){\r\n"
		"float rF = (float(s) + stepJitter) / 8.0;\r\n"
		"vec2 off = csDir * rF * uf_pc.csLength;\r\n"
		"float sd = texture(texDepth, uvC + off).r;\r\n"
		"float expd = centerDepth + slopeS.x * off.x + slopeS.y * off.y;\r\n"
		"float dcs = (expd - sd) / distK;\r\n"
		"if (dcs > uf_pc.bias){\r\n"
		// wider acceptance than the AO (shadow casters sit farther than contact
		// occluders), still fading out to reject full disocclusions. Tightened
		// from 3x: the wide window read as smoky black blotches instead of
		// contact shadows
		"float fall = 1.0 - clamp(dcs / (uf_pc.falloffRange * 2.0), 0.0, 1.0);\r\n"
		"cs = max(cs, fall * (1.0 - rF * rF));\r\n"
		"}\r\n"
		"}\r\n"
		"ao *= 1.0 - cs * uf_pc.csStrength;\r\n"
		"}\r\n"
		// transparencies (water, smoke, light shafts) don't write depth, so the
		// AO term computed from the opaque geometry BEHIND them landed on top as
		// an out-of-place black stain. There's no material info to detect them
		// directly, so protect by luminance instead: fade the AO out as the pixel
		// brightens (bright = lit water / glow / light shaft, exactly where black
		// AO looks wrong), and never darken below aoFloor anywhere so leftover
		// mismatches read as a soft shade instead of pure black
		"float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));\r\n"
		"ao = mix(ao, 1.0, smoothstep(uf_pc.aoLumFadeStart, 1.0, lum));\r\n"
		"ao = max(ao, uf_pc.aoFloor);\r\n"
		// SSGI: normalize the bounce sum over all taps and tint it toward the
		// receiving surface's own color (a red carpet mostly reflects red bounce
		// back up - crude albedo stand-in without a G-buffer)
		"vec3 gi = vec3(0.0);\r\n"
		"if (uf_pc.giStrength > 0.0){\r\n"
		"gi = giSum / float(NUM_DIRS * NUM_STEPS) * uf_pc.giStrength;\r\n"
		"gi *= 0.35 + 0.65 * color.rgb;\r\n"
		// hard cap: in very bright scenes (white rooms) the uncapped bounce
		// stacked into glowing patches that read as glass highlights
		"gi = min(gi, vec3(0.35));\r\n"
		"}\r\n"
		// the AO term is NOT applied here: it goes out in alpha, raw, and the blur
		// pass multiplies it in after the depth-aware denoise (the GI target gets
		// the same treatment). In the AO debug view rgb is plain white so the
		// blurred grayscale term is what reaches the screen (through the same blur
		// that shapes the real image), with the GI silenced to not tint it
		"if (uf_pc.debugShowAOOnly > 0.5){\r\n"
		"outColor = vec4(1.0, 1.0, 1.0, ao);\r\n"
		"outGI = vec4(0.0);\r\n"
		"} else {\r\n"
		// AO will darken the SSR-blended color (reflections are part of the
		// incoming light, occlusion applies to all of it)
		"outColor = vec4(finalRGB, ao);\r\n"
		"outGI = vec4(gi, 1.0);\r\n"
		"}\r\n"
		"}\r\n";

	// pass B: depth-aware (cross-bilateral) 5x5 blur of the AO term + composite.
	// The effects pass dithers its sample patterns with IGN to avoid banding, but
	// with nothing temporal running after this filter that dither used to reach
	// the screen as a fixed cross-hatch stamped over everything (faces, floors -
	// exactly what HBAO+ solves with its own blur stage). Blur only the AO alpha
	// channel; rgb (game color + SSR blend) passes through the center tap
	// untouched so no game detail is smeared. Depth weighting keeps the AO from
	// bleeding across silhouettes; reuses falloffRange as the rejection window
	// and the same distK normalization so the window is distance-proportional
	const char* blurSrc =
		"#version 450\r\n"
		"layout(location = 0) in vec2 passUV;\r\n"
		"layout(location = 0) out vec4 outColor;\r\n"
		"layout(binding = 0) uniform sampler2D texAO;\r\n"   // pass A output: rgb color, a = raw AO
		"layout(binding = 1) uniform sampler2D texDepth;\r\n"
		"layout(binding = 2) uniform sampler2D texGI;\r\n"   // pass A MRT output: SSGI radiance
		"layout(push_constant) uniform pushConstants {\r\n"
		"float radiusU;\r\n"
		"float radiusV;\r\n"
		"float intensity;\r\n"
		"float bias;\r\n"
		"float debugShowAOOnly;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float width;\r\n"
		"float height;\r\n"
		"float falloffRange;\r\n"
		"float heightScale;\r\n"
		"float ssrStrength;\r\n"
		"float ssrDepthScale;\r\n"
		"float ssrThickness;\r\n"
		"float ssrDebug;\r\n"
		"float aoLumFadeStart;\r\n"
		"float aoFloor;\r\n"
		"float ssrRoughness;\r\n"
		"float csStrength;\r\n"
		"float csDirX;\r\n"
		"float csDirY;\r\n"
		"float csLength;\r\n"
		"float giStrength;\r\n"
		"float depthLinearK;\r\n"
		"float fovScale;\r\n"
		"float smoothRadiusView;\r\n"
		"}uf_pc;\r\n"
		"void main(){\r\n"
		"vec4 center = texture(texAO, passUV);\r\n"
		"vec2 uvC = passUV + vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"float centerDepth = texture(texDepth, uvC).r;\r\n"
		// approximate view-space depth linearization (assumed FOV, see
		// LatteSSAO::Config::fovYDegrees/depthLinearK): linearDepth is
		// proportional to real distance from the camera, unlike raw Vulkan
		// depth which compresses nonlinearly. distK is its local derivative
		// w.r.t. raw depth (how much real distance one raw-depth unit
		// represents here) - dividing a raw depth delta by distK converts it
		// into the same world-proportional units every bias/falloff/thickness
		// constant below is tuned in, exactly like the old ad-hoc squared-
		// clamped heuristic did, but grounded in an actual depth curve instead
		// of a guess referenced against raw depth alone (which carries no
		// information about camera FOV/framing - a cutscene close-up shot via
		// a tight FOV crop has perfectly normal raw depth values despite
		// needing much wider on-screen kernels, which silently defeated the
		// old formula)
		"float linZ = 1.0 / max(1.0 - centerDepth * uf_pc.depthLinearK, 1e-4);\r\n"
		"float distK = 1.0 / max(uf_pc.depthLinearK * linZ * linZ, 1e-4);\r\n"
		"float texelU = 1.0 / uf_pc.width;\r\n"
		"float texelV = 1.0 / uf_pc.height;\r\n"
		"float aoSum = 0.0;\r\n"
		"vec3 giSum = vec3(0.0);\r\n"
		"float wSum = 0.0;\r\n"
		"for (int dy = -2; dy <= 2; dy++){\r\n"
		"for (int dx = -2; dx <= 2; dx++){\r\n"
		// 2-texel stride: at 1-texel the +-2px kernel left too much of the IGN
		// dither through at 4K ("still lots of noise")
		"vec2 off = vec2(float(dx) * texelU, float(dy) * texelV) * 2.0;\r\n"
		"float a = texture(texAO, passUV + off).a;\r\n"
		"float d = texture(texDepth, uvC + off).r;\r\n"
		// taps across a depth discontinuity belong to other surfaces - fade them
		// out linearly over the same distance-normalized window the AO itself uses
		"float dz = abs(d - centerDepth) / distK;\r\n"
		"float w = 1.0 - clamp(dz / uf_pc.falloffRange, 0.0, 1.0);\r\n"
		"aoSum += a * w;\r\n"
		"if (uf_pc.giStrength > 0.0) giSum += texture(texGI, passUV + off).rgb * w;\r\n"
		"wSum += w;\r\n"
		"}\r\n"
		"}\r\n"
		// center tap always contributes at weight 1, so wSum >= 1
		"float ao = aoSum / max(wSum, 0.0001);\r\n"
		// bounce light is added on top of the occluded direct light; the same
		// bilateral weights denoise it (it carries the same IGN dither as the AO)
		"vec3 gi = giSum / max(wSum, 0.0001);\r\n"
		"outColor = vec4(center.rgb * ao + gi, 1.0);\r\n"
		"}\r\n";

	// pass N: reconstruct per-pixel normals from depth (same estimator the SSR
	// used inline: 4-tap min-magnitude one-sided differences over a 2-texel
	// baseline, in the SSR's pseudo view space). Written at passUV so the normal
	// image is indexed in unjittered space; consumers sample it at passUV
	const char* normalSrc =
		"#version 450\r\n"
		"layout(location = 0) in vec2 passUV;\r\n"
		"layout(location = 0) out vec4 outColor;\r\n"
		"layout(binding = 1) uniform sampler2D texDepth;\r\n"
		"layout(push_constant) uniform pushConstants {\r\n"
		"float radiusU;\r\n"
		"float radiusV;\r\n"
		"float intensity;\r\n"
		"float bias;\r\n"
		"float debugShowAOOnly;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float width;\r\n"
		"float height;\r\n"
		"float falloffRange;\r\n"
		"float heightScale;\r\n"
		"float ssrStrength;\r\n"
		"float ssrDepthScale;\r\n"
		"float ssrThickness;\r\n"
		"float ssrDebug;\r\n"
		"float aoLumFadeStart;\r\n"
		"float aoFloor;\r\n"
		"float ssrRoughness;\r\n"
		"float csStrength;\r\n"
		"float csDirX;\r\n"
		"float csDirY;\r\n"
		"float csLength;\r\n"
		"float giStrength;\r\n"
		"float depthLinearK;\r\n"
		"float fovScale;\r\n"
		"float smoothRadiusView;\r\n"
		"}uf_pc;\r\n"
		"void main(){\r\n"
		"vec2 uvC = passUV + vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"float centerDepth = texture(texDepth, uvC).r;\r\n"
		"if (centerDepth >= 0.9999){ outColor = vec4(0.5, 0.5, 1.0, 1.0); return; }\r\n" // sky: camera-facing
		"float texelU = 1.0 / uf_pc.width;\r\n"
		"float texelV = 1.0 / uf_pc.height;\r\n"
		"float dL = texture(texDepth, uvC - vec2(2.0 * texelU, 0.0)).r;\r\n"
		"float dR = texture(texDepth, uvC + vec2(2.0 * texelU, 0.0)).r;\r\n"
		"float dU = texture(texDepth, uvC - vec2(0.0, 2.0 * texelV)).r;\r\n"
		"float dD = texture(texDepth, uvC + vec2(0.0, 2.0 * texelV)).r;\r\n"
		"float dduP = dR - centerDepth;\r\n"
		"float dduM = centerDepth - dL;\r\n"
		"float ddu = (abs(dduM) < abs(dduP) ? dduM : dduP) * uf_pc.width * 0.5;\r\n"
		"float ddvP = dD - centerDepth;\r\n"
		"float ddvM = centerDepth - dU;\r\n"
		"float ddv = (abs(ddvM) < abs(ddvP) ? ddvM : ddvP) * uf_pc.height * 0.5;\r\n"
		// approximate view-space depth linearization (assumed FOV, see
		// LatteSSAO::Config::fovYDegrees/depthLinearK): linearDepth is
		// proportional to real distance from the camera, unlike raw Vulkan
		// depth which compresses nonlinearly. distK is its local derivative
		// w.r.t. raw depth (how much real distance one raw-depth unit
		// represents here) - dividing a raw depth delta by distK converts it
		// into the same world-proportional units every bias/falloff/thickness
		// constant below is tuned in, exactly like the old ad-hoc squared-
		// clamped heuristic did, but grounded in an actual depth curve instead
		// of a guess referenced against raw depth alone (which carries no
		// information about camera FOV/framing - a cutscene close-up shot via
		// a tight FOV crop has perfectly normal raw depth values despite
		// needing much wider on-screen kernels, which silently defeated the
		// old formula)
		"float linZ = 1.0 / max(1.0 - centerDepth * uf_pc.depthLinearK, 1e-4);\r\n"
		"float distK = 1.0 / max(uf_pc.depthLinearK * linZ * linZ, 1e-4);\r\n"
		"float S = uf_pc.ssrDepthScale;\r\n"
		"vec3 n = normalize(vec3(-ddu * S / distK, -ddv * S / distK, 1.0));\r\n"
		"outColor = vec4(n * 0.5 + 0.5, 1.0);\r\n"
		"}\r\n";

	// pass S: Launchpad-style "smoothed normals". The raw normals above are
	// constant per polygon facet (depth contains the real low-poly triangles;
	// the smooth vertex normals and normal maps the game shades with never reach
	// the depth buffer), which made SSR reflections jump direction at every
	// facet edge. Wide averaging across neighbors recreates the lost smoothing
	// groups: depth weight rejects silhouettes (other surfaces), normal weight
	// keeps genuinely hard corners (wall/floor, ~90 deg) from rounding off while
	// letting shallow facet creases (10-30 deg) fuse into a continuous curve
	const char* smoothSrc =
		"#version 450\r\n"
		"layout(location = 0) in vec2 passUV;\r\n"
		"layout(location = 0) out vec4 outColor;\r\n"
		"layout(binding = 1) uniform sampler2D texDepth;\r\n"
		"layout(binding = 2) uniform sampler2D texNormal;\r\n"
		"layout(push_constant) uniform pushConstants {\r\n"
		"float radiusU;\r\n"
		"float radiusV;\r\n"
		"float intensity;\r\n"
		"float bias;\r\n"
		"float debugShowAOOnly;\r\n"
		"float jitterUVx;\r\n"
		"float jitterUVy;\r\n"
		"float width;\r\n"
		"float height;\r\n"
		"float falloffRange;\r\n"
		"float heightScale;\r\n"
		"float ssrStrength;\r\n"
		"float ssrDepthScale;\r\n"
		"float ssrThickness;\r\n"
		"float ssrDebug;\r\n"
		"float aoLumFadeStart;\r\n"
		"float aoFloor;\r\n"
		"float ssrRoughness;\r\n"
		"float csStrength;\r\n"
		"float csDirX;\r\n"
		"float csDirY;\r\n"
		"float csLength;\r\n"
		"float giStrength;\r\n"
		"float depthLinearK;\r\n"
		"float fovScale;\r\n"
		"float smoothRadiusView;\r\n"
		"}uf_pc;\r\n"
		"void main(){\r\n"
		"vec2 uvC = passUV + vec2(uf_pc.jitterUVx, uf_pc.jitterUVy);\r\n"
		"float centerDepth = texture(texDepth, uvC).r;\r\n"
		"vec3 nC = normalize(texture(texNormal, passUV).xyz * 2.0 - 1.0);\r\n"
		"if (centerDepth >= 0.9999){ outColor = vec4(nC * 0.5 + 0.5, 1.0); return; }\r\n"
		// approximate view-space depth linearization (assumed FOV, see
		// LatteSSAO::Config::fovYDegrees/depthLinearK): linearDepth is
		// proportional to real distance from the camera, unlike raw Vulkan
		// depth which compresses nonlinearly. distK is its local derivative
		// w.r.t. raw depth (how much real distance one raw-depth unit
		// represents here) - dividing a raw depth delta by distK converts it
		// into the same world-proportional units every bias/falloff/thickness
		// constant below is tuned in, exactly like the old ad-hoc squared-
		// clamped heuristic did, but grounded in an actual depth curve instead
		// of a guess referenced against raw depth alone (which carries no
		// information about camera FOV/framing - a cutscene close-up shot via
		// a tight FOV crop has perfectly normal raw depth values despite
		// needing much wider on-screen kernels, which silently defeated the
		// old formula)
		"float linZ = 1.0 / max(1.0 - centerDepth * uf_pc.depthLinearK, 1e-4);\r\n"
		"float distK = 1.0 / max(uf_pc.depthLinearK * linZ * linZ, 1e-4);\r\n"
		"float texelU = 1.0 / uf_pc.width;\r\n"
		"float texelV = 1.0 / uf_pc.height;\r\n"
		// membership test = tangent plane fit, NOT normal similarity. A first
		// version weighted taps by dot(nC,nT), but the pseudo view space
		// amplifies angles (ssrDepthScale/distK): two facets 15 deg apart in the
		// world read as nearly perpendicular here, so the weight rejected exactly
		// the facet neighbors the pass exists to fuse and the smoothing was a
		// no-op (reflections stayed polygonal, as reported). A tap belongs to the
		// same smooth surface iff its DEPTH fits the center's tangent plane -
		// facet creases sit near the plane, silhouettes and perpendicular
		// geometry drift off it quickly with distance. Recover the plane's slope
		// (raw depth per UV) from the encoded normal: n ~ (-ddu*S/K, -ddv*S/K, 1)
		"float S = uf_pc.ssrDepthScale;\r\n"
		"vec2 slopeC = -nC.xy / max(nC.z, 0.05) * (distK / S);\r\n"
		// directional/radial march (8 directions x 4 steps, growing radius per
		// step) instead of a flat NxN grid - a flat grid's reach is capped by
		// its tap budget (stride x half-width), while marching a few
		// directions outward with a quadratically growing step size reaches
		// much farther for the same 32-tap cost (dense near the center where
		// most real facet creases sit, sparse but far-reaching at the last
		// step). Pulled directly from how iMMERSE Launchpad's smoothed-normals
		// prepass (Pascal Gilcher / "Marty McFly", the shader this whole
		// technique is modeled on) gets its reach - verified against its
		// actual source (SmoothNormalsVS / smooth_normals_mkii in
		// MartysMods_LAUNCHPAD.fx). NOT ported wholesale: that shader fits a
		// weighted least-squares gradient plane and weights samples by
		// reconstructed view-space distance and normal angle, both of which
		// need real view-space positions from the game's projection matrix -
		// exactly the input this filter doesn't have (same reason every other
		// distance here is raw-depth based). Kept our own weighting as-is
		// (tangent-plane depth fit only, see below) rather than adding an
		// angle-based weight: that was tried before and over-rejected in this
		// shader's amplified pseudo-view space, which is why only the hard
		// opposite-facing cutoff remains
		// target reach in linearized view-space units (uf_pc.smoothRadiusView),
		// converted to texels using the reconstructed depth and assumed FOV:
		// d(viewX)/d(texel) = 2*aspect*fovScale*linZ/width, so texels-per-unit
		// is its reciprocal. This is what makes the kernel widen for a tight
		// FOV-cropped close-up (small linZ change, but the SAME viewX span now
		// needs far fewer texels since fovScale/aspect already account for the
		// crop) instead of a fixed texel count that has no idea the shot is
		// zoomed in at all
		"float aspect = uf_pc.width / uf_pc.height;\r\n"
		"float texelsPerViewUnit = uf_pc.width / max(2.0 * aspect * uf_pc.fovScale * linZ, 1e-4);\r\n"
		"float kMaxTexels = clamp(uf_pc.smoothRadiusView * texelsPerViewUnit, 6.0, 800.0);\r\n"
		"const int kDirs = 8;\r\n"
		// 4->6->8 steps: each time smoothRadiusView grows, the same tap count
		// spreads thinner and leaves gaps a facet edge can poke through even
		// though the kernel technically reaches far enough - keep bumping steps
		// alongside radius to hold coverage density. 64 taps total (was 32)
		"const int kSteps = 8;\r\n"
		// center is pre-seeded at a small weight so the normalize below never
		// divides by a near-zero vector when every neighbor gets rejected
		"vec3 accum = nC * 0.25;\r\n"
		"for (int d = 0; d < kDirs; d++){\r\n"
		"float ang = float(d) * (6.28318530718 / float(kDirs));\r\n"
		"vec2 sdir = vec2(cos(ang), sin(ang));\r\n"
		"for (int s = 0; s < kSteps; s++){\r\n"
		"float fi = float(s + 1) / float(kSteps);\r\n"
		"float r = fi * fi * kMaxTexels;\r\n"
		"vec2 off = sdir * vec2(texelU, texelV) * r;\r\n"
		"vec3 nT = normalize(texture(texNormal, passUV + off).xyz * 2.0 - 1.0);\r\n"
		"float d2 = texture(texDepth, uvC + off).r;\r\n"
		"float dExp = centerDepth + slopeC.x * off.x + slopeC.y * off.y;\r\n"
		"float wP = 1.0 - clamp(abs(d2 - dExp) / (distK * uf_pc.falloffRange), 0.0, 1.0);\r\n"
		// only hard-reject genuinely opposite-facing taps (silhouette bleed)
		"if (dot(nC, nT) > 0.0) accum += nT * wP;\r\n"
		"}\r\n"
		"}\r\n"
		"outColor = vec4(normalize(accum) * 0.5 + 0.5, 1.0);\r\n"
		"}\r\n";

	std::string vsStr(vsSrc);
	m_vertexShader = new RendererShaderVk(RendererShader::ShaderType::kVertex, 0, 0, false, false, vsStr);
	m_vertexShader->PreponeCompilation(true);

	std::string fsStr(fsSrc);
	m_fragmentShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, fsStr);
	m_fragmentShader->PreponeCompilation(true);

	std::string blurStr(blurSrc);
	m_blurShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, blurStr);
	m_blurShader->PreponeCompilation(true);

	std::string normalStr(normalSrc);
	m_normalShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, normalStr);
	m_normalShader->PreponeCompilation(true);

	std::string smoothStr(smoothSrc);
	m_smoothShader = new RendererShaderVk(RendererShader::ShaderType::kFragment, 0, 0, false, false, smoothStr);
	m_smoothShader->PreponeCompilation(true);

	return m_vertexShader != nullptr && m_fragmentShader != nullptr && m_blurShader != nullptr
		&& m_normalShader != nullptr && m_smoothShader != nullptr;
}

bool VulkanSSAOFilter::CreateStaticObjects(VulkanRenderer* renderer)
{
	VkDevice device = renderer->GetLogicalDevice();

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	m_sampler = VKRObjectSampler::GetOrCreateSampler(&samplerInfo);
	if (!m_sampler)
		return false;
	m_sampler->incRef();

	VkDescriptorSetLayoutBinding bindings[kDescriptorBindings]{};
	for (uint32 i = 0; i < kDescriptorBindings; i++)
	{
		bindings[i].binding = i;
		bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		bindings[i].descriptorCount = 1;
		bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	VkDescriptorSetLayoutCreateInfo dslInfo{};
	dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	dslInfo.bindingCount = kDescriptorBindings;
	dslInfo.pBindings = bindings;
	if (vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS)
		return false;

	VkPushConstantRange pcRange{};
	pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	pcRange.offset = 0;
	pcRange.size = sizeof(PushConstants);
	VkPipelineLayoutCreateInfo plInfo{};
	plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plInfo.setLayoutCount = 1;
	plInfo.pSetLayouts = &m_descriptorSetLayout;
	plInfo.pushConstantRangeCount = 1;
	plInfo.pPushConstantRanges = &pcRange;
	if (vkCreatePipelineLayout(device, &plInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
		return false;

	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSize.descriptorCount = kDescriptorRingSize * kDescriptorBindings;
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = kDescriptorRingSize;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS)
		return false;

	VkDescriptorSetLayout layouts[kDescriptorRingSize];
	for (uint32 i = 0; i < kDescriptorRingSize; i++)
		layouts[i] = m_descriptorSetLayout;
	VkDescriptorSetAllocateInfo dsAlloc{};
	dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAlloc.descriptorPool = m_descriptorPool;
	dsAlloc.descriptorSetCount = kDescriptorRingSize;
	dsAlloc.pSetLayouts = layouts;
	if (vkAllocateDescriptorSets(device, &dsAlloc, m_descriptorRing) != VK_SUCCESS)
		return false;

	return true;
}

bool VulkanSSAOFilter::CreateSizedObjects(VulkanRenderer* renderer)
{
	VkDevice device = renderer->GetLogicalDevice();

	// every offscreen target follows the same pattern (2D, single mip, color
	// attachment + sampled). The intermediates use fixed RGBA8 instead of the
	// scanout format: some scanout formats have a joke alpha channel (A2 in
	// 10-bit formats) and the AO term/encoded normals need the full 8 bits
	auto createTarget = [&](VkFormat format, VkImage& image, VkDeviceMemory& memory,
							VKRObjectTexture*& texObj, VKRObjectTextureView*& viewObj) -> bool
	{
		VkImageCreateInfo imgInfo{};
		imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imgInfo.imageType = VK_IMAGE_TYPE_2D;
		imgInfo.format = format;
		imgInfo.extent = { (uint32)m_width, (uint32)m_height, 1 };
		imgInfo.mipLevels = 1;
		imgInfo.arrayLayers = 1;
		imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		if (vkCreateImage(device, &imgInfo, nullptr, &image) != VK_SUCCESS)
			return false;

		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device, image, &memReq);
		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memReq.size;
		allocInfo.memoryTypeIndex = FindMemoryType(renderer->GetPhysicalDevice(), memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
			return false;
		vkBindImageMemory(device, image, memory, 0);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		VkImageView rawView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &rawView) != VK_SUCCESS)
			return false;

		texObj = new VKRObjectTexture();
		texObj->m_image = image;
		texObj->m_format = format;
		texObj->m_imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;
		viewObj = new VKRObjectTextureView(texObj, rawView);
		return true;
	};

	const VkFormat auxFormat = VK_FORMAT_R8G8B8A8_UNORM;
	if (!createTarget(m_format, m_image, m_memory, m_texObj, m_viewObj))
		return false;
	if (!createTarget(auxFormat, m_aoImage, m_aoMemory, m_aoTexObj, m_aoViewObj))
		return false;
	if (!createTarget(auxFormat, m_giImage, m_giMemory, m_giTexObj, m_giViewObj))
		return false;
	if (!createTarget(auxFormat, m_normalImage, m_normalMemory, m_normalTexObj, m_normalViewObj))
		return false;
	if (!createTarget(auxFormat, m_normalSmoothImage, m_normalSmoothMemory, m_normalSmoothTexObj, m_normalSmoothViewObj))
		return false;
	m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
	m_aoLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	m_giLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	m_normalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	m_normalSmoothLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	// final composite target (presented)
	VKRObjectRenderPass::AttachmentInfo_t attachmentInfo{};
	attachmentInfo.colorAttachment[0].viewObj = m_viewObj;
	attachmentInfo.colorAttachment[0].format = m_format;
	m_renderPass = new VKRObjectRenderPass(attachmentInfo, 1);
	std::array<VKRObjectTextureView*, 1> fbAttachments{ m_viewObj };
	m_framebuffer = new VKRObjectFramebuffer(m_renderPass, fbAttachments, Vector2i(m_width, m_height));

	// effects pass MRT: attachment 0 = color+AO, attachment 1 = SSGI radiance
	VKRObjectRenderPass::AttachmentInfo_t aoAttachmentInfo{};
	aoAttachmentInfo.colorAttachment[0].viewObj = m_aoViewObj;
	aoAttachmentInfo.colorAttachment[0].format = auxFormat;
	aoAttachmentInfo.colorAttachment[1].viewObj = m_giViewObj;
	aoAttachmentInfo.colorAttachment[1].format = auxFormat;
	m_aoRenderPass = new VKRObjectRenderPass(aoAttachmentInfo, 2);
	std::array<VKRObjectTextureView*, 2> aoFbAttachments{ m_aoViewObj, m_giViewObj };
	m_aoFramebuffer = new VKRObjectFramebuffer(m_aoRenderPass, aoFbAttachments, Vector2i(m_width, m_height));

	// normals-from-depth target
	VKRObjectRenderPass::AttachmentInfo_t normalAttachmentInfo{};
	normalAttachmentInfo.colorAttachment[0].viewObj = m_normalViewObj;
	normalAttachmentInfo.colorAttachment[0].format = auxFormat;
	m_normalRenderPass = new VKRObjectRenderPass(normalAttachmentInfo, 1);
	std::array<VKRObjectTextureView*, 1> normalFbAttachments{ m_normalViewObj };
	m_normalFramebuffer = new VKRObjectFramebuffer(m_normalRenderPass, normalFbAttachments, Vector2i(m_width, m_height));

	// smoothed normals target
	VKRObjectRenderPass::AttachmentInfo_t smoothAttachmentInfo{};
	smoothAttachmentInfo.colorAttachment[0].viewObj = m_normalSmoothViewObj;
	smoothAttachmentInfo.colorAttachment[0].format = auxFormat;
	m_normalSmoothRenderPass = new VKRObjectRenderPass(smoothAttachmentInfo, 1);
	std::array<VKRObjectTextureView*, 1> smoothFbAttachments{ m_normalSmoothViewObj };
	m_normalSmoothFramebuffer = new VKRObjectFramebuffer(m_normalSmoothRenderPass, smoothFbAttachments, Vector2i(m_width, m_height));

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = m_vertexShader->GetShaderModule();
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = m_fragmentShader->GetShaderModule();
	stages[1].pName = "main";

	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)m_width;
	viewport.height = (float)m_height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	VkRect2D scissor{};
	scissor.extent = { (uint32)m_width, (uint32)m_height };
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.cullMode = VK_CULL_MODE_NONE;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	// two identical no-blend attachment states (the effects pass writes 2 MRT
	// attachments, all other passes use the first entry only)
	VkPipelineColorBlendAttachmentState blendAttachments[2]{};
	for (uint32 i = 0; i < 2; i++)
	{
		blendAttachments[i].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		blendAttachments[i].blendEnable = VK_FALSE;
	}
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = blendAttachments;

	VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = 2;
	dynamicState.pDynamicStates = dynamicStates;

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.layout = m_pipelineLayout;
	pipelineInfo.subpass = 0;

	auto createPipeline = [&](RendererShaderVk* fs, VkRenderPass rp, uint32 attachmentCount, VkPipeline& out) -> bool
	{
		stages[1].module = fs->GetShaderModule();
		colorBlending.attachmentCount = attachmentCount;
		pipelineInfo.renderPass = rp;
		return vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &out) == VK_SUCCESS;
	};

	// pass N: normals from depth; pass S: wide depth/normal-aware smoothing;
	// pass A (effects, MRT color+AO / GI); pass B (bilateral blur + composite)
	if (!createPipeline(m_normalShader, m_normalRenderPass->m_renderPass, 1, m_pipelineNormal))
		return false;
	if (!createPipeline(m_smoothShader, m_normalSmoothRenderPass->m_renderPass, 1, m_pipelineSmooth))
		return false;
	if (!createPipeline(m_fragmentShader, m_aoRenderPass->m_renderPass, 2, m_pipeline))
		return false;
	if (!createPipeline(m_blurShader, m_renderPass->m_renderPass, 1, m_pipelineBlur))
		return false;

	return true;
}

void VulkanSSAOFilter::ReleaseResources(VulkanRenderer* renderer)
{
	m_hasValidOutput = false;
	// GPU may still be executing a command buffer recorded against the old
	// image/pipeline (e.g. still in flight while a resize churns through
	// recreations); only free them once that command buffer has actually
	// finished, not after a guessed number of Apply() calls
	uint64 releaseCmdBufferId = renderer->GetCurrentCommandBufferId();

	auto releaseTarget = [&](VkImage& image, VkDeviceMemory& memory, VKRObjectTexture*& texObj,
							 VKRObjectTextureView*& viewObj, VkImageLayout& layout)
	{
		if (image != VK_NULL_HANDLE)
		{
			m_pendingDeletes.push_back({ image, memory, VK_NULL_HANDLE, releaseCmdBufferId });
			image = VK_NULL_HANDLE;
			memory = VK_NULL_HANDLE;
		}
		if (viewObj)
		{
			renderer->ReleaseDestructibleObject(viewObj);
			viewObj = nullptr;
		}
		if (texObj)
		{
			texObj->m_image = VK_NULL_HANDLE;
			renderer->ReleaseDestructibleObject(texObj);
			texObj = nullptr;
		}
		layout = VK_IMAGE_LAYOUT_UNDEFINED;
	};
	auto releaseFramebuffer = [&](VKRObjectFramebuffer*& fb, VKRObjectRenderPass*& rp)
	{
		if (fb)
		{
			renderer->ReleaseDestructibleObject(fb);
			fb = nullptr;
		}
		if (rp)
		{
			renderer->ReleaseDestructibleObject(rp);
			rp = nullptr;
		}
	};
	auto releasePipeline = [&](VkPipeline& pipeline)
	{
		if (pipeline != VK_NULL_HANDLE)
		{
			m_pendingDeletes.push_back({ VK_NULL_HANDLE, VK_NULL_HANDLE, pipeline, releaseCmdBufferId });
			pipeline = VK_NULL_HANDLE;
		}
	};

	releaseFramebuffer(m_framebuffer, m_renderPass);
	releaseFramebuffer(m_aoFramebuffer, m_aoRenderPass);
	releaseFramebuffer(m_normalFramebuffer, m_normalRenderPass);
	releaseFramebuffer(m_normalSmoothFramebuffer, m_normalSmoothRenderPass);
	releaseTarget(m_image, m_memory, m_texObj, m_viewObj, m_layout);
	releaseTarget(m_aoImage, m_aoMemory, m_aoTexObj, m_aoViewObj, m_aoLayout);
	releaseTarget(m_giImage, m_giMemory, m_giTexObj, m_giViewObj, m_giLayout);
	releaseTarget(m_normalImage, m_normalMemory, m_normalTexObj, m_normalViewObj, m_normalLayout);
	releaseTarget(m_normalSmoothImage, m_normalSmoothMemory, m_normalSmoothTexObj, m_normalSmoothViewObj, m_normalSmoothLayout);
	releasePipeline(m_pipeline);
	releasePipeline(m_pipelineBlur);
	releasePipeline(m_pipelineNormal);
	releasePipeline(m_pipelineSmooth);
}

void VulkanSSAOFilter::TickPendingDeletes(VulkanRenderer* renderer)
{
	VkDevice device = renderer->GetLogicalDevice();
	for (auto it = m_pendingDeletes.begin(); it != m_pendingDeletes.end();)
	{
		if (!renderer->HasCommandBufferFinished(it->safeCmdBufferId))
		{
			++it;
			continue;
		}
		if (it->pipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(device, it->pipeline, nullptr);
		if (it->image != VK_NULL_HANDLE)
			vkDestroyImage(device, it->image, nullptr);
		if (it->memory != VK_NULL_HANDLE)
			vkFreeMemory(device, it->memory, nullptr);
		it = m_pendingDeletes.erase(it);
	}
}

bool VulkanSSAOFilter::RecreateIfNeeded(VulkanRenderer* renderer, sint32 width, sint32 height, VkFormat format)
{
	if (width == m_width && height == m_height && format == m_format && m_pipeline != VK_NULL_HANDLE)
		return true;

	if (m_vertexShader == nullptr)
	{
		if (!CreateShaders() || !CreateStaticObjects(renderer))
		{
			cemuLog_log(LogType::Force, "VulkanSSAOFilter: failed to create static objects");
			return false;
		}
	}

	ReleaseResources(renderer);
	m_width = width;
	m_height = height;
	m_format = format;
	if (!CreateSizedObjects(renderer))
	{
		cemuLog_log(LogType::Force, "VulkanSSAOFilter: failed to create sized objects ({}x{})", width, height);
		ReleaseResources(renderer);
		m_width = 0;
		m_height = 0;
		m_format = VK_FORMAT_UNDEFINED;
		return false;
	}
	return true;
}

void VulkanSSAOFilter::Apply(VulkanRenderer* renderer, LatteTextureViewVk* scanoutView)
{
	auto& config = LatteSSAO::GetConfig();
	if (!LatteSSAO::AnyEffectEnabled())
		return;

	// re-presents without new game draws carry no new frame (e.g. 30fps cutscenes
	// shown at 60Hz): the depth buffer is unchanged, but TAA's jitter sequence
	// still advances every present, so recomputing here would unjitter the same
	// static depth by a different amount each redundant present and flicker.
	// Output stays valid, just don't touch it - mirrors VulkanTAAFilter::Apply
	if (LatteGPUState.drawCallCounter == m_lastDrawCallCounter)
		return;
	// same guard against overlay-only presents (subtitles/letterbox between the
	// two presents of a 30fps cutscene frame): the scene and its depth are
	// unchanged, recomputing with the advanced jitter index just flickers the AO.
	// And like the TAA filter, a long scene-less run means a pure-2D mode
	// (menus/fades) where holding the last output would freeze the screen -
	// step aside and present raw until scene geometry returns
	const uint32 drawsSinceApply = LatteGPUState.drawCallCounter - m_lastDrawCallCounter;
	const uint32 sceneDrawCounter = LatteTAA::GetSceneDrawCounter();
	if (sceneDrawCounter == m_lastSceneDrawCounter && drawsSinceApply < 200)
	{
		m_lastDrawCallCounter = LatteGPUState.drawCallCounter;
		m_consecutiveSceneless++;
		if (m_consecutiveSceneless > 30)
			m_hasValidOutput = false;
		return;
	}
	m_consecutiveSceneless = 0;

	// every bailout below must invalidate any previously produced output: the
	// scanout's color resolution (checked by GetPresentDescriptorSet) can stay
	// identical across these frames even though SSAO itself isn't updating, which
	// would otherwise present one stale AO'd frame forever (screen looks frozen
	// while the game keeps simulating/drawing normally underneath)
	LatteTextureVk* scanoutTex = (LatteTextureVk*)scanoutView->baseTexture;
	const uint32 scanMip = (uint32)scanoutView->firstMip;
	const uint32 scanSlice = (uint32)scanoutView->firstSlice;

	sint32 effWidth = 0, effHeight = 0;
	scanoutTex->GetEffectiveSize(effWidth, effHeight, scanoutView->firstMip);
	if (effWidth < 2 || effHeight < 2)
	{
		m_hasValidOutput = false;
		return;
	}
	// the scene-draw classifier (LatteTAA::GetViewportJitter) refuses to count
	// any draw until it knows the scene output size, and only the AA filter's
	// Apply used to report it. With AA disabled and only these effects on, the
	// scene counter therefore never advanced and the guard above only let a
	// frame through when it happened to carry >=200 draws - cutscene shots
	// hover around that number, which is exactly the reported on/off flapping
	// in cinematics and camera cuts. Report the size from here too.
	LatteTAA::SetOutputSize(effWidth, effHeight);

	// depth buffer the game had bound just before the scanout swap - same timing
	// TAA uses for color. Confirmed via logging (2026-07-08): the game commonly
	// rebinds render targets after the main scene pass for HUD/UI, letterbox bars,
	// subtitles etc, none of which carry a depth buffer of their own, so at this
	// exact instant GetDepthAttachment() is very often null (constantly during
	// gameplay's HUD, intermittently during cutscenes too) even though the real
	// scene depth rendered fine into the same persistent GX2 buffer this same
	// frame. Fall back to the last depth attachment that DID match the scene
	// resolution instead of producing nothing on those frames.
	LatteTextureView* depthView = LatteMRT::GetDepthAttachment();
	LatteTextureVk* depthTexVk = nullptr;
	sint32 depthWidth = 0, depthHeight = 0;
	bool haveDepth = false;
	const char* depthSource = "none";
	if (depthView && depthView->baseTexture)
	{
		depthTexVk = (LatteTextureVk*)depthView->baseTexture;
		depthTexVk->GetEffectiveSize(depthWidth, depthHeight, depthView->firstMip);
		if (depthWidth == effWidth && depthHeight == effHeight)
		{
			haveDepth = true;
			depthSource = "current";
			m_cachedDepthView = depthView; // remember for the next frame that lacks one
		}
	}
	// prefer the depth view captured at bind time DURING this frame (survives
	// cutscene camera cuts that swap in brand-new depth buffers), then fall back
	// to the older cross-frame cache
	if (!haveDepth && m_frameDepthView && m_frameDepthView->baseTexture)
	{
		LatteTextureVk* frameTexVk = (LatteTextureVk*)m_frameDepthView->baseTexture;
		sint32 frameWidth = 0, frameHeight = 0;
		frameTexVk->GetEffectiveSize(frameWidth, frameHeight, m_frameDepthView->firstMip);
		if (frameWidth == effWidth && frameHeight == effHeight)
		{
			depthView = m_frameDepthView;
			depthTexVk = frameTexVk;
			depthWidth = frameWidth;
			depthHeight = frameHeight;
			haveDepth = true;
			depthSource = "frame-bind";
			m_cachedDepthView = depthView;
		}
	}
	if (!haveDepth && m_cachedDepthView && m_cachedDepthView->baseTexture)
	{
		LatteTextureVk* cachedTexVk = (LatteTextureVk*)m_cachedDepthView->baseTexture;
		sint32 cachedWidth = 0, cachedHeight = 0;
		cachedTexVk->GetEffectiveSize(cachedWidth, cachedHeight, m_cachedDepthView->firstMip);
		if (cachedWidth == effWidth && cachedHeight == effHeight)
		{
			depthView = m_cachedDepthView;
			depthTexVk = cachedTexVk;
			depthWidth = cachedWidth;
			depthHeight = cachedHeight;
			haveDepth = true;
			depthSource = "cached";
		}
	}
	// TEMP diagnostic (2026-07-08): the depth-cache fallback above didn't fully
	// fix on/off flicker reported in certain specific cutscenes - log only on
	// state transitions (not on a timer) to catch the exact moment, likely a
	// resolution change (letterbox/camera-cut) that invalidates the cache too.
	// Remove once the cause is confirmed.
	static sint32 s_diagLastEffW = -1, s_diagLastEffH = -1;
	static bool s_diagLastHaveDepth = true;
	static const char* s_diagLastSource = "";
	if (effWidth != s_diagLastEffW || effHeight != s_diagLastEffH || haveDepth != s_diagLastHaveDepth || depthSource != s_diagLastSource)
	{
		cemuLog_log(LogType::Force, "SSAO diag: transition - color {}x{} (was {}x{}), haveDepth={} (was {}), source={}, cachedDepthView={}",
			effWidth, effHeight, s_diagLastEffW, s_diagLastEffH, haveDepth, s_diagLastHaveDepth, depthSource, (void*)m_cachedDepthView);
		s_diagLastEffW = effWidth;
		s_diagLastEffH = effHeight;
		s_diagLastHaveDepth = haveDepth;
		s_diagLastSource = depthSource;
	}

	if (!haveDepth)
	{
		m_hasValidOutput = false;
		return; // no usable depth this frame, and no cached one of the right size either
	}
	LatteTextureViewVk* depthViewVk = (LatteTextureViewVk*)depthView;

	if (!RecreateIfNeeded(renderer, effWidth, effHeight, scanoutTex->GetFormat()))
	{
		m_hasValidOutput = false;
		return;
	}

	VkDevice device = renderer->GetLogicalDevice();
	TickPendingDeletes(renderer);

	// caller has already closed any pending render pass (TAA_Apply / SSAO_Apply
	// both call draw_endRenderPass() before recording)
	VkCommandBuffer cmd = renderer->SSAO_GetCommandBuffer();

	m_renderPass->flagForCurrentCommandBuffer();
	m_framebuffer->flagForCurrentCommandBuffer();
	m_aoRenderPass->flagForCurrentCommandBuffer();
	m_aoFramebuffer->flagForCurrentCommandBuffer();
	m_aoViewObj->flagForCurrentCommandBuffer();
	m_giViewObj->flagForCurrentCommandBuffer();
	m_normalRenderPass->flagForCurrentCommandBuffer();
	m_normalFramebuffer->flagForCurrentCommandBuffer();
	m_normalViewObj->flagForCurrentCommandBuffer();
	m_normalSmoothRenderPass->flagForCurrentCommandBuffer();
	m_normalSmoothFramebuffer->flagForCurrentCommandBuffer();
	m_normalSmoothViewObj->flagForCurrentCommandBuffer();
	m_sampler->flagForCurrentCommandBuffer();

	// color source: DLAA's output if valid (real NGX temporal AA, replaces TAA's own
	// resolve when active), else TAA's resolved output (already shader-readable, left
	// that way by Apply() for the backbuffer blit), else the raw scanout
	VkImageView colorViewRaw = VK_NULL_HANDLE;
#if BOOST_OS_WINDOWS
	colorViewRaw = VulkanDLSSFilter::GetInstance().GetResolvedImageViewIfValid(effWidth, effHeight);
#endif
	if (colorViewRaw == VK_NULL_HANDLE)
		colorViewRaw = VulkanTAAFilter::GetInstance().GetResolvedImageViewIfValid(effWidth, effHeight);
	if (colorViewRaw == VK_NULL_HANDLE)
	{
		VkImage scanoutImage = scanoutTex->GetImageObj()->m_image;
		VkImageSubresource scanoutSubres{ VK_IMAGE_ASPECT_COLOR_BIT, scanMip, scanSlice };
		VkImageLayout scanoutLayout = scanoutTex->GetImageLayout(scanoutSubres);
		_ssaoBarrierImage(cmd, scanoutImage, VK_IMAGE_ASPECT_COLOR_BIT, scanoutLayout, VK_IMAGE_LAYOUT_GENERAL,
						  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
						  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
						  scanMip, scanSlice);
		VkImageSubresourceRange scanoutRange{ VK_IMAGE_ASPECT_COLOR_BIT, scanMip, 1, scanSlice, 1 };
		scanoutTex->SetImageLayout(scanoutRange, VK_IMAGE_LAYOUT_GENERAL);

		VKRObjectTextureView* scanoutViewObj = scanoutView->GetViewRGBA();
		scanoutViewObj->flagForCurrentCommandBuffer();
		colorViewRaw = scanoutViewObj->m_textureImageView;
	}

	// depth source: never touched by TAA, always needs its own read barrier.
	// Cemu's tracker treats depth textures as GENERAL like everything else, and
	// GetViewRGBA() already restricts sampling to the depth aspect only (see
	// LatteTextureViewVk.cpp) - keep the barrier depth-only to match, safe here
	// because old==new==GENERAL is a sync barrier, not a real layout transition
	const uint32 depthMip = (uint32)depthView->firstMip;
	const uint32 depthSlice = (uint32)depthView->firstSlice;
	VkImage depthImage = depthTexVk->GetImageObj()->m_image;
	VkImageSubresource depthSubres{ VK_IMAGE_ASPECT_DEPTH_BIT, depthMip, depthSlice };
	VkImageLayout depthLayout = depthTexVk->GetImageLayout(depthSubres);
	_ssaoBarrierImage(cmd, depthImage, VK_IMAGE_ASPECT_DEPTH_BIT, depthLayout, VK_IMAGE_LAYOUT_GENERAL,
					  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					  depthMip, depthSlice);
	VkImageSubresourceRange depthRange{ VK_IMAGE_ASPECT_DEPTH_BIT, depthMip, 1, depthSlice, 1 };
	depthTexVk->SetImageLayout(depthRange, VK_IMAGE_LAYOUT_GENERAL);

	VKRObjectTextureView* depthViewObj = depthViewVk->GetViewRGBA();
	depthViewObj->flagForCurrentCommandBuffer();
	VkImageView depthViewRaw = depthViewObj->m_textureImageView;

	// one descriptor set per pass on the shared 4-binding layout; unused slots
	// are filled with the depth view so every set matches the layout
	auto nextDescriptorSet = [&]() {
		VkDescriptorSet s = m_descriptorRing[m_descriptorRingIndex];
		m_descriptorRingIndex = (m_descriptorRingIndex + 1) % kDescriptorRingSize;
		return s;
	};
	VkDescriptorSet setNormal = nextDescriptorSet();
	VkDescriptorSet setSmooth = nextDescriptorSet();
	VkDescriptorSet setEffects = nextDescriptorSet();
	VkDescriptorSet setBlur = nextDescriptorSet();

	const VkImageView normalView = m_normalViewObj->m_textureImageView;
	const VkImageView smoothView = m_normalSmoothViewObj->m_textureImageView;
	const VkImageView giView = m_giViewObj->m_textureImageView;
	const VkImageView aoView = m_aoViewObj->m_textureImageView;
	const VkImageView setViews[4][kDescriptorBindings] = {
		{ depthViewRaw, depthViewRaw, depthViewRaw, depthViewRaw },   // pass N: normals from depth
		{ depthViewRaw, depthViewRaw, normalView, depthViewRaw },     // pass S: smooth the normals
		{ colorViewRaw, depthViewRaw, smoothView, depthViewRaw },     // pass A: effects (MRT)
		{ aoView, depthViewRaw, giView, depthViewRaw },               // pass B: blur + composite
	};
	const VkDescriptorSet sets[4] = { setNormal, setSmooth, setEffects, setBlur };
	VkDescriptorImageInfo imageInfos[4 * kDescriptorBindings]{};
	VkWriteDescriptorSet writes[4 * kDescriptorBindings]{};
	for (uint32 s = 0; s < 4; s++)
	{
		for (uint32 b = 0; b < kDescriptorBindings; b++)
		{
			const uint32 i = s * kDescriptorBindings + b;
			imageInfos[i].sampler = m_sampler->GetSampler();
			imageInfos[i].imageView = setViews[s][b];
			imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[i].dstSet = sets[s];
			writes[i].dstBinding = b;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			writes[i].pImageInfo = &imageInfos[i];
		}
	}
	vkUpdateDescriptorSets(device, 4 * kDescriptorBindings, writes, 0, nullptr);

	VkRenderPassBeginInfo rpBegin{};
	rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpBegin.renderArea.extent = { (uint32)m_width, (uint32)m_height };

	// dynamic viewport/scissor and the push constants are command buffer state
	// shared by every pass (all pipelines use the same layout) - set them once
	VkViewport passViewport{};
	passViewport.x = 0.0f;
	passViewport.y = 0.0f;
	passViewport.width = (float)m_width;
	passViewport.height = (float)m_height;
	passViewport.minDepth = 0.0f;
	passViewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &passViewport);
	VkRect2D passScissor{};
	passScissor.extent = { (uint32)m_width, (uint32)m_height };
	vkCmdSetScissor(cmd, 0, 1, &passScissor);

	PushConstants pc;
	// aspect-correct the UV radius so the sample disk is round in screen space
	pc.radiusU = config.radius * ((float)m_height / (float)m_width);
	pc.radiusV = config.radius;
	// each effect is gated in the shader by its own weight being zero, so the
	// one shared pass honors the two independent menu toggles
	pc.intensity = config.enabled ? config.intensity : 0.0f;
	pc.bias = config.bias;
	pc.debugShowAOOnly = config.debugShowAOOnly ? 1.0f : 0.0f;
	// same unjitter LatteTAA's own resolve applies to color (0,0 when TAA/jitter is off)
	float jitterPxX, jitterPxY;
	LatteTAA::GetCurrentFrameJitter(jitterPxX, jitterPxY);
	pc.jitterUVx = jitterPxX / (float)m_width;
	pc.jitterUVy = jitterPxY / (float)m_height;
	pc.width = (float)m_width;
	pc.height = (float)m_height;
	pc.falloffRange = config.falloffRange;
	pc.heightScale = config.heightScale;
	pc.ssrStrength = config.ssrEnabled ? config.ssrStrength : 0.0f;
	pc.ssrDepthScale = config.ssrDepthScale;
	pc.ssrThickness = config.ssrThickness;
	pc.ssrDebug = (config.ssrEnabled && config.ssrDebugView) ? 1.0f : 0.0f;
	pc.aoLumFadeStart = config.aoLumFadeStart;
	pc.aoFloor = config.aoFloor;
	pc.ssrRoughness = config.ssrRoughness;
	pc.csStrength = config.contactShadowsEnabled ? config.csStrength : 0.0f;
	pc.csDirX = config.csDirX;
	pc.csDirY = config.csDirY;
	pc.csLength = config.csLength;
	pc.giStrength = config.giEnabled ? config.giStrength : 0.0f;
	pc.depthLinearK = config.depthLinearK;
	pc.fovScale = std::tan(config.fovYDegrees * 0.5f * 0.017453292f); // degrees -> radians
	pc.smoothRadiusView = config.smoothRadiusView;
	vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

	auto toAttachment = [&](VkImage image, VkImageLayout& layout) {
		_ssaoBarrierImage(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, layout, VK_IMAGE_LAYOUT_GENERAL,
						  0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
						  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
		layout = VK_IMAGE_LAYOUT_GENERAL;
	};
	auto toShaderRead = [&](VkImage image) {
		_ssaoBarrierImage(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
						  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
						  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	};
	auto recordPass = [&](VKRObjectRenderPass* rp, VKRObjectFramebuffer* fb, VkPipeline pipeline, VkDescriptorSet set) {
		rpBegin.renderPass = rp->m_renderPass;
		rpBegin.framebuffer = fb->m_frameBuffer;
		vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &set, 0, nullptr);
		vkCmdDraw(cmd, 3, 1, 0, 0);
		vkCmdEndRenderPass(cmd);
	};

	// pass N + S: reconstruct normals from depth, then smooth them wide
	// (Launchpad-style). Every effect consumes the smoothed tangent plane now
	// (AO/GI/contact shadows would otherwise quantize into per-facet patches),
	// so both passes always run
	toAttachment(m_normalImage, m_normalLayout);
	recordPass(m_normalRenderPass, m_normalFramebuffer, m_pipelineNormal, setNormal);
	toShaderRead(m_normalImage);

	toAttachment(m_normalSmoothImage, m_normalSmoothLayout);
	recordPass(m_normalSmoothRenderPass, m_normalSmoothFramebuffer, m_pipelineSmooth, setSmooth);
	toShaderRead(m_normalSmoothImage);

	// pass A: HBAO + SSR + contact shadows + SSGI, MRT into color+AO / GI
	toAttachment(m_aoImage, m_aoLayout);
	toAttachment(m_giImage, m_giLayout);
	recordPass(m_aoRenderPass, m_aoFramebuffer, m_pipeline, setEffects);
	toShaderRead(m_aoImage);
	toShaderRead(m_giImage);

	// pass B: bilateral blur of AO+GI, composite into the presented image
	toAttachment(m_image, m_layout);
	recordPass(m_renderPass, m_framebuffer, m_pipelineBlur, setBlur);
	toShaderRead(m_image);

	m_hasValidOutput = true;
	m_lastDrawCallCounter = LatteGPUState.drawCallCounter;
	m_lastSceneDrawCounter = sceneDrawCounter;
}

VkDescriptorSet VulkanSSAOFilter::GetPresentDescriptorSet(VulkanRenderer* renderer, VkDescriptorSetLayout blitLayout,
														   sint32 expectedWidth, sint32 expectedHeight)
{
	if (!m_hasValidOutput || expectedWidth != m_width || expectedHeight != m_height)
		return VK_NULL_HANDLE;
	if (!m_sampler || !m_viewObj)
		return VK_NULL_HANDLE;
	VkDevice device = renderer->GetLogicalDevice();

	if (m_presentDescriptorPool == VK_NULL_HANDLE)
	{
		VkDescriptorPoolSize poolSize{};
		poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSize.descriptorCount = kPresentRingSize;
		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.maxSets = kPresentRingSize;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_presentDescriptorPool) != VK_SUCCESS)
			return VK_NULL_HANDLE;
		VkDescriptorSetLayout layouts[kPresentRingSize];
		for (uint32 i = 0; i < kPresentRingSize; i++)
			layouts[i] = blitLayout;
		VkDescriptorSetAllocateInfo dsAlloc{};
		dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		dsAlloc.descriptorPool = m_presentDescriptorPool;
		dsAlloc.descriptorSetCount = kPresentRingSize;
		dsAlloc.pSetLayouts = layouts;
		if (vkAllocateDescriptorSets(device, &dsAlloc, m_presentRing) != VK_SUCCESS)
		{
			vkDestroyDescriptorPool(device, m_presentDescriptorPool, nullptr);
			m_presentDescriptorPool = VK_NULL_HANDLE;
			return VK_NULL_HANDLE;
		}
	}

	m_viewObj->flagForCurrentCommandBuffer();
	m_sampler->flagForCurrentCommandBuffer();

	VkDescriptorSet descSet = m_presentRing[m_presentRingIndex];
	m_presentRingIndex = (m_presentRingIndex + 1) % kPresentRingSize;

	VkDescriptorImageInfo imageInfo{};
	imageInfo.sampler = m_sampler->GetSampler();
	imageInfo.imageView = m_viewObj->m_textureImageView;
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = descSet;
	write.dstBinding = 0;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	write.pImageInfo = &imageInfo;
	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
	return descSet;
}

void VulkanSSAOFilter::NotifyTextureDeletion(LatteTexture* texture)
{
	if (m_cachedDepthView && m_cachedDepthView->baseTexture == texture)
		m_cachedDepthView = nullptr;
	if (m_frameDepthView && m_frameDepthView->baseTexture == texture)
		m_frameDepthView = nullptr;
}

void VulkanSSAOFilter::NotifyDepthBind(LatteTextureView* view)
{
	// m_width/m_height hold the scene output size from the last completed
	// Apply(); before the first one there is nothing to match against yet
	if (!view || !view->baseTexture || m_width <= 0)
		return;
	sint32 w = 0, h = 0;
	view->baseTexture->GetEffectiveSize(w, h, view->firstMip);
	if (w == m_width && h == m_height)
		m_frameDepthView = view;
}

void VulkanSSAOFilter::Shutdown(VulkanRenderer* renderer)
{
	m_cachedDepthView = nullptr;
	m_frameDepthView = nullptr;
	ReleaseResources(renderer);
	VkDevice device = renderer->GetLogicalDevice();
	for (auto& pd : m_pendingDeletes)
	{
		if (pd.pipeline != VK_NULL_HANDLE)
			vkDestroyPipeline(device, pd.pipeline, nullptr);
		if (pd.image != VK_NULL_HANDLE)
			vkDestroyImage(device, pd.image, nullptr);
		if (pd.memory != VK_NULL_HANDLE)
			vkFreeMemory(device, pd.memory, nullptr);
	}
	m_pendingDeletes.clear();
	if (m_descriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
		m_descriptorPool = VK_NULL_HANDLE;
	}
	if (m_presentDescriptorPool != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorPool(device, m_presentDescriptorPool, nullptr);
		m_presentDescriptorPool = VK_NULL_HANDLE;
	}
	if (m_pipelineLayout != VK_NULL_HANDLE)
	{
		vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
		m_pipelineLayout = VK_NULL_HANDLE;
	}
	if (m_descriptorSetLayout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
		m_descriptorSetLayout = VK_NULL_HANDLE;
	}
	if (m_sampler)
	{
		m_sampler->decRef();
		m_sampler = nullptr;
	}
}

// member of VulkanRenderer so the filter can be driven from common Latte code
void VulkanRenderer::SSAO_Apply(LatteTextureView* textureView)
{
	if (!textureView || !textureView->baseTexture)
		return;
	LatteTextureVk* texVk = (LatteTextureVk*)textureView->baseTexture;
	if (texVk->isDepth)
		return;
	draw_endRenderPass();
	VulkanSSAOFilter::GetInstance().Apply(this, (LatteTextureViewVk*)textureView);
	m_state.currentPipeline = VK_NULL_HANDLE;
	vkCmdSetViewport(m_state.currentCommandBuffer, 0, 1, &m_state.currentViewport);
	vkCmdSetScissor(m_state.currentCommandBuffer, 0, 1, &m_state.currentScissorRect);
}

VkCommandBuffer VulkanRenderer::SSAO_GetCommandBuffer()
{
	return getCurrentCommandBuffer();
}
