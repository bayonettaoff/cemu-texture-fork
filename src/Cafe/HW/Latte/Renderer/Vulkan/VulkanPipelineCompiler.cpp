#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanRenderer.h"
#include "Cafe/HW/Latte/Core/FetchShader.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineCompiler.h"
#include "Cafe/HW/Latte/Renderer/Vulkan/VulkanPipelineStableCache.h"
#include "Cafe/HW/Latte/Core/LatteShader.h"
#include "Cafe/HW/Latte/Core/LattePerformanceMonitor.h"
#include "Cafe/HW/Latte/Core/LatteGeometryAmp.h"
#include "Cafe/OS/libs/gx2/GX2.h"
#include "config/ActiveSettings.h"
#include "util/helpers/Serializer.h"
#include "Cafe/HW/Latte/Common/RegisterSerializer.h"

std::mutex s_nvidiaWorkaround;

/* rects emulation */

void rectsEmulationGS_outputSingleVertex(std::string& gsSrc, LatteDecompilerShader* vertexShader, LatteShaderPSInputTable* psInputTable, sint32 vIdx, const LatteContextRegister& latteRegister)
{
	auto parameterMask = vertexShader->outputParameterMask;
	for (uint32 i = 0; i < 32; i++)
	{
		if ((parameterMask & (1 << i)) == 0)
			continue;
		sint32 vsSemanticId = psInputTable->getVertexShaderOutParamSemanticId(latteRegister.GetRawView(), i);
		if (vsSemanticId < 0)
			continue;
		// make sure PS has matching input
		if (!psInputTable->hasPSImportForSemanticId(vsSemanticId))
			continue;
		gsSrc.append(fmt::format("passParameterSem{}Out = passParameterSem{}In[{}];\r\n", vsSemanticId, vsSemanticId, vIdx));
	}
	gsSrc.append(fmt::format("gl_Position = gl_in[{}].gl_Position;\r\n", vIdx));
	gsSrc.append("EmitVertex();\r\n");
}

void rectsEmulationGS_outputGeneratedVertex(std::string& gsSrc, LatteDecompilerShader* vertexShader, LatteShaderPSInputTable* psInputTable, const char* variant, const LatteContextRegister& latteRegister)
{
	auto parameterMask = vertexShader->outputParameterMask;
	for (uint32 i = 0; i < 32; i++)
	{
		if ((parameterMask & (1 << i)) == 0)
			continue;
		sint32 vsSemanticId = psInputTable->getVertexShaderOutParamSemanticId(latteRegister.GetRawView(), i);
		if (vsSemanticId < 0)
			continue;
		// make sure PS has matching input
		if (!psInputTable->hasPSImportForSemanticId(vsSemanticId))
			continue;
		gsSrc.append(fmt::format("passParameterSem{}Out = gen4thVertex{}(passParameterSem{}In[0], passParameterSem{}In[1], passParameterSem{}In[2]);\r\n", vsSemanticId, variant, vsSemanticId, vsSemanticId, vsSemanticId));
	}
	gsSrc.append(fmt::format("gl_Position = gen4thVertex{}(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position);\r\n", variant));
	gsSrc.append("EmitVertex();\r\n");
}

void rectsEmulationGS_outputVerticesCode(std::string& gsSrc, LatteDecompilerShader* vertexShader, LatteShaderPSInputTable* psInputTable, sint32 p0, sint32 p1, sint32 p2, sint32 p3, const char* variant, const LatteContextRegister& latteRegister)
{
	sint32 pList[4] = { p0, p1, p2, p3 };
	for (sint32 i = 0; i < 4; i++)
	{
		if (pList[i] == 3)
			rectsEmulationGS_outputGeneratedVertex(gsSrc, vertexShader, psInputTable, variant, latteRegister);
		else
			rectsEmulationGS_outputSingleVertex(gsSrc, vertexShader, psInputTable, pList[i], latteRegister);
	}
}

RendererShaderVk* rectsEmulationGS_generate(LatteDecompilerShader* vertexShader, const LatteContextRegister& latteRegister)
{
	std::string gsSrc;

	gsSrc.append("#version 450\r\n");

	LatteShaderPSInputTable* psInputTable = LatteSHRC_GetPSInputTable();

	// layout
	gsSrc.append("layout(triangles) in;\r\n");
	gsSrc.append("layout(triangle_strip) out;\r\n");
	gsSrc.append("layout(max_vertices = 4) out;\r\n");

	// inputs & outputs
	auto parameterMask = vertexShader->outputParameterMask;
	for (sint32 f = 0; f < 2; f++)
	{
		for (uint32 i = 0; i < 32; i++)
		{
			if ((parameterMask & (1 << i)) == 0)
				continue;
			sint32 vsSemanticId = psInputTable->getVertexShaderOutParamSemanticId(latteRegister.GetRawView(), i);
			if (vsSemanticId < 0)
				continue;
			auto psImport = psInputTable->getPSImportBySemanticId(vsSemanticId);
			if (psImport == nullptr)
				continue;

			gsSrc.append(fmt::format("layout(location = {}) ", psInputTable->getPSImportLocationBySemanticId(vsSemanticId)));
			if (psImport->isFlat)
				gsSrc.append("flat ");
			if (psImport->isNoPerspective)
				gsSrc.append("noperspective ");

			if (f == 0)
				gsSrc.append("in");
			else
				gsSrc.append("out");

			if (f == 0)
				gsSrc.append(fmt::format(" vec4 passParameterSem{}In[];\r\n", vsSemanticId));
			else
				gsSrc.append(fmt::format(" vec4 passParameterSem{}Out;\r\n", vsSemanticId));
		}
	}

	// gen function
	gsSrc.append("vec4 gen4thVertexA(vec4 a, vec4 b, vec4 c)\r\n");
	gsSrc.append("{\r\n");
	gsSrc.append("return b - (c - a);\r\n");
	gsSrc.append("}\r\n");

	gsSrc.append("vec4 gen4thVertexB(vec4 a, vec4 b, vec4 c)\r\n");
	gsSrc.append("{\r\n");
	gsSrc.append("return c - (b - a);\r\n");
	gsSrc.append("}\r\n");

	gsSrc.append("vec4 gen4thVertexC(vec4 a, vec4 b, vec4 c)\r\n");
	gsSrc.append("{\r\n");
	gsSrc.append("return c + (b - a);\r\n");
	gsSrc.append("}\r\n");

	// main
	gsSrc.append("void main()\r\n");
	gsSrc.append("{\r\n");

	// there are two possible winding orders that need different triangle generation:
	// 0 1
	// 2 3
	// and
	// 0 1
	// 3 2
	// all others are just symmetries of these cases

	// we can determine the case by comparing the distance 0<->1 and 0<->2

	gsSrc.append("float dist0_1 = length(gl_in[1].gl_Position.xy - gl_in[0].gl_Position.xy);\r\n");
	gsSrc.append("float dist0_2 = length(gl_in[2].gl_Position.xy - gl_in[0].gl_Position.xy);\r\n");
	gsSrc.append("float dist1_2 = length(gl_in[2].gl_Position.xy - gl_in[1].gl_Position.xy);\r\n");

	// emit vertices
	gsSrc.append("if(dist0_1 > dist0_2 && dist0_1 > dist1_2)\r\n");
	gsSrc.append("{\r\n");
	// p0 to p1 is diagonal
	rectsEmulationGS_outputVerticesCode(gsSrc, vertexShader, psInputTable, 2, 1, 0, 3, "A", latteRegister);
	gsSrc.append("} else if ( dist0_2 > dist0_1 && dist0_2 > dist1_2 ) {\r\n");
	// p0 to p2 is diagonal
	rectsEmulationGS_outputVerticesCode(gsSrc, vertexShader, psInputTable, 1, 2, 0, 3, "B", latteRegister);
	gsSrc.append("} else {\r\n");
	// p1 to p2 is diagonal
	rectsEmulationGS_outputVerticesCode(gsSrc, vertexShader, psInputTable, 0, 1, 2, 3, "C", latteRegister);
	gsSrc.append("}\r\n");

	gsSrc.append("}\r\n");

	auto vkShader = new RendererShaderVk(RendererShader::ShaderType::kGeometry, 0, 0, false, false, gsSrc);
	vkShader->PreponeCompilation(true);
	return vkShader;
}

/* geometry amplification */
// See Core/LatteGeometryAmp.h for the design rationale. Subdivides each input
// triangle 1->4 via edge midpoints, displacing the 3 new midpoints along a
// Phong-tessellation-style projection to round low-poly silhouettes exposed
// by 4K texture replacement (confirmed on Bayonetta 2's arms - real geometry
// limit, not a shading bug). Original corner vertices are passed through
// unmodified so amplified draws still seam correctly against anything
// unamplified sharing those vertices.

// emits the passthrough-varying declarations shared by rects and amplification
// (both iterate the same VS-output/PS-input intersection). Returns the list
// of semantic IDs actually declared, so the caller knows what it can use as
// normal-detection candidates.
static std::vector<sint32> geometryAmp_emitVaryingDecls(std::string& gsSrc, LatteDecompilerShader* vertexShader, LatteShaderPSInputTable* psInputTable, const LatteContextRegister& latteRegister)
{
	std::vector<sint32> semIds;
	auto parameterMask = vertexShader->outputParameterMask;
	for (sint32 f = 0; f < 2; f++)
	{
		for (uint32 i = 0; i < 32; i++)
		{
			if ((parameterMask & (1 << i)) == 0)
				continue;
			sint32 vsSemanticId = psInputTable->getVertexShaderOutParamSemanticId(latteRegister.GetRawView(), i);
			if (vsSemanticId < 0)
				continue;
			auto psImport = psInputTable->getPSImportBySemanticId(vsSemanticId);
			if (psImport == nullptr)
				continue;

			if (f == 0)
				semIds.emplace_back(vsSemanticId);

			gsSrc.append(fmt::format("layout(location = {}) ", psInputTable->getPSImportLocationBySemanticId(vsSemanticId)));
			if (psImport->isFlat)
				gsSrc.append("flat ");
			if (psImport->isNoPerspective)
				gsSrc.append("noperspective ");
			if (f == 0)
				gsSrc.append(fmt::format("in vec4 passParameterSem{}In[];\r\n", vsSemanticId));
			else
				gsSrc.append(fmt::format("out vec4 passParameterSem{}Out;\r\n", vsSemanticId));
		}
	}
	return semIds;
}

// emits one output vertex: gl_Position = pos, and each passthrough varying
// either copied straight from corner index `srcA` (srcB < 0) or lerped
// between corners srcA/srcB (edge midpoint - same weights as `pos` itself,
// which the caller already computed the same way, satisfying the brief's
// "UV interpolation with same barycentric weights as position" requirement)
static void geometryAmp_emitVertex(std::string& gsSrc, const std::vector<sint32>& semIds, const char* posExpr, sint32 srcA, sint32 srcB)
{
	for (sint32 semId : semIds)
	{
		if (srcB < 0)
			gsSrc.append(fmt::format("passParameterSem{}Out = passParameterSem{}In[{}];\r\n", semId, semId, srcA));
		else
			gsSrc.append(fmt::format("passParameterSem{}Out = (passParameterSem{}In[{}] + passParameterSem{}In[{}]) * 0.5;\r\n", semId, semId, srcA, semId, srcB));
	}
	gsSrc.append(fmt::format("gl_Position = {};\r\n", posExpr));
	gsSrc.append("EmitVertex();\r\n");
}

RendererShaderVk* geometryAmp_generate(LatteDecompilerShader* vertexShader, const LatteContextRegister& latteRegister)
{
	const auto& config = LatteGeometryAmp::GetConfig();

	std::string gsSrc;
	gsSrc.append("#version 450\r\n");
	gsSrc.append("layout(triangles) in;\r\n");
	gsSrc.append("layout(triangle_strip, max_vertices = 12) out;\r\n");

	LatteShaderPSInputTable* psInputTable = LatteSHRC_GetPSInputTable();
	std::vector<sint32> semIds = geometryAmp_emitVaryingDecls(gsSrc, vertexShader, psInputTable, latteRegister);

	gsSrc.append(fmt::format("const float kFactor = {:.6f};\r\n", config.factor));
	gsSrc.append(fmt::format("const float kTolerance = {:.6f};\r\n", config.normalDetectTolerance));

	// Phong-tessellation-style edge midpoint (Boubekeur & Alexa 2008). Divides
	// by w first so the projection happens in an approximately metric space
	// instead of raw (perspective-distorted) clip space, then re-homogenizes
	// back to a valid clip-space position for the rasterizer. Purely local to
	// the 2 edge endpoints (position + normal) -> two triangles sharing this
	// edge compute the identical midpoint independently, so there are no
	// cracks (same principle as PN-triangles/Phong tessellation in general,
	// simplified because a fixed midpoint subdivision only ever needs the
	// symmetric t=0.5 case).
	gsSrc.append(
	    // 2026-07-14: BUG FOUND (root cause of catastrophic map/environment
	    // corruption seen with kFactor==0, i.e. even with displacement fully
	    // disabled): this used to divide by w and re-homogenize UNCONDITIONALLY
	    // to compute the midpoint. That divide-then-multiply round trip is only
	    // equal to the correct clip-space linear midpoint (Pa+Pb)*0.5 when
	    // wa==wb - for two vertices at similar depth (small character-mesh
	    // triangles, e.g. Bayonetta's arm) the error is invisible, but for a
	    // large environment surface where a single triangle spans a big depth
	    // range (wa and wb far apart), the pre-divide/post-multiply result
	    // lands nowhere near the real edge - exactly the giant misplaced-shard
	    // corruption seen in-game, and exactly why it hit the map but not the
	    // character. Fix: always compute the CORRECT clip-space linear
	    // midpoint first, and only fall into the perspective-divided
	    // projection math to compute a small ADDITIVE displacement delta when
	    // actually displacing (hasNormal && kFactor>0) - never as a
	    // replacement for the base position\r\n"
	    "vec4 gaPhongMid(vec4 Pa, vec4 Pb, vec3 Na, vec3 Nb, bool hasNormal)\r\n"
	    "{\r\n"
	    "    vec4 flatMid = (Pa + Pb) * 0.5;\r\n"
	    "    if (!hasNormal || kFactor <= 0.0) return flatMid;\r\n"
	    "    float wa = abs(Pa.w) < 1e-6 ? 1e-6 : Pa.w;\r\n"
	    "    float wb = abs(Pb.w) < 1e-6 ? 1e-6 : Pb.w;\r\n"
	    "    vec3 pa = Pa.xyz / wa;\r\n"
	    "    vec3 pb = Pb.xyz / wb;\r\n"
	    "    vec3 lin = (pa + pb) * 0.5;\r\n"
	    "    vec3 projA = lin - dot(lin - pa, Na) * Na;\r\n"
	    "    vec3 projB = lin - dot(lin - pb, Nb) * Nb;\r\n"
	    "    vec3 phong = (projA + projB) * 0.5;\r\n"
	    "    vec3 deltaNDC = (phong - lin) * kFactor;\r\n"
	    "    float wMid = flatMid.w;\r\n"
	    "    return vec4(flatMid.xyz + deltaNDC * wMid, wMid);\r\n"
	    "}\r\n");

	gsSrc.append("void main()\r\n{\r\n");
	gsSrc.append("vec4 P0 = gl_in[0].gl_Position;\r\n");
	gsSrc.append("vec4 P1 = gl_in[1].gl_Position;\r\n");
	gsSrc.append("vec4 P2 = gl_in[2].gl_Position;\r\n");

	// normal auto-detect: try every passthrough varying as a candidate
	// direction, keep whichever reads closest to unit length across the
	// triangle's 3 corners (see LatteGeometryAmp.h - no per-game knowledge of
	// which semantic is "the normal" is required or assumed).
	gsSrc.append("vec3 n0 = vec3(0.0), n1 = vec3(0.0), n2 = vec3(0.0);\r\n");
	gsSrc.append("float bestScore = kTolerance;\r\n");
	gsSrc.append("bool hasNormal = false;\r\n");
	gsSrc.append("{\r\n float avgLen, score;\r\n");
	for (sint32 semId : semIds)
	{
		gsSrc.append(fmt::format(
		    "avgLen = (length(passParameterSem{0}In[0].xyz) + length(passParameterSem{0}In[1].xyz) + length(passParameterSem{0}In[2].xyz)) / 3.0;\r\n"
		    "score = abs(avgLen - 1.0);\r\n"
		    "if (score < bestScore) {{ bestScore = score; hasNormal = true; n0 = passParameterSem{0}In[0].xyz; n1 = passParameterSem{0}In[1].xyz; n2 = passParameterSem{0}In[2].xyz; }}\r\n",
		    semId));
	}
	gsSrc.append("}\r\n");
	gsSrc.append("if (hasNormal) { n0 = normalize(n0); n1 = normalize(n1); n2 = normalize(n2); }\r\n");

	gsSrc.append("vec4 M01 = gaPhongMid(P0, P1, n0, n1, hasNormal);\r\n");
	gsSrc.append("vec4 M12 = gaPhongMid(P1, P2, n1, n2, hasNormal);\r\n");
	gsSrc.append("vec4 M20 = gaPhongMid(P2, P0, n2, n0, hasNormal);\r\n");

	// 4 sub-triangles from the classic single-step triangular quadrisection
	// (0,M01,M20) (M01,1,M12) (M20,M12,2) (M01,M12,M20) - each keeps the
	// original triangle's winding order, so front/back-face culling is
	// unaffected.
	geometryAmp_emitVertex(gsSrc, semIds, "P0",  0, -1);
	geometryAmp_emitVertex(gsSrc, semIds, "M01", 0, 1);
	geometryAmp_emitVertex(gsSrc, semIds, "M20", 2, 0);
	gsSrc.append("EndPrimitive();\r\n");

	geometryAmp_emitVertex(gsSrc, semIds, "M01", 0, 1);
	geometryAmp_emitVertex(gsSrc, semIds, "P1",  1, -1);
	geometryAmp_emitVertex(gsSrc, semIds, "M12", 1, 2);
	gsSrc.append("EndPrimitive();\r\n");

	geometryAmp_emitVertex(gsSrc, semIds, "M20", 2, 0);
	geometryAmp_emitVertex(gsSrc, semIds, "M12", 1, 2);
	geometryAmp_emitVertex(gsSrc, semIds, "P2",  2, -1);
	gsSrc.append("EndPrimitive();\r\n");

	geometryAmp_emitVertex(gsSrc, semIds, "M01", 0, 1);
	geometryAmp_emitVertex(gsSrc, semIds, "M12", 1, 2);
	geometryAmp_emitVertex(gsSrc, semIds, "M20", 2, 0);
	gsSrc.append("EndPrimitive();\r\n");

	gsSrc.append("}\r\n");

	auto vkShader = new RendererShaderVk(RendererShader::ShaderType::kGeometry, 0, 0, false, false, gsSrc);
	vkShader->PreponeCompilation(true);
	return vkShader;
}

/* pipeline compiler and cache helper */

extern std::atomic_int g_compiling_pipelines;
extern std::atomic_int g_compiling_pipelines_async;
extern std::atomic_uint64_t g_compiling_pipelines_syncTimeSum;

PipelineCompiler::PipelineCompiler() {};
PipelineCompiler::~PipelineCompiler()
{
	if (m_vkrObjPipeline)
		m_vkrObjPipeline->decRef();
	if (m_renderPassObj)
		m_renderPassObj->decRef();
};

VkFormat PipelineCompiler::GetVertexFormat(uint8 format)
{
	switch (format)
	{
	case FMT_32_32_32_32_FLOAT:
		return VK_FORMAT_R32G32B32A32_UINT;
	case FMT_32_32_32_FLOAT:
		return VK_FORMAT_R32G32B32_UINT;
	case FMT_32_32_FLOAT:
		return VK_FORMAT_R32G32_UINT;
	case FMT_32_FLOAT:
		return VK_FORMAT_R32_UINT;
	case FMT_8_8_8_8:
		return VK_FORMAT_R8G8B8A8_UINT;
	case FMT_8_8_8:
		return VK_FORMAT_R8G8B8_UINT;
	case FMT_8_8:
		return VK_FORMAT_R8G8_UINT;
	case FMT_8:
		return VK_FORMAT_R8_UINT;
	case FMT_32_32_32_32:
		return VK_FORMAT_R32G32B32A32_UINT;
	case FMT_32_32_32:
		return VK_FORMAT_R32G32B32_UINT;
	case FMT_32_32:
		return VK_FORMAT_R32G32_UINT;
	case FMT_32:
		return VK_FORMAT_R32_UINT;
	case FMT_16_16_16_16:
		return VK_FORMAT_R16G16B16A16_UINT; // verified to match OpenGL
	case FMT_16_16_16:
		return VK_FORMAT_R16G16B16_UINT;
	case FMT_16_16:
		return VK_FORMAT_R16G16_UINT;
	case FMT_16:
		return VK_FORMAT_R16_UINT;
	case FMT_16_16_16_16_FLOAT:
		return VK_FORMAT_R16G16B16A16_UINT; // verified to match OpenGL
	case FMT_16_16_16_FLOAT:
		return VK_FORMAT_R16G16B16_UINT;
	case FMT_16_16_FLOAT:
		return VK_FORMAT_R16G16_UINT;
	case FMT_16_FLOAT:
		return VK_FORMAT_R16_UINT;
	case FMT_2_10_10_10:
		return VK_FORMAT_R32_UINT; // verified to match OpenGL
	default:
		cemuLog_log(LogType::Force, "Unsupported vertex format: {:02x}", format);
		assert_dbg();
		return VK_FORMAT_UNDEFINED;
	}
}

static VkBlendOp GetVkBlendOp(Latte::LATTE_CB_BLENDN_CONTROL::E_COMBINEFUNC combineFunc)
{
	switch (combineFunc)
	{
	case Latte::LATTE_CB_BLENDN_CONTROL::E_COMBINEFUNC::DST_PLUS_SRC:
		return VK_BLEND_OP_ADD;
	case Latte::LATTE_CB_BLENDN_CONTROL::E_COMBINEFUNC::SRC_MINUS_DST:
		return VK_BLEND_OP_SUBTRACT;
	case Latte::LATTE_CB_BLENDN_CONTROL::E_COMBINEFUNC::MIN_DST_SRC:
		return VK_BLEND_OP_MIN;
	case Latte::LATTE_CB_BLENDN_CONTROL::E_COMBINEFUNC::MAX_DST_SRC:
		return VK_BLEND_OP_MAX;
	case Latte::LATTE_CB_BLENDN_CONTROL::E_COMBINEFUNC::DST_MINUS_SRC:
		return VK_BLEND_OP_REVERSE_SUBTRACT;
	default:
		cemu_assert_suspicious();
		return VK_BLEND_OP_ADD;
	}
}

static VkBlendFactor GetVkBlendFactor(Latte::LATTE_CB_BLENDN_CONTROL::E_BLENDFACTOR factor)
{
	const VkBlendFactor factors[] =
	{
		/* 0x00 */ VK_BLEND_FACTOR_ZERO,
		/* 0x01 */ VK_BLEND_FACTOR_ONE,
		/* 0x02 */ VK_BLEND_FACTOR_SRC_COLOR,
		/* 0x03 */ VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
		/* 0x04 */ VK_BLEND_FACTOR_SRC_ALPHA,
		/* 0x05 */ VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		/* 0x06 */ VK_BLEND_FACTOR_DST_ALPHA,
		/* 0x07 */ VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
		/* 0x08 */ VK_BLEND_FACTOR_DST_COLOR,
		/* 0x09 */ VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
		/* 0x0A */ VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
		/* 0x0B */ VK_BLEND_FACTOR_MAX_ENUM, // todo
		/* 0x0C */ VK_BLEND_FACTOR_MAX_ENUM, // todo
		/* 0x0D */ VK_BLEND_FACTOR_CONSTANT_COLOR,
		/* 0x0E */ VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
		/* 0x0F */ VK_BLEND_FACTOR_SRC1_COLOR,
		/* 0x10 */ VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
		/* 0x11 */ VK_BLEND_FACTOR_SRC1_ALPHA,
		/* 0x12 */ VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
		/* 0x13 */ VK_BLEND_FACTOR_CONSTANT_ALPHA,
		/* 0x14 */ VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA
	};
	cemu_assert_debug((uint32)factor < std::size(factors));
	return factors[(uint32)factor];
}

bool PipelineCompiler::ConsumesBlendConstants(VkBlendFactor blendFactor)
{
	if (blendFactor == VK_BLEND_FACTOR_CONSTANT_COLOR ||
		blendFactor == VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR ||
		blendFactor == VK_BLEND_FACTOR_CONSTANT_ALPHA ||
		blendFactor == VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
		return true;
	return false;
}

void PipelineCompiler::CreateDescriptorSetLayout(VulkanRenderer* vkRenderer, LatteDecompilerShader* shader, VkDescriptorSetLayout& layout, PipelineInfo* vkrPipelineInfo)
{
	// create vertex shader descriptor set
	std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;

	VkShaderStageFlags stageFlags = 0;
	uint32 stageIndex = 0;
	if (shader->shaderType == LatteConst::ShaderType::Vertex)
	{
		stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		stageIndex = VulkanRendererConst::SHADER_STAGE_INDEX_VERTEX;
	}
	else if (shader->shaderType == LatteConst::ShaderType::Pixel)
	{
		stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		stageIndex = VulkanRendererConst::SHADER_STAGE_INDEX_FRAGMENT;
	}
	else if (shader->shaderType == LatteConst::ShaderType::Geometry)
	{
		stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;
		stageIndex = VulkanRendererConst::SHADER_STAGE_INDEX_GEOMETRY;
	}
	// attributes
	// -> not part of descriptor

	// textures
	sint32 textureBindingBase = shader->resourceMapping.getTextureBaseBindingPoint();
	if (textureBindingBase >= 0)
	{
		sint32 textureCount = shader->resourceMapping.getTextureCount();
		for (sint32 i = 0; i < textureCount; i++)
		{
			VkDescriptorSetLayoutBinding entry{};
			entry.binding = (uint32)textureBindingBase + i;
			entry.descriptorCount = 1;
			entry.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			entry.pImmutableSamplers = nullptr;
			entry.stageFlags = stageFlags;
			descriptorSetLayoutBindings.emplace_back(entry);
		}
	}

	// uniform buffers
	if (shader->resourceMapping.uniformVarsBufferBindingPoint >= 0)
	{
		VkDescriptorSetLayoutBinding entry{};
		entry.binding = shader->resourceMapping.uniformVarsBufferBindingPoint;
		entry.descriptorCount = 1;
		entry.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		entry.pImmutableSamplers = nullptr;
		entry.stageFlags = stageFlags;
		descriptorSetLayoutBindings.emplace_back(entry);
	}

	for (sint32 i = 0; i < LATTE_NUM_MAX_UNIFORM_BUFFERS; i++)
	{
		if (shader->resourceMapping.uniformBuffersBindingPoint[i] >= 0)
		{
			VkDescriptorSetLayoutBinding entry{};
			entry.binding = shader->resourceMapping.uniformBuffersBindingPoint[i];
			entry.descriptorCount = 1;
			entry.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
			entry.pImmutableSamplers = nullptr;
			entry.stageFlags = stageFlags;
			descriptorSetLayoutBindings.emplace_back(entry);

			vkrPipelineInfo->dynamicOffsetInfo.list_uniformBuffers[stageIndex].emplace_back((uint8)i);
		}
	}

	// storage buffer for TF
	if (shader->resourceMapping.tfStorageBindingPoint >= 0)
	{
		VkDescriptorSetLayoutBinding entry{};
		entry.binding = shader->resourceMapping.tfStorageBindingPoint;
		entry.descriptorCount = 1;
		entry.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		entry.pImmutableSamplers = nullptr;
		entry.stageFlags = stageFlags;
		descriptorSetLayoutBindings.emplace_back(entry);
	}

	if (shader->resourceMapping.uniformVarsBufferBindingPoint >= 0)
		vkrPipelineInfo->dynamicOffsetInfo.hasUniformVar[stageIndex] = true;
	if (shader->resourceMapping.hasUniformBuffers())
		vkrPipelineInfo->dynamicOffsetInfo.hasUniformBuffers[stageIndex] = true;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = descriptorSetLayoutBindings.size();
	layoutInfo.pBindings = descriptorSetLayoutBindings.data();

	if (vkCreateDescriptorSetLayout(vkRenderer->m_logicalDevice, &layoutInfo, nullptr, &layout) != VK_SUCCESS)
		vkRenderer->UnrecoverableError(fmt::format("Failed to create descriptor set layout for shader {0:#x}", shader->baseHash).c_str());
}

bool PipelineCompiler::InitShaderStages(VulkanRenderer* vkRenderer, RendererShaderVk* vkVertexShader, RendererShaderVk* vkPixelShader, RendererShaderVk* vkGeometryShader)
{
	// prepare shader stages
	cemu_assert_debug(vkVertexShader == nullptr || vkVertexShader->IsCompiled());
	cemu_assert_debug(vkPixelShader == nullptr || vkPixelShader->IsCompiled());
	cemu_assert_debug(vkGeometryShader == nullptr || vkGeometryShader->IsCompiled());

	if ((vkVertexShader && vkVertexShader->GetShaderModule() == VK_NULL_HANDLE) ||
		(vkGeometryShader && vkGeometryShader->GetShaderModule() == VK_NULL_HANDLE) ||
		(vkPixelShader && vkPixelShader->GetShaderModule() == VK_NULL_HANDLE))
	{
		cemuLog_log(LogType::Force, "Vulkan-Info: Pipeline creation failed due to invalid shader(s)");
		return false;
	}

	if (vkVertexShader)
		shaderStages.emplace_back(vkRenderer->CreatePipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vkVertexShader->GetShaderModule(), "main"));

	if (vkGeometryShader)
		shaderStages.emplace_back(vkRenderer->CreatePipelineShaderStageCreateInfo(VK_SHADER_STAGE_GEOMETRY_BIT, vkGeometryShader->GetShaderModule(), "main"));
	else if (m_rectEmulationGS)
		shaderStages.emplace_back(vkRenderer->CreatePipelineShaderStageCreateInfo(VK_SHADER_STAGE_GEOMETRY_BIT, m_rectEmulationGS->GetShaderModule(), "main"));
	else if (m_geometryAmpGS)
		shaderStages.emplace_back(vkRenderer->CreatePipelineShaderStageCreateInfo(VK_SHADER_STAGE_GEOMETRY_BIT, m_geometryAmpGS->GetShaderModule(), "main"));

	if (vkPixelShader)
		shaderStages.emplace_back(vkRenderer->CreatePipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, vkPixelShader->GetShaderModule(), "main"));

	return true;
}

void PipelineCompiler::InitVertexInputState(const LatteContextRegister& latteRegister, LatteDecompilerShader* vertexShader, LatteFetchShader* fetchShader)
{
	vertexInputAttributeDescription.reserve(16);
	vertexInputBindingDescription.reserve(fetchShader->bufferGroups.size());

	for (auto& bufferGroup : fetchShader->bufferGroups)
	{
		std::optional<LatteConst::VertexFetchType2> fetchType;

		for (sint32 j = 0; j < bufferGroup.attribCount; ++j)
		{
			auto& attr = bufferGroup.attrib[j];

			uint32 semanticId = vertexShader->resourceMapping.attributeMapping[attr.semanticId];
			if (semanticId == (uint32)-1)
				continue; // attribute not used?

			VkVertexInputAttributeDescription entry{};
			entry.location = semanticId;
			entry.offset = attr.offset;
			entry.binding = attr.attributeBufferIndex;
			entry.format = GetVertexFormat(attr.format);
			vertexInputAttributeDescription.emplace_back(entry);

			if (fetchType.has_value())
				cemu_assert_debug(fetchType == attr.fetchType);
			else
				fetchType = attr.fetchType;

			if (attr.fetchType == LatteConst::INSTANCE_DATA)
			{
				cemu_assert_debug(attr.aluDivisor == 1); // other divisor not yet supported
				// use VK_EXT_vertex_attribute_divisor
			}
		}

		uint32 bufferIndex = bufferGroup.attributeBufferIndex;
		uint32 bufferBaseRegisterIndex = mmSQ_VTX_ATTRIBUTE_BLOCK_START + bufferIndex * 7;
		uint32 bufferStride = (latteRegister.GetRawView()[bufferBaseRegisterIndex + 2] >> 11) & 0xFFFF;

		VkVertexInputBindingDescription entry{};
#if BOOST_OS_MACOS
		if (bufferStride % 4 != 0) {
			bufferStride = bufferStride + (4-(bufferStride % 4));
		}
#endif
		entry.stride = bufferStride;
		if (!fetchType.has_value() || fetchType == LatteConst::VertexFetchType2::VERTEX_DATA)
			entry.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		else if (fetchType == LatteConst::VertexFetchType2::INSTANCE_DATA)
			entry.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
		else
		{
			cemu_assert(false);
		}
		entry.binding = bufferIndex;
		vertexInputBindingDescription.emplace_back(entry);
	}

	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = vertexInputBindingDescription.size();
	vertexInputInfo.pVertexBindingDescriptions = vertexInputBindingDescription.data();
	vertexInputInfo.vertexAttributeDescriptionCount = vertexInputAttributeDescription.size();
	vertexInputInfo.pVertexAttributeDescriptions = vertexInputAttributeDescription.data();
}

void PipelineCompiler::InitInputAssemblyState(const Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE primitiveMode)
{
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.primitiveRestartEnable = VK_TRUE;
	switch (primitiveMode)
	{
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::POINTS:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		inputAssembly.primitiveRestartEnable = false;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::LINES:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		inputAssembly.primitiveRestartEnable = false;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::LINE_STRIP:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::LINE_LOOP:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; // line loops are emulated as line strips with an extra connecting strip at the end
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::LINE_STRIP_ADJACENT: // Tropical Freeze level 3-6
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLES:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = false;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLE_FAN:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLE_STRIP:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::QUADS:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // quads are emulated as 2 triangles
		inputAssembly.primitiveRestartEnable = false;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::QUAD_STRIP:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // quad strips are emulated as (count-2)/2 triangles
		inputAssembly.primitiveRestartEnable = false;
		break;
	case Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::RECTS:
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // rects are emulated as 2 triangles
		inputAssembly.primitiveRestartEnable = false;
		break;
	default:
		cemuLog_logDebug(LogType::Force, "Vulkan-Unsupported: Graphics pipeline with primitive mode {} created", primitiveMode);
		cemu_assert_debug(false);
	}
}

void PipelineCompiler::InitViewportState()
{
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;
}

void PipelineCompiler::InitRasterizerState(const LatteContextRegister& latteRegister, VulkanRenderer* vkRenderer, bool isPrimitiveRect, bool& usesDepthBias)
{
	// polygon control
	const auto& polygonControlReg = latteRegister.PA_SU_SC_MODE_CNTL;
	const auto frontFace = polygonControlReg.get_FRONT_FACE();
	uint32 cullFront = polygonControlReg.get_CULL_FRONT();
	uint32 cullBack = polygonControlReg.get_CULL_BACK();
	uint32 polyOffsetFrontEnable = polygonControlReg.get_OFFSET_FRONT_ENABLED();

	cemu_assert_debug(LatteGPUState.contextNew.PA_CL_CLIP_CNTL.get_ZCLIP_NEAR_DISABLE() == LatteGPUState.contextNew.PA_CL_CLIP_CNTL.get_ZCLIP_FAR_DISABLE()); // near or far clipping can be disabled individually
	bool zClipEnable = LatteGPUState.contextNew.PA_CL_CLIP_CNTL.get_ZCLIP_FAR_DISABLE() == false;

	// z-clipping
	rasterizerExt.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT;
	rasterizerExt.depthClipEnable = zClipEnable;
	rasterizerExt.flags = 0;

	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.pNext = &rasterizerExt;
	rasterizer.rasterizerDiscardEnable = LatteGPUState.contextNew.PA_CL_CLIP_CNTL.get_DX_RASTERIZATION_KILL();
	// GX2SetSpecialState(0, true) workaround
	if (!LatteGPUState.contextNew.PA_CL_VTE_CNTL.get_VPORT_X_OFFSET_ENA())
		rasterizer.rasterizerDiscardEnable = false;

	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	if (vkRenderer->m_featureControl.deviceExtensions.nv_fill_rectangle && isPrimitiveRect)
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL_RECTANGLE_NV;

	rasterizer.depthClampEnable = VK_TRUE; // depth clamping is always enabled

	rasterizer.lineWidth = 1.0f; // TODO -> mmPA_SU_LINE_CNTL

	usesDepthBias = polyOffsetFrontEnable;
	if (polyOffsetFrontEnable)
	{
		rasterizer.depthBiasEnable = VK_TRUE;
		// initialize to zero, set dynamically via vkCmdSetDepthBias
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
	}
	else
		rasterizer.depthBiasEnable = VK_FALSE;

	// todo - how does culling behave with rects?
	// right now we just assume that their winding is always CW
	if (isPrimitiveRect)
	{
		if (frontFace == Latte::LATTE_PA_SU_SC_MODE_CNTL::E_FRONTFACE::CW)
			cullFront = cullBack;
		else
			cullBack = cullFront;
	}

	if (cullFront && cullBack)
		rasterizer.cullMode = VK_CULL_MODE_FRONT_AND_BACK;
	else if (cullFront)
		rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
	else if (cullBack)
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	else
		rasterizer.cullMode = VK_CULL_MODE_NONE;

	if (frontFace == Latte::LATTE_PA_SU_SC_MODE_CNTL::E_FRONTFACE::CCW)
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	else
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

	// multisampling
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
}

bool _IsVkIntegerFormat(VkFormat fmt)
{
	return
		// 8bit integer formats
		fmt == VK_FORMAT_R8_UINT || fmt == VK_FORMAT_R8_SINT ||
		fmt == VK_FORMAT_R8G8_UINT || fmt == VK_FORMAT_R8G8_SINT ||
		fmt == VK_FORMAT_R8G8B8_UINT || fmt == VK_FORMAT_R8G8B8_SINT ||
		fmt == VK_FORMAT_R8G8B8A8_UINT || fmt == VK_FORMAT_R8G8B8A8_SINT ||
		fmt == VK_FORMAT_B8G8R8A8_UINT || fmt == VK_FORMAT_B8G8R8A8_SINT ||
		// 16bit integer formats
		fmt == VK_FORMAT_R16_UINT || fmt == VK_FORMAT_R16_SINT ||
		fmt == VK_FORMAT_R16G16_UINT || fmt == VK_FORMAT_R16G16_SINT ||
		fmt == VK_FORMAT_R16G16B16_UINT || fmt == VK_FORMAT_R16G16B16_SINT ||
		fmt == VK_FORMAT_R16G16B16A16_UINT || fmt == VK_FORMAT_R16G16B16A16_SINT ||
		// 32bit integer formats
		fmt == VK_FORMAT_R32_UINT || fmt == VK_FORMAT_R32_SINT ||
		fmt == VK_FORMAT_R32G32_UINT || fmt == VK_FORMAT_R32G32_SINT ||
		fmt == VK_FORMAT_R32G32B32_UINT || fmt == VK_FORMAT_R32G32B32_SINT ||
		fmt == VK_FORMAT_R32G32B32A32_UINT || fmt == VK_FORMAT_R32G32B32A32_SINT;
}


void PipelineCompiler::InitBlendState(const LatteContextRegister& latteRegister, PipelineInfo* pipelineInfo, bool& usesBlendConstants, VKRObjectRenderPass* renderPassObj)
{
	const Latte::LATTE_CB_COLOR_CONTROL& colorControlReg = latteRegister.CB_COLOR_CONTROL;
	uint32 blendEnableMask = colorControlReg.get_BLEND_MASK();
	uint32 renderTargetMask = latteRegister.CB_TARGET_MASK.get_MASK();

	usesBlendConstants = false;

	for (size_t i = 0; i < colorBlendAttachments.size(); i++)
	{
		auto& entry = colorBlendAttachments[i];
		if (((blendEnableMask & (1 << i))) != 0)
			entry.blendEnable = VK_TRUE;
		else
			entry.blendEnable = VK_FALSE;

		if (entry.blendEnable != VK_FALSE && _IsVkIntegerFormat(renderPassObj->GetColorFormat(i)))
		{
			// force-disable blending for integer formats
			entry.blendEnable = VK_FALSE;
		}
		
		const auto& blendControlReg = latteRegister.CB_BLENDN_CONTROL[i];

		entry.colorWriteMask = (renderTargetMask >> (i * 4)) & 0xF;
		entry.colorBlendOp = GetVkBlendOp(blendControlReg.get_COLOR_COMB_FCN());
		entry.srcColorBlendFactor = GetVkBlendFactor(blendControlReg.get_COLOR_SRCBLEND());
		entry.dstColorBlendFactor = GetVkBlendFactor(blendControlReg.get_COLOR_DSTBLEND());
		if (blendControlReg.get_SEPARATE_ALPHA_BLEND())
		{
			entry.alphaBlendOp = GetVkBlendOp(blendControlReg.get_ALPHA_COMB_FCN());
			entry.srcAlphaBlendFactor = GetVkBlendFactor(blendControlReg.get_ALPHA_SRCBLEND());
			entry.dstAlphaBlendFactor = GetVkBlendFactor(blendControlReg.get_ALPHA_DSTBLEND());
		}
		else
		{
			entry.alphaBlendOp = entry.colorBlendOp;
			entry.srcAlphaBlendFactor = entry.srcColorBlendFactor;
			entry.dstAlphaBlendFactor = entry.dstColorBlendFactor;
		}

		usesBlendConstants |= ConsumesBlendConstants(entry.srcColorBlendFactor);
		usesBlendConstants |= ConsumesBlendConstants(entry.dstColorBlendFactor);
		usesBlendConstants |= ConsumesBlendConstants(entry.srcAlphaBlendFactor);
		usesBlendConstants |= ConsumesBlendConstants(entry.dstAlphaBlendFactor);
	}

	// setup VkPipelineColorBlendStateCreateInfo
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

	const auto logicOp = colorControlReg.get_ROP();
	if (logicOp == Latte::LATTE_CB_COLOR_CONTROL::E_LOGICOP::COPY)
	{
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
	}
	else
	{
		colorBlending.logicOpEnable = VK_TRUE;
		switch (logicOp)
		{
		case Latte::LATTE_CB_COLOR_CONTROL::E_LOGICOP::SET:
			colorBlending.logicOp = VK_LOGIC_OP_SET;
			break;
		case Latte::LATTE_CB_COLOR_CONTROL::E_LOGICOP::CLEAR:
			colorBlending.logicOp = VK_LOGIC_OP_CLEAR;
			break;
		case Latte::LATTE_CB_COLOR_CONTROL::E_LOGICOP::OR:
			colorBlending.logicOp = VK_LOGIC_OP_OR;
			break;
		default:
			colorBlending.logicOp = VK_LOGIC_OP_COPY;
			cemu_assert_unimplemented();
		}
	}

	colorBlending.attachmentCount = colorBlendAttachments.size();
	colorBlending.pAttachments = colorBlendAttachments.data();

	// we use VK_DYNAMIC_STATE_BLEND_CONSTANTS, the blend constants here don't matter
	colorBlending.blendConstants[0] = 0;
	colorBlending.blendConstants[1] = 0;
	colorBlending.blendConstants[2] = 0;
	colorBlending.blendConstants[3] = 0;
}

void PipelineCompiler::InitDescriptorSetLayouts(VulkanRenderer* vkRenderer, PipelineInfo* vkrPipelineInfo, LatteDecompilerShader* vertexShader, LatteDecompilerShader* pixelShader, LatteDecompilerShader* geometryShader)
{
	auto vkObjPipeline = vkrPipelineInfo->m_vkrObjPipeline;

	if (vertexShader)
	{
		cemu_assert_debug(descriptorSetLayoutCount == 0);
		CreateDescriptorSetLayout(vkRenderer, vertexShader, descriptorSetLayout[descriptorSetLayoutCount], vkrPipelineInfo);
		vkObjPipeline->vertexDSL = descriptorSetLayout[descriptorSetLayoutCount];
		descriptorSetLayoutCount++;
	}

	if (pixelShader)
	{
		cemu_assert_debug(descriptorSetLayoutCount == 1);
		CreateDescriptorSetLayout(vkRenderer, pixelShader, descriptorSetLayout[descriptorSetLayoutCount], vkrPipelineInfo);
		vkObjPipeline->pixelDSL = descriptorSetLayout[descriptorSetLayoutCount];
		descriptorSetLayoutCount++;
	}
	else if (geometryShader)
	{
		// if no pixel shader is present, create empty placeholder descriptor set layout (geometry shader set must be at index 2)
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 0;
		layoutInfo.pBindings = nullptr;
		if (vkCreateDescriptorSetLayout(vkRenderer->m_logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout[descriptorSetLayoutCount]) != VK_SUCCESS)
			vkRenderer->UnrecoverableError(fmt::format("Failed to create placeholder descriptor set layout for shader {0:#x}", geometryShader->baseHash).c_str());
		descriptorSetLayoutCount++;
	}

	if (geometryShader)
	{
		cemu_assert_debug(descriptorSetLayoutCount == 2);
		CreateDescriptorSetLayout(vkRenderer, geometryShader, descriptorSetLayout[descriptorSetLayoutCount], vkrPipelineInfo);
		vkObjPipeline->geometryDSL = descriptorSetLayout[descriptorSetLayoutCount];
		descriptorSetLayoutCount++;
	}
}

void PipelineCompiler::InitDepthStencilState()
{
	// get depth control parameters
	bool depthEnable = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_Z_ENABLE();
	auto depthFunc = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_Z_FUNC();
	bool depthWriteEnable = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_Z_WRITE_ENABLE();

	// setup VkPipelineDepthStencilStateCreateInfo
	depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilState.depthTestEnable = depthEnable ? VK_TRUE : VK_FALSE;
	depthStencilState.depthWriteEnable = depthWriteEnable ? VK_TRUE : VK_FALSE;

	static const VkCompareOp vkDepthCompareTable[8] =
	{
		VK_COMPARE_OP_NEVER,
		VK_COMPARE_OP_LESS,
		VK_COMPARE_OP_EQUAL,
		VK_COMPARE_OP_LESS_OR_EQUAL,
		VK_COMPARE_OP_GREATER,
		VK_COMPARE_OP_NOT_EQUAL,
		VK_COMPARE_OP_GREATER_OR_EQUAL,
		VK_COMPARE_OP_ALWAYS
	};

	depthStencilState.depthCompareOp = vkDepthCompareTable[(size_t)depthFunc];

	depthStencilState.depthBoundsTestEnable = false; // todo
	depthStencilState.minDepthBounds = 0.0f;
	depthStencilState.maxDepthBounds = 1.0f;

	// get stencil control parameters
	bool stencilEnable = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_ENABLE();
	bool backStencilEnable = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_BACK_STENCIL_ENABLE();
	auto frontStencilFunc = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_FUNC_F();
	auto frontStencilZPass = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_ZPASS_F();
	auto frontStencilZFail = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_ZFAIL_F();
	auto frontStencilFail = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_FAIL_F();
	auto backStencilFunc = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_FUNC_B();
	auto backStencilZPass = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_ZPASS_B();
	auto backStencilZFail = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_ZFAIL_B();
	auto backStencilFail = LatteGPUState.contextNew.DB_DEPTH_CONTROL.get_STENCIL_FAIL_B();
	// get stencil control parameters
	uint32 stencilCompareMaskFront = LatteGPUState.contextNew.DB_STENCILREFMASK.get_STENCILMASK_F();
	uint32 stencilWriteMaskFront = LatteGPUState.contextNew.DB_STENCILREFMASK.get_STENCILWRITEMASK_F();
	uint32 stencilRefFront = LatteGPUState.contextNew.DB_STENCILREFMASK.get_STENCILREF_F();
	uint32 stencilCompareMaskBack = LatteGPUState.contextNew.DB_STENCILREFMASK_BF.get_STENCILMASK_B();
	uint32 stencilWriteMaskBack = LatteGPUState.contextNew.DB_STENCILREFMASK_BF.get_STENCILWRITEMASK_B();
	uint32 stencilRefBack = LatteGPUState.contextNew.DB_STENCILREFMASK_BF.get_STENCILREF_B();

	static const VkStencilOp stencilOpTable[8] = {
		VK_STENCIL_OP_KEEP,
		VK_STENCIL_OP_ZERO,
		VK_STENCIL_OP_REPLACE,
		VK_STENCIL_OP_INCREMENT_AND_CLAMP,
		VK_STENCIL_OP_DECREMENT_AND_CLAMP,
		VK_STENCIL_OP_INVERT,
		VK_STENCIL_OP_INCREMENT_AND_WRAP,
		VK_STENCIL_OP_DECREMENT_AND_WRAP
	};

	depthStencilState.stencilTestEnable = stencilEnable ? VK_TRUE : VK_FALSE;

	depthStencilState.front.reference = stencilRefFront;
	depthStencilState.front.compareMask = stencilCompareMaskFront;
	depthStencilState.front.writeMask = stencilWriteMaskFront;
	depthStencilState.front.compareOp = vkDepthCompareTable[(size_t)frontStencilFunc];
	depthStencilState.front.depthFailOp = stencilOpTable[(size_t)frontStencilZFail];
	depthStencilState.front.failOp = stencilOpTable[(size_t)frontStencilFail];
	depthStencilState.front.passOp = stencilOpTable[(size_t)frontStencilZPass];

	if (backStencilEnable)
	{
		depthStencilState.back.reference = stencilRefBack;
		depthStencilState.back.compareMask = stencilCompareMaskBack;
		depthStencilState.back.writeMask = stencilWriteMaskBack;
		depthStencilState.back.compareOp = vkDepthCompareTable[(size_t)backStencilFunc];
		depthStencilState.back.depthFailOp = stencilOpTable[(size_t)backStencilZFail];
		depthStencilState.back.failOp = stencilOpTable[(size_t)backStencilFail];
		depthStencilState.back.passOp = stencilOpTable[(size_t)backStencilZPass];
	}
	else
	{
		depthStencilState.back.reference = stencilRefFront;
		depthStencilState.back.compareMask = stencilCompareMaskFront;
		depthStencilState.back.writeMask = stencilWriteMaskFront;
		depthStencilState.back.compareOp = vkDepthCompareTable[(size_t)frontStencilFunc];
		depthStencilState.back.depthFailOp = stencilOpTable[(size_t)frontStencilZFail];
		depthStencilState.back.failOp = stencilOpTable[(size_t)frontStencilFail];
		depthStencilState.back.passOp = stencilOpTable[(size_t)frontStencilZPass];
	}
}

void PipelineCompiler::InitDynamicState(PipelineInfo* pipelineInfo, bool usesBlendConstants, bool usesDepthBias)
{

	if (usesBlendConstants)
	{
		dynamicStates.emplace_back(VK_DYNAMIC_STATE_BLEND_CONSTANTS);
		pipelineInfo->usesBlendConstants = true;
	}
	if (usesDepthBias)
	{
		dynamicStates.emplace_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
		pipelineInfo->usesDepthBias = true;
	}

	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = dynamicStates.size();
	dynamicState.pDynamicStates = dynamicStates.data();
}

bool PipelineCompiler::InitFromCurrentGPUState(PipelineInfo* pipelineInfo, const LatteContextRegister& latteRegister, VKRObjectRenderPass* renderPassObj)
{
	VulkanRenderer* vkRenderer = VulkanRenderer::GetInstance();

	// ##########################################################################################################################################
	bool isPrimitiveRect = false;
	const auto primitiveMode = latteRegister.VGT_PRIMITIVE_TYPE.get_PRIMITIVE_MODE();
	isPrimitiveRect = (primitiveMode == Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::RECTS);

	m_fetchShader = pipelineInfo->fetchShader;
	m_vkVertexShader = pipelineInfo->vertexShaderVk;
	m_vkPixelShader = pipelineInfo->pixelShaderVk;
	m_vkGeometryShader = pipelineInfo->geometryShaderVk;
	m_vkrObjPipeline = pipelineInfo->m_vkrObjPipeline;
	m_renderPassObj = renderPassObj;

	// if required generate RECT emulation geometry shader
	if (!vkRenderer->m_featureControl.deviceExtensions.nv_fill_rectangle && isPrimitiveRect)
	{
		cemu_assert(m_vkGeometryShader == nullptr); // todo - handle cases where the game already provides a GS
		m_rectEmulationGS = rectsEmulationGS_generate(pipelineInfo->vertexShader, latteRegister);
		pipelineInfo->rectEmulationGS = m_rectEmulationGS;
	}
	else if (LatteGeometryAmp::GetConfig().enabled && m_vkGeometryShader == nullptr &&
	         (primitiveMode == Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLES ||
	          primitiveMode == Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLE_STRIP ||
	          primitiveMode == Latte::LATTE_VGT_PRIMITIVE_TYPE::E_PRIMITIVE_TYPE::TRIANGLE_FAN) &&
	         pipelineInfo->vertexShader->outputParameterMask != 0)
	{
		// deliberately excludes QUADS/QUAD_STRIP/RECTS - those are triangle
		// topology by the time they reach Vulkan but carry their own emulation
		// requirements (RECTS needs the 4th-vertex-reconstruction GS above);
		// only plain triangle draws (the common case for skinned character
		// meshes, our actual target) get amplified
		m_geometryAmpGS = geometryAmp_generate(pipelineInfo->vertexShader, latteRegister);
		pipelineInfo->geometryAmpGS = m_geometryAmpGS;
	}

	// ##########################################################################################################################################

	pipelineInfo->primitiveMode = primitiveMode;
	InitVertexInputState(latteRegister, pipelineInfo->vertexShader, pipelineInfo->fetchShader);
	InitInputAssemblyState(primitiveMode);
	InitViewportState();
	bool usesDepthBias = false;
	InitRasterizerState(latteRegister, vkRenderer, isPrimitiveRect, usesDepthBias);
	bool usesBlendConstants = false;
	InitBlendState(latteRegister, pipelineInfo, usesBlendConstants, renderPassObj);
	InitDescriptorSetLayouts(vkRenderer, pipelineInfo, pipelineInfo->vertexShader, pipelineInfo->pixelShader, pipelineInfo->geometryShader);

	// ##########################################################################################################################################

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = descriptorSetLayoutCount;
	pipelineLayoutInfo.pSetLayouts = descriptorSetLayout;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	VkResult result = vkCreatePipelineLayout(vkRenderer->m_logicalDevice, &pipelineLayoutInfo, nullptr, &m_pipeline_layout);
	if (result != VK_SUCCESS)
	{
		cemuLog_log(LogType::Force, "Failed to create pipeline layout: {}", result);
		s_nvidiaWorkaround.unlock();
		return false;
	}

	// ###################################################

	InitDepthStencilState();

	// ##########################################################################################################################################

	InitDynamicState(pipelineInfo, usesBlendConstants, usesDepthBias);

	// ##########################################################################################################################################

	pipelineInfo->m_vkrObjPipeline->pipeline_layout = m_pipeline_layout;

	// increment ref counter for vkrObjPipeline and renderpass object to make sure they dont get released while we are using them
	m_vkrObjPipeline->incRef();
	renderPassObj->incRef();
	return true;
}

bool PipelineCompiler::Compile(bool forceCompile, bool isRenderThread, bool showInOverlay)
{
	VulkanRenderer* vkRenderer = VulkanRenderer::GetInstance();

	if (!vkRenderer->m_featureControl.deviceExtensions.pipeline_creation_cache_control)
		forceCompile = true; // if VK_EXT_pipeline_creation_cache_control is not supported we always force synchronous compilation

	if (!forceCompile)
	{
		// fail early if some shader stages are not compiled
		if (m_vkVertexShader && m_vkVertexShader->IsCompiled() == false)
			return false;
		if (m_vkPixelShader && m_vkPixelShader->IsCompiled() == false)
			return false;
		if (m_vkGeometryShader && m_vkGeometryShader->IsCompiled() == false)
			return false;
	}
	else
	{
		// if some shader stages are not compiled yet, compile them now
		if (m_vkVertexShader && m_vkVertexShader->IsCompiled() == false)
			m_vkVertexShader->PreponeCompilation(isRenderThread);
		if (m_vkPixelShader && m_vkPixelShader->IsCompiled() == false)
			m_vkPixelShader->PreponeCompilation(isRenderThread);
		if (m_vkGeometryShader && m_vkGeometryShader->IsCompiled() == false)
			m_vkGeometryShader->PreponeCompilation(isRenderThread);
	}

	if (shaderStages.empty())
	{
		if (!InitShaderStages(vkRenderer, m_vkVertexShader, m_vkPixelShader, m_vkGeometryShader))
			return true; // invalid shaders, cannot compile
	}

	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pDynamicState = &dynamicState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = m_pipeline_layout;
	pipelineInfo.renderPass = m_renderPassObj->m_renderPass;
	pipelineInfo.pDepthStencilState = &depthStencilState;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = nullptr;
	pipelineInfo.flags = 0;
	if (!forceCompile)
		pipelineInfo.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_EXT;

	VkPipelineCreationFeedbackCreateInfoEXT creationFeedbackInfo;
	VkPipelineCreationFeedbackEXT creationFeedback;
	std::vector<VkPipelineCreationFeedbackEXT> creationStageFeedback(0);
	if (vkRenderer->m_featureControl.deviceExtensions.pipeline_feedback)
	{
		creationFeedback = {};
		creationFeedback.flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT_EXT;

		creationStageFeedback.reserve(pipelineInfo.stageCount);
		for (uint32_t i = 0; i < pipelineInfo.stageCount; ++i)
			creationStageFeedback.data()[i] = { VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT_EXT, 0 };

		creationFeedbackInfo = {};
		creationFeedbackInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO_EXT;
		creationFeedbackInfo.pPipelineCreationFeedback = &creationFeedback;
		creationFeedbackInfo.pPipelineStageCreationFeedbacks = creationStageFeedback.data();
		creationFeedbackInfo.pipelineStageCreationFeedbackCount = pipelineInfo.stageCount;
		pipelineInfo.pNext = &creationFeedbackInfo;
	}

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkResult result;
	uint8 retryCount = 0;
	while (retryCount < 3)
	{
		std::shared_lock lock(vkRenderer->m_pipeline_cache_save_mutex);
		result = vkCreateGraphicsPipelines(vkRenderer->m_logicalDevice, vkRenderer->m_pipeline_cache, 1, &pipelineInfo, nullptr, &pipeline);
		lock.unlock();
		if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY)
			break;
		retryCount++;
	}

	if (result == VK_ERROR_PIPELINE_COMPILE_REQUIRED_EXT)
	{
		return false;
	}
	else if (result == VK_SUCCESS)
	{
		m_vkrObjPipeline->setPipeline(pipeline);
	}
	else
	{
		cemuLog_log(LogType::Force, "Failed to create graphics pipeline. Error {}", (sint32)result);
		cemu_assert_debug(false);
		return true; // true indicates that caller should no longer attempt to compile this pipeline again
	}
	vkRenderer->m_pipeline_cache_semaphore.notify();

	if (vkRenderer->m_featureControl.deviceExtensions.pipeline_feedback)
	{
		if (HAS_FLAG(creationFeedback.flags, VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT_EXT))
		{
			bool hasCacheHit = HAS_FLAG(creationFeedback.flags, VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT_EXT);
			if (!hasCacheHit)
			{
				if (showInOverlay)
				{
					if (isRenderThread)
						g_compiling_pipelines_syncTimeSum += creationFeedback.duration;
					else
						g_compiling_pipelines_async++;
					g_compiling_pipelines++;
				}
			}
		}
	}
	return true;
}

void PipelineCompiler::TrackAsCached(uint64 baseHash, uint64 pipelineStateHash)
{
	auto& pipelineCache = VulkanPipelineStableCache::GetInstance();
	if (pipelineCache.HasPipelineCached(baseHash, pipelineStateHash))
		return;
	pipelineCache.AddCurrentStateToCache(baseHash, pipelineStateHash);
}
