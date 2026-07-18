#include "Cafe/HW/Latte/Renderer/RendererShader.h"
#include "Cafe/GameProfile/GameProfile.h"

// bump whenever a decompiler change alters the GLSL/SPIR-V this fork emits for the
// same GX2 shader bytecode (baseHash/auxHash only hash the GAME's shader, not our
// own decompiler logic) - otherwise old precompiled SPIR-V blobs keep getting served
// from disk and silently paired with freshly-compiled shaders that assume the new
// output, e.g. mismatched vertex/fragment interfaces (VUID-RuntimeSpirv-OpEntryPoint-08743)
// or wrong render-target numeric type. Root-caused 2026-07-13 via Mario Kart 8: its
// precompiled cache predates the uf_taaJitter uniform (clipSpaceJitter, default on)
// and the ATTR_LAYOUT descriptor-set fix, both of which change what every vertex/
// fragment shader in every game emits, and neither was reflected here until now.
constexpr uint32 kDecompilerOutputVersion = 1;

// generate a Cemu version and setting dependent id
uint32 RendererShader::GeneratePrecompiledCacheId()
{
	uint32 v = 0;
	const char* s = EMULATOR_VERSION_SUFFIX;
	while (*s)
	{
		v = std::rotl<uint32>(v, 7);
		v += (uint32)(*s);
		s++;
	}
	v += (EMULATOR_VERSION_MAJOR * 1000000u);
	v += (EMULATOR_VERSION_MINOR * 10000u);
	v += (EMULATOR_VERSION_PATCH * 100u);

	// settings that can influence shaders
	v += (uint32)g_current_game_profile->GetAccurateShaderMul() * 133;
	v += kDecompilerOutputVersion * 7919u;

	return v;
}

void RendererShader::GenerateShaderPrecompiledCacheFilename(RendererShader::ShaderType type, uint64 baseHash, uint64 auxHash, uint64& h1, uint64& h2)
{
	h1 = baseHash;
	h2 = auxHash;

	if (type == RendererShader::ShaderType::kVertex)
		h2 += 0xA16374cull;
	else if (type == RendererShader::ShaderType::kFragment)
		h2 += 0x8752deull;
	else if (type == RendererShader::ShaderType::kGeometry)
		h2 += 0x65a035ull;
}