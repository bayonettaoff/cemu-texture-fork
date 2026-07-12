#pragma once

// File-based texture replacement (Dolphin-style).
// Textures are identified by an FNV-1a 64 hash of their decoded host data, which
// is stable across sessions (unlike physical addresses). Replacement files live in
// "textureReplace/" next to the Cemu executable:
//   <hash>_<W>x<H>_fmt<fmt>_slice<S>_mip<M>.png  (any supported format; BC1/BC3 are compressed on load)
//   <hash>_<W>x<H>_fmt<fmt>_slice<S>_mip<M>.dds  (BC1="DXT1"/BC3="DXT5" or uncompressed 32-bit)
//   <hash>_<W>x<H>_fmt<fmt>_slice<S>_mip<M>.tga  (uncompressed 32-bit, for RGBA8 textures)
// Dump mode writes decoded RGBA copies of mip 0 with exactly these names (as
// .png) into "textureReplaceDump/", so authoring a replacement is: dump, edit
// the PNG, drop it into textureReplace/. Mip levels above 0 keep the game's
// original data (the PNG-authored replacement only covers mip 0).
//
// Upscaling (v2, Vulkan only): a PNG larger than the original (any size, e.g.
// 256x256 -> 1024x1024) triggers a resolution overwrite on the host texture,
// like graphic pack texture rules do. The decision is made when the LatteTexture
// is constructed (the host image size is fixed there), which requires decoding
// mip 0 early to compute the content hash. All mip levels are then generated
// from the PNG by box-downscaling (and BC1/BC3-compressed when needed), since
// the game's own data no longer matches the allocated size. Limited to 2D
// non-array textures. Caveat: if the game later rewrites the texture with
// different content, the replacement (matched against the original content)
// stays on screen until the texture is evicted.

class LatteTexture;
class TextureDecoder;
struct LatteTextureLoaderCtx;

namespace LatteTextureReplacement
{
	// overwrites pixelData with the replacement file's payload when one exists.
	// Called after the decoder filled pixelData (hash is computed from that data).
	// Returns true if a replacement was applied.
	bool Apply(LatteTexture* tex, LatteTextureLoaderCtx* loader, TextureDecoder* decoder,
			   uint32 sliceIndex, uint32 mipIndex, uint8* pixelData, uint32 imageSize);

	// writes the decoded RGBA image with the stable replacement name (dump mode)
	void DumpForReplacement(LatteTexture* tex, LatteTextureLoaderCtx* loader, TextureDecoder* decoder,
							uint32 sliceIndex, uint32 mipIndex, const uint8* pixelData, uint32 imageSize);

	// called at the end of the LatteTexture constructor, before the host image is
	// created: sets a resolution overwrite when a larger PNG replacement exists
	void OnTextureCreated(LatteTexture* tex);
	void OnTextureDeleted(LatteTexture* tex);

	// uploads the upscaled replacement for one slice/mip of a texture registered
	// by OnTextureCreated; returns false if the texture has no upscale pending
	bool LoadUpscaledSlice(LatteTexture* tex, uint32 sliceIndex, uint32 mipIndex);

	bool IsDumpEnabled();
	void SetDumpEnabled(bool enabled);

	// forget the cached folder index; the next lookup rescans textureReplace/
	void InvalidateIndex();

	// UI thread: ask the GPU thread to drop every live texture (so they get
	// recreated from scratch against a freshly-rescanned folder) at the next
	// frame boundary. Needed because upscaled replacements are baked into the
	// host texture at creation time and otherwise stick around, unaffected by
	// deleting the file or by InvalidateIndex(), until the game itself evicts
	// that texture.
	void RequestForget();
	// GPU thread: returns true (once) if RequestForget() was called since the
	// last check, and rescans the folder as part of consuming the request
	bool ConsumeForgetRequest();

	// runs once per game boot (from the shader cache loading phase, before any
	// texture loads): bakes PNG replacements into mip-mapped DDS files so
	// gameplay never pays for PNG decoding or BC compression; restores alpha
	// from the dump original when the upscaler flattened it
	void PreprocessReplaceFolder();
}
