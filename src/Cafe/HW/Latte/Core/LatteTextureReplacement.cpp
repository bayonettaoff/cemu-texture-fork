#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Core/LatteTextureReplacement.h"
#include "Cafe/HW/Latte/Core/LatteTexture.h"
#include "Cafe/HW/Latte/Core/LatteTextureLoader.h" // also brings in tga.h (which has no include guard)
#include "Cafe/HW/Latte/Renderer/Renderer.h"
#include "Cafe/CafeSystem.h"

#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>

// PNG support goes through wxImage; the PNG handler is registered by
// wxInitAllImageHandlers() during app startup and wxImage file IO is safe off
// the main thread (the screenshot path already relies on this)
#include <wx/image.h>
#include <wx/log.h>

namespace LatteTextureReplacement
{
	static bool s_dumpEnabled = false;
	static bool s_indexValid = false;
	// lowercase filename (without extension) -> full path
	static std::unordered_map<std::string, fs::path> s_replacementIndex;
	// "{w}x{h}_fmt{fmt:04x}_slice{s}_mip{m}" (the part of an index name after the
	// hash) for every indexed entry, so a texture load can check in O(1) whether
	// ANY replacement could possibly exist for its geometry before paying for a
	// full-buffer hash of the pixel data
	static std::unordered_set<std::string> s_candidateSuffixes;

	// filesystem-invalid characters (Windows) replaced with '_'; games with a
	// ": " subtitle (e.g. some first-party titles) would otherwise break the path
	static std::string SanitizeFilename(std::string s)
	{
		for (char& c : s)
		{
			if (c == '<' || c == '>' || c == ':' || c == '"' || c == '/' || c == '\\' || c == '|' || c == '?' || c == '*' || (unsigned char)c < 0x20)
				c = '_';
		}
		while (!s.empty() && (s.back() == '.' || s.back() == ' '))
			s.pop_back();
		if (s.empty())
			s = "Unknown Game";
		return s;
	}

	static std::mutex s_titleDirMutex;
	static uint64 s_lastTitleId = 0;
	static std::string s_titleSubdir;

	// names (as produced by MakeName) already written to textureReplaceDump/ this
	// session, so DumpForReplacement can skip the directory-create + exists-check
	// + decode work for a texture it has already dumped instead of redoing it
	// every time the texture gets reloaded (which can be every frame)
	static std::mutex s_dumpedMutex;
	static std::unordered_set<std::string> s_dumpedNames;

	// raw-data fingerprint (FNV over the tiled input bytes, geometry folded in)
	// -> decoded pixel hash. OnTextureCreated has to know the content hash of
	// every candidate texture at creation time, which costs a full mip0
	// untile+decode+hash on the GPU thread; cutscenes stream the same textures
	// in and out repeatedly, so that cost was being re-paid constantly (the
	// reported cutscene stuttering, with the GPU itself only ~50% loaded).
	// Hashing the raw bytes instead is several times cheaper than decoding them
	// (raw BC data is 1/4 the decoded size and needs no per-texel work), and the
	// decoded hash is a pure function of raw bytes + geometry, so entries can
	// never go stale - the cache is only cleared to bound memory (title change /
	// size cap), never for correctness
	static std::mutex s_hashCacheMutex;
	static std::unordered_map<uint64, uint64> s_hashCache;

	// lightweight profiling of the replacement hot paths (all of which run on the
	// GPU thread): user reports cutscene stuttering with the GPU only ~50% busy,
	// so whatever is left is CPU time here or elsewhere - log a one-line summary
	// at most every 10s, and only when the window actually accumulated real work,
	// to identify the culprit from a live session without a profiler
	struct PerfBucket
	{
		std::atomic<uint64> us{ 0 };
		std::atomic<uint32> calls{ 0 };
	};
	static PerfBucket s_perfCreated, s_perfApply, s_perfUpscale, s_perfDiskRead;
	struct PerfScope
	{
		PerfBucket& bucket;
		std::chrono::steady_clock::time_point t0;
		PerfScope(PerfBucket& b) : bucket(b), t0(std::chrono::steady_clock::now()) {}
		~PerfScope()
		{
			bucket.us += (uint64)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
			bucket.calls++;
		}
	};
	static void PerfReport()
	{
		static std::atomic<uint64> s_lastLogMs{ 0 };
		const uint64 nowMs = (uint64)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		uint64 last = s_lastLogMs.load();
		if (nowMs - last < 10000)
			return;
		if (!s_lastLogMs.compare_exchange_strong(last, nowMs))
			return;
		const uint64 usCreated = s_perfCreated.us.exchange(0);
		const uint32 nCreated = s_perfCreated.calls.exchange(0);
		const uint64 usApply = s_perfApply.us.exchange(0);
		const uint32 nApply = s_perfApply.calls.exchange(0);
		const uint64 usUpscale = s_perfUpscale.us.exchange(0);
		const uint32 nUpscale = s_perfUpscale.calls.exchange(0);
		const uint64 usDisk = s_perfDiskRead.us.exchange(0);
		const uint32 nDisk = s_perfDiskRead.calls.exchange(0);
		if (usCreated + usApply + usUpscale < 20000)
			return; // under 20ms of total work in 10s: irrelevant, don't spam
		cemuLog_log(LogType::Force, "Texture replacement perf (last 10s, GPU thread): create-hash {}ms/{}x, apply {}ms/{}x, upscale-upload {}ms/{}x (of which disk reads {}ms/{}x)",
			usCreated / 1000, nCreated, usApply / 1000, nApply, usUpscale / 1000, nUpscale, usDisk / 1000, nDisk);
	}

	// global LRU cache of replacement file contents (pre-baked DDS mostly),
	// keyed by path. The file bytes used to be cached inside the per-texture
	// UpscaleEntry, but that entry dies with its LatteTexture object - and a
	// cutscene camera cut evicts/recreates the whole scene's textures, so every
	// cut re-read dozens of multi-MB DDS files from disk on the GPU thread
	// (reported as "stutter on every camera change"). shared_ptr so a file
	// still referenced by a live entry survives eviction. Budget defaults to
	// 1GB, override with CEMU_TEXCACHE_MB
	struct FileCacheEntry
	{
		std::shared_ptr<const std::vector<uint8>> data;
		uint64 lastUse = 0;
	};
	static std::mutex s_fileCacheMutex;
	static std::unordered_map<std::string, FileCacheEntry> s_fileCache;
	static size_t s_fileCacheBytes = 0;
	static uint64 s_fileCacheClock = 0;

	static size_t GetFileCacheBudget()
	{
		static const size_t s_budget = []() -> size_t {
			size_t mb = 1024;
			if (const char* v = std::getenv("CEMU_TEXCACHE_MB"))
				mb = (size_t)strtoul(v, nullptr, 10);
			return mb * 1024ull * 1024ull;
		}();
		return s_budget;
	}

	static std::shared_ptr<const std::vector<uint8>> GetReplacementFileCached(const fs::path& path)
	{
		const std::string key = _pathToUtf8(path);
		{
			std::lock_guard<std::mutex> lock(s_fileCacheMutex);
			const auto it = s_fileCache.find(key);
			if (it != s_fileCache.end())
			{
				it->second.lastUse = ++s_fileCacheClock;
				return it->second.data;
			}
		}
		// read outside the lock; only cache misses pay disk time (tracked separately)
		std::shared_ptr<std::vector<uint8>> data;
		{
			PerfScope perfScope(s_perfDiskRead);
			data = std::make_shared<std::vector<uint8>>();
			std::ifstream fileStream(path, std::ios::binary);
			if (fileStream)
				data->assign((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());
		}
		const size_t s_budget = GetFileCacheBudget();
		std::lock_guard<std::mutex> lock(s_fileCacheMutex);
		FileCacheEntry& entry = s_fileCache[key];
		if (entry.data) // raced with another insert of the same file
			s_fileCacheBytes -= entry.data->size();
		entry.data = data;
		entry.lastUse = ++s_fileCacheClock;
		s_fileCacheBytes += data->size();
		// evict least-recently-used files until under budget (never the one just
		// inserted - its lastUse is the newest). Linear scan is fine: at most
		// ~1000 entries and eviction only happens past the 1GB mark
		while (s_fileCacheBytes > s_budget && s_fileCache.size() > 1)
		{
			auto lru = s_fileCache.end();
			for (auto it = s_fileCache.begin(); it != s_fileCache.end(); ++it)
			{
				if (it->first != key && (lru == s_fileCache.end() || it->second.lastUse < lru->second.lastUse))
					lru = it;
			}
			if (lru == s_fileCache.end())
				break;
			s_fileCacheBytes -= lru->second.data->size();
			s_fileCache.erase(lru);
		}
		return data;
	}

	// "<game name> [<16-hex title id>]"; also invalidates the replacement index
	// on a title change so switching games without restarting Cemu re-scans the
	// new title's own folder instead of keeping the previous title's index
	static std::string GetTitleSubdir()
	{
		std::lock_guard<std::mutex> lock(s_titleDirMutex);
		const uint64 titleId = CafeSystem::GetForegroundTitleId();
		if (titleId != s_lastTitleId)
		{
			s_lastTitleId = titleId;
			s_titleSubdir = fmt::format("{} [{:016x}]", SanitizeFilename(CafeSystem::GetForegroundTitleName()), titleId);
			s_indexValid = false;
			{
				std::lock_guard<std::mutex> dumpLock(s_dumpedMutex);
				s_dumpedNames.clear();
			}
			std::lock_guard<std::mutex> hashLock(s_hashCacheMutex);
			s_hashCache.clear();
		}
		return s_titleSubdir;
	}

	static fs::path GetReplaceDir()
	{
		return fs::path("textureReplace") / GetTitleSubdir();
	}

	static fs::path GetDumpDir()
	{
		return fs::path("textureReplaceDump") / GetTitleSubdir();
	}

	bool IsDumpEnabled()
	{
		static bool s_envChecked = false;
		if (!s_envChecked)
		{
			s_envChecked = true;
			if (const char* v = std::getenv("CEMU_TEXDUMP"))
				s_dumpEnabled = (v[0] == '1');
		}
		return s_dumpEnabled;
	}

	void SetDumpEnabled(bool enabled)
	{
		IsDumpEnabled(); // consume env override first
		s_dumpEnabled = enabled;
	}

	void InvalidateIndex()
	{
		s_indexValid = false;
		s_replacementIndex.clear();
		s_candidateSuffixes.clear();
	}

	static std::atomic<bool> s_forgetRequested{false};

	void RequestForget()
	{
		s_forgetRequested = true;
	}

	bool ConsumeForgetRequest()
	{
		if (!s_forgetRequested.exchange(false))
			return false;
		InvalidateIndex();
		return true;
	}

	static void BuildIndexIfNeeded()
	{
		const fs::path replaceDir = GetReplaceDir(); // also clears s_indexValid on a title change
		if (s_indexValid)
			return;
		s_indexValid = true;
		s_replacementIndex.clear();
		s_candidateSuffixes.clear();
		std::error_code ec;
		if (!fs::exists(replaceDir, ec))
			return;
		for (const auto& entry : fs::directory_iterator(replaceDir, ec))
		{
			if (!entry.is_regular_file(ec))
				continue;
			std::string ext = entry.path().extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
			if (ext != ".dds" && ext != ".tga" && ext != ".png")
				continue;
			std::string stem = entry.path().stem().string();
			std::transform(stem.begin(), stem.end(), stem.begin(), ::tolower);
			// stem is "{hash:016x}_{w}x{h}_fmt{fmt:04x}_slice{s}_mip{m}" (see MakeName);
			// index the geometry suffix separately so it can be checked without the hash
			if (stem.size() > 17 && stem[16] == '_')
				s_candidateSuffixes.insert(stem.substr(17));
			s_replacementIndex.emplace(std::move(stem), entry.path());
		}
		cemuLog_log(LogType::Force, "Texture replacement: indexed {} file(s) in {}", s_replacementIndex.size(), _pathToUtf8(replaceDir));
	}

	static uint64 HashData(const uint8* data, uint32 size)
	{
		uint64 h = 0xcbf29ce484222325ull; // FNV-1a 64
		// hash in 8-byte strides for speed; tail bytes handled separately
		const uint32 wordCount = size / 8;
		const uint64* words = (const uint64*)data;
		for (uint32 i = 0; i < wordCount; i++)
		{
			h ^= words[i];
			h *= 0x100000001b3ull;
		}
		for (uint32 i = wordCount * 8; i < size; i++)
		{
			h ^= data[i];
			h *= 0x100000001b3ull;
		}
		return h;
	}

	// cheap raw-data fingerprint for the hash cache: full FNV of the first 4KB
	// plus 8 sampled bytes every 512 thereafter, size folded in. A first version
	// FNV'd the entire raw buffer, which for a streaming burst of multi-MB
	// textures still added up to visible per-load CPU on the GPU thread; textures
	// that differ at all differ pervasively (different art), so sparse sampling
	// keeps the collision risk negligible at ~1/20th the cost
	static uint64 FingerprintData(const uint8* data, uint32 size)
	{
		uint64 h = 0xcbf29ce484222325ull;
		const uint32 headSize = std::min<uint32>(size, 4096);
		const uint32 headWords = headSize / 8;
		const uint64* words = (const uint64*)data;
		for (uint32 i = 0; i < headWords; i++)
		{
			h ^= words[i];
			h *= 0x100000001b3ull;
		}
		for (uint32 offset = 4096; offset + 8 <= size; offset += 512)
		{
			h ^= *(const uint64*)(data + offset);
			h *= 0x100000001b3ull;
		}
		h ^= size;
		h *= 0x100000001b3ull;
		return h;
	}

	static std::string MakeName(uint64 hash, LatteTexture* tex, LatteTextureLoaderCtx* loader, uint32 sliceIndex, uint32 mipIndex)
	{
		return fmt::format("{:016x}_{}x{}_fmt{:04x}_slice{}_mip{}", hash, loader->width, loader->height, (uint32)tex->format, sliceIndex, mipIndex);
	}

	enum class PayloadKind
	{
		Unsupported,
		BC1,   // 8 bytes per 4x4 block
		BC3,   // 16 bytes per 4x4 block
		RGBA8, // 4 bytes per pixel
	};

	static PayloadKind GetPayloadKind(Latte::E_GX2SURFFMT format)
	{
		switch (format)
		{
		case Latte::E_GX2SURFFMT::BC1_UNORM:
		case Latte::E_GX2SURFFMT::BC1_SRGB:
			return PayloadKind::BC1;
		case Latte::E_GX2SURFFMT::BC3_UNORM:
		case Latte::E_GX2SURFFMT::BC3_SRGB:
			return PayloadKind::BC3;
		case Latte::E_GX2SURFFMT::R8_G8_B8_A8_UNORM:
		case Latte::E_GX2SURFFMT::R8_G8_B8_A8_SRGB:
			return PayloadKind::RGBA8;
		default:
			return PayloadKind::Unsupported;
		}
	}

	// minimal DDS reader for the payload kinds above
	struct DDSInfo
	{
		uint32 width;
		uint32 height;
		PayloadKind kind;
		size_t dataOffset;
	};

	static bool ParseDDSHeader(const std::vector<uint8>& file, DDSInfo& info)
	{
		if (file.size() < 128)
			return false;
		if (memcmp(file.data(), "DDS ", 4) != 0)
			return false;
		const uint32 headerSize = *(const uint32*)(file.data() + 4);
		if (headerSize != 124)
			return false;
		info.height = *(const uint32*)(file.data() + 12);
		info.width = *(const uint32*)(file.data() + 16);
		const uint8* pf = file.data() + 76; // DDS_PIXELFORMAT
		const uint32 pfFlags = *(const uint32*)(pf + 4);
		const uint32 fourCC = *(const uint32*)(pf + 8);
		info.dataOffset = 128;
		if (pfFlags & 0x4) // DDPF_FOURCC
		{
			if (fourCC == '1TXD') // "DXT1"
				info.kind = PayloadKind::BC1;
			else if (fourCC == '5TXD') // "DXT5"
				info.kind = PayloadKind::BC3;
			else
				return false; // DX10 header etc. not supported
		}
		else if (pfFlags & 0x40) // DDPF_RGB, uncompressed
		{
			const uint32 bitCount = *(const uint32*)(pf + 12);
			if (bitCount != 32)
				return false;
			info.kind = PayloadKind::RGBA8; // channel masks handled by caller
		}
		else
			return false;
		return true;
	}

	// swizzles a 32-bit uncompressed DDS payload into RGBA order using its channel masks
	static void CopyDDS32AsRGBA(const std::vector<uint8>& file, size_t dataOffset, uint32 pixelCount, uint8* out)
	{
		const uint8* pf = file.data() + 76;
		const uint32 rMask = *(const uint32*)(pf + 16);
		const uint32 gMask = *(const uint32*)(pf + 20);
		const uint32 bMask = *(const uint32*)(pf + 24);
		const uint32 aMask = *(const uint32*)(pf + 28);
		auto shiftOf = [](uint32 mask) -> uint32 {
			if (mask == 0)
				return 0;
			uint32 s = 0;
			while (((mask >> s) & 1) == 0)
				s++;
			return s;
		};
		const uint32 rs = shiftOf(rMask), gs = shiftOf(gMask), bs = shiftOf(bMask), as = shiftOf(aMask);
		const uint32* src = (const uint32*)(file.data() + dataOffset);
		for (uint32 i = 0; i < pixelCount; i++)
		{
			const uint32 v = src[i];
			out[i * 4 + 0] = rMask ? (uint8)((v & rMask) >> rs) : 0;
			out[i * 4 + 1] = gMask ? (uint8)((v & gMask) >> gs) : 0;
			out[i * 4 + 2] = bMask ? (uint8)((v & bMask) >> bs) : 0;
			out[i * 4 + 3] = aMask ? (uint8)((v & aMask) >> as) : 0xFF;
		}
	}

	// minimal TGA reader: uncompressed 32-bit, handles bottom-up and top-down
	static bool LoadTGAAsRGBA(const std::vector<uint8>& file, uint32 expectedWidth, uint32 expectedHeight, uint8* out)
	{
		if (file.size() < 18)
			return false;
		const uint8 idLength = file[0];
		const uint8 imageType = file[2];
		if (imageType != 2) // uncompressed true-color only
			return false;
		const uint32 width = file[12] | (file[13] << 8);
		const uint32 height = file[14] | (file[15] << 8);
		const uint8 bpp = file[16];
		const bool topDown = (file[17] & 0x20) != 0;
		if (bpp != 32 || width != expectedWidth || height != expectedHeight)
			return false;
		const size_t dataOffset = 18 + idLength;
		if (file.size() < dataOffset + (size_t)width * height * 4)
			return false;
		for (uint32 y = 0; y < height; y++)
		{
			const uint32 srcRow = topDown ? y : (height - 1 - y);
			const uint8* src = file.data() + dataOffset + (size_t)srcRow * width * 4;
			uint8* dst = out + (size_t)y * width * 4;
			for (uint32 x = 0; x < width; x++)
			{
				dst[x * 4 + 0] = src[x * 4 + 2]; // TGA stores BGRA
				dst[x * 4 + 1] = src[x * 4 + 1];
				dst[x * 4 + 2] = src[x * 4 + 0];
				dst[x * 4 + 3] = src[x * 4 + 3];
			}
		}
		return true;
	}

	// loads a PNG into a tightly packed RGBA8 buffer, returning its dimensions
	static bool LoadPNGAsRGBAAnySize(const fs::path& path, uint32& width, uint32& height, std::vector<uint8>& rgbaOut)
	{
		wxLogNull noLog; // no message boxes on malformed files
		wxImage img;
		if (!img.LoadFile(path.wstring(), wxBITMAP_TYPE_PNG))
			return false;
		width = (uint32)img.GetWidth();
		height = (uint32)img.GetHeight();
		const uint8* rgb = img.GetData();
		const uint8* alpha = img.HasAlpha() ? img.GetAlpha() : nullptr;
		const size_t pixelCount = (size_t)width * height;
		rgbaOut.resize(pixelCount * 4);
		for (size_t i = 0; i < pixelCount; i++)
		{
			rgbaOut[i * 4 + 0] = rgb[i * 3 + 0];
			rgbaOut[i * 4 + 1] = rgb[i * 3 + 1];
			rgbaOut[i * 4 + 2] = rgb[i * 3 + 2];
			rgbaOut[i * 4 + 3] = alpha ? alpha[i] : 0xFF;
		}
		return true;
	}

	// loads a PNG into a tightly packed RGBA8 buffer, enforcing the expected size
	static bool LoadPNGAsRGBA(const fs::path& path, uint32 expectedWidth, uint32 expectedHeight, std::vector<uint8>& rgbaOut)
	{
		uint32 width = 0, height = 0;
		if (!LoadPNGAsRGBAAnySize(path, width, height, rgbaOut))
			return false;
		return width == expectedWidth && height == expectedHeight;
	}

	// --- minimal BC1/BC3 block compressors (bounding-box endpoints, nearest-palette indices) ---

	static uint16 PackRGB565(const uint8* rgb)
	{
		return (uint16)(((rgb[0] >> 3) << 11) | ((rgb[1] >> 2) << 5) | (rgb[2] >> 3));
	}

	static void Unpack565(uint16 c, uint8* rgb)
	{
		rgb[0] = (uint8)(((c >> 11) & 0x1F) * 255 / 31);
		rgb[1] = (uint8)(((c >> 5) & 0x3F) * 255 / 63);
		rgb[2] = (uint8)((c & 0x1F) * 255 / 31);
	}

	// gathers a 4x4 block from the image, clamping coordinates at the edges
	static void FetchBlockRGBA(const uint8* rgba, uint32 width, uint32 height, uint32 bx, uint32 by, uint8 block[16][4])
	{
		for (uint32 py = 0; py < 4; py++)
		{
			const uint32 sy = std::min(by * 4 + py, height - 1);
			for (uint32 px = 0; px < 4; px++)
			{
				const uint32 sx = std::min(bx * 4 + px, width - 1);
				memcpy(block[py * 4 + px], rgba + ((size_t)sy * width + sx) * 4, 4);
			}
		}
	}

	// writes an 8-byte BC1 color block. allowPunchthrough enables BC1's 1-bit
	// transparency mode (must be false when used as the color half of BC3)
	static void CompressBlockBC1(const uint8 block[16][4], bool allowPunchthrough, uint8* out)
	{
		bool hasTransparency = false;
		if (allowPunchthrough)
		{
			for (uint32 i = 0; i < 16; i++)
				if (block[i][3] < 128)
					hasTransparency = true;
		}
		// bounding box endpoints over opaque pixels (all pixels if none are opaque)
		uint8 minC[3] = {255, 255, 255}, maxC[3] = {0, 0, 0};
		bool any = false;
		for (uint32 i = 0; i < 16; i++)
		{
			if (hasTransparency && block[i][3] < 128)
				continue;
			any = true;
			for (uint32 c = 0; c < 3; c++)
			{
				minC[c] = std::min(minC[c], block[i][c]);
				maxC[c] = std::max(maxC[c], block[i][c]);
			}
		}
		if (!any)
			memset(minC, 0, 3), memset(maxC, 0, 3);
		uint16 c0 = PackRGB565(maxC);
		uint16 c1 = PackRGB565(minC);
		// mode is selected by endpoint order: c0>c1 = 4 colors, c0<=c1 = 3 colors + transparent
		if (!hasTransparency && c0 < c1)
			std::swap(c0, c1);
		else if (hasTransparency && c0 > c1)
			std::swap(c0, c1);
		uint8 palette[4][3];
		Unpack565(c0, palette[0]);
		Unpack565(c1, palette[1]);
		const bool fourColorMode = c0 > c1;
		if (fourColorMode)
		{
			for (uint32 c = 0; c < 3; c++)
			{
				palette[2][c] = (uint8)((2 * palette[0][c] + palette[1][c]) / 3);
				palette[3][c] = (uint8)((palette[0][c] + 2 * palette[1][c]) / 3);
			}
		}
		else
		{
			for (uint32 c = 0; c < 3; c++)
			{
				palette[2][c] = (uint8)((palette[0][c] + palette[1][c]) / 2);
				palette[3][c] = 0;
			}
		}
		uint32 indices = 0;
		for (uint32 i = 0; i < 16; i++)
		{
			uint32 best;
			if (!fourColorMode && block[i][3] < 128)
				best = 3; // transparent
			else
			{
				const uint32 candidates = fourColorMode ? 4 : 3;
				uint32 bestDist = UINT32_MAX;
				best = 0;
				for (uint32 p = 0; p < candidates; p++)
				{
					uint32 dist = 0;
					for (uint32 c = 0; c < 3; c++)
					{
						const sint32 d = (sint32)block[i][c] - (sint32)palette[p][c];
						dist += (uint32)(d * d);
					}
					if (dist < bestDist)
						bestDist = dist, best = p;
				}
			}
			indices |= best << (i * 2);
		}
		*(uint16*)(out + 0) = c0;
		*(uint16*)(out + 2) = c1;
		*(uint32*)(out + 4) = indices;
	}

	// writes the 8-byte interpolated-alpha block of BC3
	static void CompressBlockBC3Alpha(const uint8 block[16][4], uint8* out)
	{
		uint8 aMin = 255, aMax = 0;
		for (uint32 i = 0; i < 16; i++)
		{
			aMin = std::min(aMin, block[i][3]);
			aMax = std::max(aMax, block[i][3]);
		}
		const uint8 a0 = aMax, a1 = aMin; // a0>a1 selects the 8-value mode
		uint8 palette[8];
		palette[0] = a0;
		palette[1] = a1;
		for (uint32 i = 1; i <= 6; i++)
			palette[i + 1] = (uint8)(((7 - i) * a0 + i * a1) / 7);
		uint64 bits = 0;
		for (uint32 i = 0; i < 16; i++)
		{
			uint32 best = 0;
			if (a0 != a1)
			{
				uint32 bestDist = UINT32_MAX;
				for (uint32 p = 0; p < 8; p++)
				{
					const uint32 d = (uint32)std::abs((sint32)block[i][3] - (sint32)palette[p]);
					if (d < bestDist)
						bestDist = d, best = p;
				}
			}
			bits |= (uint64)best << (i * 3);
		}
		out[0] = a0;
		out[1] = a1;
		for (uint32 i = 0; i < 6; i++)
			out[2 + i] = (uint8)(bits >> (i * 8));
	}

	// compresses a full RGBA image into BC1 or BC3 blocks; expects out to hold
	// ceil(w/4)*ceil(h/4) blocks of 8 (BC1) or 16 (BC3) bytes
	static void CompressRGBAToBC(const uint8* rgba, uint32 width, uint32 height, PayloadKind kind, uint8* out)
	{
		const uint32 blocksX = (width + 3) / 4;
		const uint32 blocksY = (height + 3) / 4;
		uint8 block[16][4];
		for (uint32 by = 0; by < blocksY; by++)
		{
			for (uint32 bx = 0; bx < blocksX; bx++)
			{
				FetchBlockRGBA(rgba, width, height, bx, by, block);
				if (kind == PayloadKind::BC1)
				{
					CompressBlockBC1(block, true, out);
					out += 8;
				}
				else // BC3
				{
					CompressBlockBC3Alpha(block, out);
					CompressBlockBC1(block, false, out + 8);
					out += 16;
				}
			}
		}
	}

	static uint32 GetLevelPayloadSize(uint32 width, uint32 height, PayloadKind kind)
	{
		if (kind == PayloadKind::RGBA8)
			return width * height * 4;
		const uint32 blocks = ((width + 3) / 4) * ((height + 3) / 4);
		return blocks * (kind == PayloadKind::BC1 ? 8 : 16);
	}

	// writes a legacy-header DDS (DXT1/DXT5 fourCC or 32-bit RGBA masks, no DX10
	// extension) whose payload is the concatenated mip chain
	static bool WriteDDSWithMips(const fs::path& path, uint32 width, uint32 height, PayloadKind kind, const std::vector<std::vector<uint8>>& mipPayloads)
	{
		uint8 header[128] = {};
		memcpy(header, "DDS ", 4);
		*(uint32*)(header + 4) = 124;
		const bool compressed = (kind != PayloadKind::RGBA8);
		uint32 flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000; // CAPS|HEIGHT|WIDTH|PIXELFORMAT|MIPMAPCOUNT
		flags |= compressed ? 0x80000 : 0x8; // LINEARSIZE : PITCH
		*(uint32*)(header + 8) = flags;
		*(uint32*)(header + 12) = height;
		*(uint32*)(header + 16) = width;
		*(uint32*)(header + 20) = compressed ? GetLevelPayloadSize(width, height, kind) : width * 4;
		*(uint32*)(header + 28) = (uint32)mipPayloads.size();
		uint8* pf = header + 76; // DDS_PIXELFORMAT
		*(uint32*)(pf + 0) = 32;
		if (compressed)
		{
			*(uint32*)(pf + 4) = 0x4; // DDPF_FOURCC
			memcpy(pf + 8, kind == PayloadKind::BC1 ? "DXT1" : "DXT5", 4);
		}
		else
		{
			*(uint32*)(pf + 4) = 0x41; // DDPF_RGB | DDPF_ALPHAPIXELS
			*(uint32*)(pf + 12) = 32;
			*(uint32*)(pf + 16) = 0x000000FF;
			*(uint32*)(pf + 20) = 0x0000FF00;
			*(uint32*)(pf + 24) = 0x00FF0000;
			*(uint32*)(pf + 28) = 0xFF000000;
		}
		*(uint32*)(header + 108) = 0x1000 | 0x8 | 0x400000; // DDSCAPS_TEXTURE|COMPLEX|MIPMAP
		std::ofstream f(path, std::ios::binary | std::ios::trunc);
		if (!f)
			return false;
		f.write((const char*)header, sizeof(header));
		for (const auto& level : mipPayloads)
			f.write((const char*)level.data(), level.size());
		return (bool)f;
	}

	// --- upscaled replacements (resolution overwrite driven by a larger PNG) ---

	struct UpscaleEntry
	{
		fs::path path;
		uint32 baseWidth;
		uint32 baseHeight;
		std::string name; // for logging
		std::vector<std::vector<uint8>> rgbaMips; // lazily built; level 0 = the PNG
		bool applyLogged = false;
		// DDS replacements upload their pre-baked mip payloads directly
		bool isDDS = false;
		std::shared_ptr<const std::vector<uint8>> ddsFile; // from the global file cache, fetched lazily
		size_t ddsDataOffset = 0;
		uint32 ddsMipCount = 0;
	};
	// textures live on the GPU thread but InvalidateIndex comes from the UI thread,
	// so keep the map guarded
	static std::mutex s_upscaleMutex;
	static std::unordered_map<LatteTexture*, UpscaleEntry> s_upscaleMap;

	// reads only the IHDR dimensions; cheap enough to run during texture creation
	static bool ReadPNGSize(const fs::path& path, uint32& width, uint32& height)
	{
		std::ifstream f(path, std::ios::binary);
		uint8 hdr[24];
		if (!f.read((char*)hdr, sizeof(hdr)))
			return false;
		static const uint8 sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
		if (memcmp(hdr, sig, 8) != 0 || memcmp(hdr + 12, "IHDR", 4) != 0)
			return false;
		width = ((uint32)hdr[16] << 24) | ((uint32)hdr[17] << 16) | ((uint32)hdr[18] << 8) | hdr[19];
		height = ((uint32)hdr[20] << 24) | ((uint32)hdr[21] << 16) | ((uint32)hdr[22] << 8) | hdr[23];
		return true;
	}

	// reads only the DDS header dimensions; cheap enough to run during texture creation
	static bool ReadDDSSize(const fs::path& path, uint32& width, uint32& height)
	{
		std::ifstream f(path, std::ios::binary);
		uint8 hdr[20];
		if (!f.read((char*)hdr, sizeof(hdr)))
			return false;
		if (memcmp(hdr, "DDS ", 4) != 0)
			return false;
		height = *(const uint32*)(hdr + 12);
		width = *(const uint32*)(hdr + 16);
		return true;
	}

	// 2x box downscale with edge clamping (handles odd source dimensions)
	static void DownscaleRGBA(const std::vector<uint8>& src, uint32 srcW, uint32 srcH, std::vector<uint8>& dst, uint32 dstW, uint32 dstH)
	{
		dst.resize((size_t)dstW * dstH * 4);
		for (uint32 y = 0; y < dstH; y++)
		{
			const uint32 sy0 = std::min(y * 2, srcH - 1);
			const uint32 sy1 = std::min(y * 2 + 1, srcH - 1);
			for (uint32 x = 0; x < dstW; x++)
			{
				const uint32 sx0 = std::min(x * 2, srcW - 1);
				const uint32 sx1 = std::min(x * 2 + 1, srcW - 1);
				const uint8* p00 = src.data() + ((size_t)sy0 * srcW + sx0) * 4;
				const uint8* p01 = src.data() + ((size_t)sy0 * srcW + sx1) * 4;
				const uint8* p10 = src.data() + ((size_t)sy1 * srcW + sx0) * 4;
				const uint8* p11 = src.data() + ((size_t)sy1 * srcW + sx1) * 4;
				uint8* out = dst.data() + ((size_t)y * dstW + x) * 4;
				for (uint32 c = 0; c < 4; c++)
					out[c] = (uint8)(((uint32)p00[c] + p01[c] + p10[c] + p11[c] + 2) / 4);
			}
		}
	}

	// --- boot-time preprocessing (filter + PNG -> mip-mapped DDS bake) ---

	// parses "<hash>_<W>x<H>_fmt<fmt>_slice<S>_mip<M>" (dimensions are the
	// ORIGINAL texture's; the file itself may be an upscale)
	static bool ParseReplacementStem(const std::string& stem, uint32& fmtRaw, uint32& outW, uint32& outH)
	{
		unsigned long long hash = 0;
		unsigned int w = 0, h = 0, f = 0, s = 0, m = 0;
		if (sscanf(stem.c_str(), "%16llx_%ux%u_fmt%x_slice%u_mip%u", &hash, &w, &h, &f, &s, &m) != 6)
			return false;
		fmtRaw = f;
		outW = w;
		outH = h;
		return true;
	}

	struct TexAnalysis
	{
		uint8 alphaMin;
		uint32 meanR;
		uint32 meanG;
		uint32 meanB;
	};

	// overwrites dst's alpha channel with src's, bilinearly resampled to the
	// destination size (used when an upscaler flattened the alpha of a PNG)
	static void MergeAlphaFromOriginal(std::vector<uint8>& dst, uint32 dstW, uint32 dstH, const std::vector<uint8>& src, uint32 srcW, uint32 srcH)
	{
		for (uint32 y = 0; y < dstH; y++)
		{
			float fy = ((float)y + 0.5f) * (float)srcH / (float)dstH - 0.5f;
			fy = std::clamp(fy, 0.0f, (float)(srcH - 1));
			const uint32 y0 = (uint32)fy;
			const uint32 y1 = std::min(y0 + 1, srcH - 1);
			const float wy = fy - (float)y0;
			for (uint32 x = 0; x < dstW; x++)
			{
				float fx = ((float)x + 0.5f) * (float)srcW / (float)dstW - 0.5f;
				fx = std::clamp(fx, 0.0f, (float)(srcW - 1));
				const uint32 x0 = (uint32)fx;
				const uint32 x1 = std::min(x0 + 1, srcW - 1);
				const float wx = fx - (float)x0;
				const float a00 = src[((size_t)y0 * srcW + x0) * 4 + 3];
				const float a01 = src[((size_t)y0 * srcW + x1) * 4 + 3];
				const float a10 = src[((size_t)y1 * srcW + x0) * 4 + 3];
				const float a11 = src[((size_t)y1 * srcW + x1) * 4 + 3];
				const float a = (a00 * (1.0f - wx) + a01 * wx) * (1.0f - wy) + (a10 * (1.0f - wx) + a11 * wx) * wy;
				dst[((size_t)y * dstW + x) * 4 + 3] = (uint8)std::clamp(a + 0.5f, 0.0f, 255.0f);
			}
		}
	}

	static void AnalyzeRGBA(const std::vector<uint8>& rgba, TexAnalysis& out)
	{
		uint64 sumR = 0, sumG = 0, sumB = 0;
		uint8 alphaMin = 255;
		const size_t pixelCount = rgba.size() / 4;
		for (size_t i = 0; i < pixelCount; i++)
		{
			sumR += rgba[i * 4 + 0];
			sumG += rgba[i * 4 + 1];
			sumB += rgba[i * 4 + 2];
			alphaMin = std::min(alphaMin, rgba[i * 4 + 3]);
		}
		const size_t n = std::max<size_t>(1, pixelCount);
		out.alphaMin = alphaMin;
		out.meanR = (uint32)(sumR / n);
		out.meanG = (uint32)(sumG / n);
		out.meanB = (uint32)(sumB / n);
	}

	void PreprocessReplaceFolder()
	{
		const fs::path replaceDir = GetReplaceDir();
		const fs::path dumpDir = GetDumpDir();
		std::error_code ec;
		fs::create_directories(replaceDir, ec); // so the per-game folder exists and is ready to drop files into, even before any dump happens
		ec.clear();
		if (!fs::exists(replaceDir, ec))
			return;
		std::vector<fs::path> pngFiles;
		for (const auto& entry : fs::directory_iterator(replaceDir, ec))
		{
			if (!entry.is_regular_file(ec))
				continue;
			std::string ext = entry.path().extension().string();
			std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
			if (ext == ".png")
				pngFiles.push_back(entry.path());
		}
		if (pngFiles.empty())
			return;
		const auto timeStart = now_cached();
		cemuLog_log(LogType::Force, "Texture replacement: preprocessing {} PNG file(s)...", pngFiles.size());
		std::atomic<uint32> nextIndex{0};
		std::atomic<uint32> numAlpha{0}, numConverted{0}, numSkipped{0};

		auto worker = [&]() {
			wxLogNull noLog;
			for (;;)
			{
				const uint32 i = nextIndex.fetch_add(1);
				if (i >= (uint32)pngFiles.size())
					return;
				const fs::path& path = pngFiles[i];
				std::string stem = path.stem().string();
				std::transform(stem.begin(), stem.end(), stem.begin(), ::tolower);
				uint32 fmtRaw = 0, origW = 0, origH = 0;
				if (!ParseReplacementStem(stem, fmtRaw, origW, origH))
				{
					numSkipped++;
					continue;
				}
				const PayloadKind kind = GetPayloadKind((Latte::E_GX2SURFFMT)fmtRaw);
				if (kind == PayloadKind::Unsupported)
				{
					numSkipped++;
					continue;
				}
				uint32 pngW = 0, pngH = 0;
				std::vector<uint8> rgba;
				if (!LoadPNGAsRGBAAnySize(path, pngW, pngH, rgba))
				{
					numSkipped++;
					continue;
				}
				// analyze the upscaled PNG and, when available, the dump original
				// (the original is authoritative for classification: upscalers may
				// shift colors and some flatten the alpha channel)
				TexAnalysis pngAnalysis;
				AnalyzeRGBA(rgba, pngAnalysis);
				TexAnalysis origAnalysis = pngAnalysis;
				std::vector<uint8> dumpRGBA;
				uint32 dumpW = 0, dumpH = 0;
				{
					const fs::path dumpPath = dumpDir / (stem + ".png");
					std::error_code ec2;
					if (fs::exists(dumpPath, ec2) && LoadPNGAsRGBAAnySize(dumpPath, dumpW, dumpH, dumpRGBA))
						AnalyzeRGBA(dumpRGBA, origAnalysis);
					else
						dumpRGBA.clear();
				}
				std::error_code mec;
				if (origAnalysis.alphaMin < 250 && pngAnalysis.alphaMin >= 250 && !dumpRGBA.empty())
				{
					// the original has transparency but the upscaler flattened it:
					// rebuild alpha from the original's alpha channel
					MergeAlphaFromOriginal(rgba, pngW, pngH, dumpRGBA, dumpW, dumpH);
					numAlpha++;
				}
				// bake the full mip chain into a DDS next to the PNG
				std::vector<std::vector<uint8>> mipPayloads;
				std::vector<uint8> current = std::move(rgba);
				uint32 w = pngW, h = pngH;
				for (;;)
				{
					std::vector<uint8> payload;
					if (kind == PayloadKind::RGBA8)
						payload = current;
					else
					{
						payload.resize(GetLevelPayloadSize(w, h, kind));
						CompressRGBAToBC(current.data(), w, h, kind, payload.data());
					}
					mipPayloads.push_back(std::move(payload));
					if (w == 1 && h == 1)
						break;
					const uint32 nw = std::max<uint32>(1, w >> 1);
					const uint32 nh = std::max<uint32>(1, h >> 1);
					std::vector<uint8> next;
					DownscaleRGBA(current, w, h, next, nw, nh);
					current = std::move(next);
					w = nw;
					h = nh;
				}
				fs::path ddsPath = path;
				ddsPath.replace_extension(".dds");
				if (WriteDDSWithMips(ddsPath, pngW, pngH, kind, mipPayloads))
				{
					fs::remove(path, mec);
					numConverted++;
				}
				else
				{
					fs::remove(ddsPath, mec); // don't leave a truncated file behind
					numSkipped++;
				}
			}
		};
		const uint32 threadCount = std::clamp<uint32>(std::thread::hardware_concurrency(), 1, 8);
		std::vector<std::thread> threads;
		for (uint32 t = 0; t < threadCount; t++)
			threads.emplace_back(worker);
		for (auto& t : threads)
			t.join();
		InvalidateIndex();
		const uint32 elapsedMs = (uint32)std::chrono::duration_cast<std::chrono::milliseconds>(now_cached() - timeStart).count();
		cemuLog_log(LogType::Force, "Texture replacement: preprocessing done in {}ms ({} baked to DDS, {} of those with alpha restored from dump, {} left as-is)",
					elapsedMs, numConverted.load(), numAlpha.load(), numSkipped.load());

		// warm the file cache off-thread: with a cold cache the first pass through
		// a cutscene paid first-touch disk reads on the GPU thread (measured 1.2s
		// of reads per 10s window in-game) - reading the pack up front on a
		// background thread makes even the first camera cut serve from RAM.
		// Detached and budget-capped; worst case it reads files that get evicted
		// again, which costs nothing on the GPU thread
		std::thread([replaceDir]() {
			const size_t stopAt = (size_t)((double)GetFileCacheBudget() * 0.9);
			std::error_code prefetchEc;
			for (const auto& entry : fs::directory_iterator(replaceDir, prefetchEc))
			{
				if (!entry.is_regular_file(prefetchEc))
					continue;
				std::string ext = entry.path().extension().string();
				std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
				if (ext != ".dds" && ext != ".tga")
					continue;
				{
					std::lock_guard<std::mutex> lock(s_fileCacheMutex);
					if (s_fileCacheBytes >= stopAt)
						break;
				}
				GetReplacementFileCached(entry.path());
			}
			std::lock_guard<std::mutex> lock(s_fileCacheMutex);
			cemuLog_log(LogType::Force, "Texture replacement: prefetched {} file(s), {} MB in cache", s_fileCache.size(), s_fileCacheBytes / (1024 * 1024));
		}).detach();
	}

	void OnTextureCreated(LatteTexture* tex)
	{
		if (tex->isDepth || tex->overwriteInfo.hasResolutionOverwrite || tex->overwriteInfo.hasFormatOverwrite)
			return;
		if (tex->dim != Latte::E_DIM::DIM_2D || tex->depth > 1)
			return;
		if (GetPayloadKind(tex->format) == PayloadKind::Unsupported)
			return;
		// the upscale upload path is only wired up for Vulkan (GL's loadSlice
		// asserts on resolution overwrites)
		if (!g_renderer || g_renderer->GetType() != RendererAPI::Vulkan)
			return;
		BuildIndexIfNeeded();
		if (s_replacementIndex.empty())
			return;
		// cheap O(1) pre-filter on geometry/format before paying for an early decode
		const std::string suffix = fmt::format("{}x{}_fmt{:04x}_slice0_mip0", tex->width, tex->height, (uint32)tex->format);
		if (!s_candidateSuffixes.count(suffix))
			return;
		PerfScope perfScope(s_perfCreated);
		// decode mip 0 the same way the loader will, to compute the content hash
		LatteTextureLoaderCtx ctx = {0};
		LatteTextureLoader_begin(&ctx, 0, 0, tex->physAddress, tex->physMipAddress, tex->format, tex->dim,
								 tex->width, tex->height, tex->depth, tex->mipLevels, tex->pitch, tex->tileMode, tex->swizzle);
		TextureDecoder* decoder = g_renderer->texture_chooseDecodedFormat(tex->format, tex->isDepth, tex->dim, tex->width, tex->height);
		if (!decoder)
			return;
		ctx.decodedTexelCountX = decoder->getTexelCountX(&ctx);
		ctx.decodedTexelCountY = decoder->getTexelCountY(&ctx);
		const sint32 imageSize = decoder->calculateImageSize(&ctx);
		if (imageSize <= 0)
			return;
		// fingerprint the raw tiled input first and only untile+decode+hash on a
		// cache miss: the game streaming the same texture back in (constant during
		// cutscenes) then costs one linear read of the raw bytes instead of the
		// full per-texel decode that was stalling the GPU thread
		uint64 hash = 0;
		bool haveHash = false;
		uint64 fingerprint = 0;
		bool haveFingerprint = false;
		const uint32 rawSize = ctx.computeAddrInfo.sliceBytes;
		if (ctx.inputData && rawSize > 0 && rawSize <= 0x8000000)
		{
			fingerprint = FingerprintData(ctx.inputData, rawSize);
			// fold the decode geometry in: identical bytes decoded under different
			// dimensions/format/pitch yield different pixel data. Same key scheme
			// as Apply() below (slice/mip are 0 here) so mip0 entries are shared
			fingerprint ^= ((uint64)ctx.width << 48) ^ ((uint64)ctx.height << 32) ^ ((uint64)ctx.pitch << 16) ^ (uint64)tex->format;
			haveFingerprint = true;
			std::lock_guard<std::mutex> lock(s_hashCacheMutex);
			const auto cacheIt = s_hashCache.find(fingerprint);
			if (cacheIt != s_hashCache.cend())
			{
				hash = cacheIt->second;
				haveHash = true;
			}
		}
		if (!haveHash)
		{
			std::vector<uint8> pixelData((size_t)imageSize);
			decoder->decode(&ctx, pixelData.data());
			hash = HashData(pixelData.data(), (uint32)imageSize);
			if (haveFingerprint)
			{
				std::lock_guard<std::mutex> lock(s_hashCacheMutex);
				if (s_hashCache.size() >= 65536) // ~1MB; blunt reset is fine, it refills fast
					s_hashCache.clear();
				s_hashCache.emplace(fingerprint, hash);
			}
		}
		const std::string name = MakeName(hash, tex, &ctx, 0, 0);
		const auto it = s_replacementIndex.find(name);
		if (it == s_replacementIndex.cend())
			return;
		std::string ext = it->second.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		uint32 repW = 0, repH = 0;
		bool isDDS = false;
		if (ext == ".png")
		{
			if (!ReadPNGSize(it->second, repW, repH))
				return;
		}
		else if (ext == ".dds")
		{
			if (!ReadDDSSize(it->second, repW, repH))
				return;
			isDDS = true;
		}
		else
			return; // TGA stays same-resolution
		if (repW <= (uint32)tex->width && repH <= (uint32)tex->height)
			return; // same resolution: the regular Apply() path handles it
		tex->overwriteInfo.width = repW;
		tex->overwriteInfo.height = repH;
		tex->overwriteInfo.depth = tex->depth;
		tex->overwriteInfo.hasResolutionOverwrite = true;
		UpscaleEntry entry;
		entry.path = it->second;
		entry.baseWidth = repW;
		entry.baseHeight = repH;
		entry.name = name;
		entry.isDDS = isDDS;
		std::unique_lock lock(s_upscaleMutex);
		s_upscaleMap[tex] = std::move(entry);
		cemuLog_log(LogType::TextureCache, "Texture replacement: {} upscaled {}x{} -> {}x{}{}", name, tex->width, tex->height, repW, repH, isDDS ? " (DDS)" : "");
	}

	void OnTextureDeleted(LatteTexture* tex)
	{
		std::unique_lock lock(s_upscaleMutex);
		s_upscaleMap.erase(tex);
	}

	bool LoadUpscaledSlice(LatteTexture* tex, uint32 sliceIndex, uint32 mipIndex)
	{
		std::unique_lock lock(s_upscaleMutex);
		const auto it = s_upscaleMap.find(tex);
		if (it == s_upscaleMap.end())
			return false;
		PerfScope perfScope(s_perfUpscale);
		UpscaleEntry& entry = it->second;
		if (sliceIndex != 0)
			return false; // registered textures are single-slice 2D
		if (entry.isDDS)
		{
			// pre-baked DDS: upload the stored mip payload directly, no decode/compress
			const PayloadKind kind = GetPayloadKind(tex->format);
			if (!entry.ddsFile)
			{
				entry.ddsFile = GetReplacementFileCached(entry.path);
				DDSInfo info;
				if (entry.ddsFile->size() < 128 || !ParseDDSHeader(*entry.ddsFile, info) || info.kind != kind ||
					info.width != entry.baseWidth || info.height != entry.baseHeight)
				{
					cemuLog_log(LogType::Force, "Texture replacement: {} upscale DDS unreadable or format/size mismatch, texture stays cleared", entry.name);
					s_upscaleMap.erase(it);
					return false;
				}
				entry.ddsDataOffset = info.dataOffset;
				entry.ddsMipCount = std::max<uint32>(1, *(const uint32*)(entry.ddsFile->data() + 28));
			}
			if (mipIndex >= entry.ddsMipCount)
			{
				cemuLog_log(LogType::Force, "Texture replacement: {} DDS has no mip {} ({} stored), texture stays cleared", entry.name, mipIndex, entry.ddsMipCount);
				return false;
			}
			const uint32 levelW = std::max<uint32>(1, entry.baseWidth >> mipIndex);
			const uint32 levelH = std::max<uint32>(1, entry.baseHeight >> mipIndex);
			size_t offset = entry.ddsDataOffset;
			for (uint32 lvl = 0; lvl < mipIndex; lvl++)
				offset += GetLevelPayloadSize(std::max<uint32>(1, entry.baseWidth >> lvl), std::max<uint32>(1, entry.baseHeight >> lvl), kind);
			const uint32 levelSize = GetLevelPayloadSize(levelW, levelH, kind);
			if (entry.ddsFile->size() < offset + levelSize)
			{
				cemuLog_log(LogType::Force, "Texture replacement: {} DDS payload truncated at mip {}", entry.name, mipIndex);
				return false;
			}
			if (kind == PayloadKind::RGBA8)
			{
				// honor the file's channel masks (third-party DDS may be BGRA)
				std::vector<uint8> rgbaLevel((size_t)levelW * levelH * 4);
				CopyDDS32AsRGBA(*entry.ddsFile, offset, levelW * levelH, rgbaLevel.data());
				g_renderer->texture_loadSlice(tex, levelW, levelH, 1, rgbaLevel.data(), sliceIndex, mipIndex, (uint32)rgbaLevel.size());
			}
			else
			{
				g_renderer->texture_loadSlice(tex, levelW, levelH, 1, (void*)(entry.ddsFile->data() + offset), sliceIndex, mipIndex, levelSize);
			}
			if (!entry.applyLogged)
			{
				entry.applyLogged = true;
				cemuLog_log(LogType::TextureCache, "Texture replacement: applied {} (upscaled to {}x{}, pre-baked DDS)", entry.name, entry.baseWidth, entry.baseHeight);
			}
			return true;
		}
		if (entry.rgbaMips.empty())
		{
			std::vector<uint8> rgba;
			if (!LoadPNGAsRGBA(entry.path, entry.baseWidth, entry.baseHeight, rgba))
			{
				cemuLog_log(LogType::Force, "Texture replacement: {} upscale failed to load PNG, texture stays cleared", entry.name);
				s_upscaleMap.erase(it);
				return false;
			}
			entry.rgbaMips.push_back(std::move(rgba));
		}
		// build missing mip levels by successive box downscale
		while ((uint32)entry.rgbaMips.size() <= mipIndex)
		{
			const uint32 lvl = (uint32)entry.rgbaMips.size();
			const uint32 srcW = std::max<uint32>(1, entry.baseWidth >> (lvl - 1));
			const uint32 srcH = std::max<uint32>(1, entry.baseHeight >> (lvl - 1));
			const uint32 dstW = std::max<uint32>(1, entry.baseWidth >> lvl);
			const uint32 dstH = std::max<uint32>(1, entry.baseHeight >> lvl);
			std::vector<uint8> level;
			DownscaleRGBA(entry.rgbaMips.back(), srcW, srcH, level, dstW, dstH);
			entry.rgbaMips.push_back(std::move(level));
		}
		const uint32 levelW = std::max<uint32>(1, entry.baseWidth >> mipIndex);
		const uint32 levelH = std::max<uint32>(1, entry.baseHeight >> mipIndex);
		const std::vector<uint8>& levelRGBA = entry.rgbaMips[mipIndex];
		const PayloadKind kind = GetPayloadKind(tex->format);
		if (kind == PayloadKind::RGBA8)
		{
			g_renderer->texture_loadSlice(tex, levelW, levelH, 1, (void*)levelRGBA.data(), sliceIndex, mipIndex, levelW * levelH * 4);
		}
		else
		{
			const uint32 blocks = ((levelW + 3) / 4) * ((levelH + 3) / 4);
			const uint32 bytesPerBlock = (kind == PayloadKind::BC1) ? 8 : 16;
			std::vector<uint8> bcData((size_t)blocks * bytesPerBlock);
			CompressRGBAToBC(levelRGBA.data(), levelW, levelH, kind, bcData.data());
			g_renderer->texture_loadSlice(tex, levelW, levelH, 1, bcData.data(), sliceIndex, mipIndex, (uint32)bcData.size());
		}
		if (!entry.applyLogged)
		{
			entry.applyLogged = true;
			cemuLog_log(LogType::TextureCache, "Texture replacement: applied {} (upscaled to {}x{})", entry.name, entry.baseWidth, entry.baseHeight);
		}
		return true;
	}

	bool Apply(LatteTexture* tex, LatteTextureLoaderCtx* loader, TextureDecoder* decoder,
			   uint32 sliceIndex, uint32 mipIndex, uint8* pixelData, uint32 imageSize)
	{
		PerfReport(); // ~2 atomic reads in the common case; Apply runs on every texture load, ideal report point
		if (tex->isDepth || imageSize == 0)
			return false;
		const PayloadKind kind = GetPayloadKind(tex->format);
		if (kind == PayloadKind::Unsupported)
			return false;
		BuildIndexIfNeeded();
		if (s_replacementIndex.empty())
			return false;

		// cheap O(1) pre-filter on geometry/format: skip hashing the whole decoded
		// buffer (the actual cost here, paid by every texture load in the game)
		// unless a replacement could possibly exist for this exact geometry
		const std::string suffix = fmt::format("{}x{}_fmt{:04x}_slice{}_mip{}", loader->width, loader->height, (uint32)tex->format, sliceIndex, mipIndex);
		if (!s_candidateSuffixes.count(suffix))
			return false;
		PerfScope perfScope(s_perfApply);

		// same raw-fingerprint cache OnTextureCreated uses: the decode was already
		// paid by the loader here, but hashing the decoded buffer (4x the raw BC
		// size) on every reload of the same content is still real per-load cost
		uint64 hash = 0;
		bool haveHash = false;
		uint64 fingerprint = 0;
		bool haveFingerprint = false;
		const uint32 rawSize = loader->computeAddrInfo.sliceBytes;
		if (loader->inputData && rawSize > 0 && rawSize <= 0x8000000)
		{
			fingerprint = FingerprintData(loader->inputData, rawSize);
			fingerprint ^= ((uint64)loader->width << 48) ^ ((uint64)loader->height << 32) ^ ((uint64)loader->pitch << 16) ^ (uint64)tex->format;
			fingerprint ^= ((uint64)sliceIndex << 12) ^ ((uint64)mipIndex << 8);
			haveFingerprint = true;
			std::lock_guard<std::mutex> lock(s_hashCacheMutex);
			const auto cacheIt = s_hashCache.find(fingerprint);
			if (cacheIt != s_hashCache.cend())
			{
				hash = cacheIt->second;
				haveHash = true;
			}
		}
		if (!haveHash)
		{
			hash = HashData(pixelData, imageSize);
			if (haveFingerprint)
			{
				std::lock_guard<std::mutex> lock(s_hashCacheMutex);
				if (s_hashCache.size() >= 65536)
					s_hashCache.clear();
				s_hashCache.emplace(fingerprint, hash);
			}
		}
		const std::string name = MakeName(hash, tex, loader, sliceIndex, mipIndex);
		const auto it = s_replacementIndex.find(name);
		if (it == s_replacementIndex.cend())
			return false;

		std::string ext = it->second.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

		// DDS/TGA go through the global file cache so re-streaming the same
		// texture (every cutscene camera cut) doesn't re-read the file from disk
		static const std::vector<uint8> s_noFile;
		std::shared_ptr<const std::vector<uint8>> filePtr;
		if (ext != ".png") // wxImage reads PNGs straight from the path
		{
			filePtr = GetReplacementFileCached(it->second);
			if (!filePtr || filePtr->empty())
				return false;
		}
		const std::vector<uint8>& file = filePtr ? *filePtr : s_noFile;

		if (ext == ".dds")
		{
			DDSInfo dds;
			if (!ParseDDSHeader(file, dds) || dds.kind != kind ||
				dds.width != (uint32)loader->width || dds.height != (uint32)loader->height)
			{
				cemuLog_log(LogType::Force, "Texture replacement: {} rejected (format/resolution mismatch)", name);
				return false;
			}
			if (kind == PayloadKind::RGBA8)
			{
				const uint32 pixelCount = imageSize / 4;
				if (file.size() < dds.dataOffset + (size_t)pixelCount * 4)
					return false;
				CopyDDS32AsRGBA(file, dds.dataOffset, pixelCount, pixelData);
			}
			else
			{
				const size_t payload = std::min((size_t)imageSize, file.size() - dds.dataOffset);
				memcpy(pixelData, file.data() + dds.dataOffset, payload);
			}
		}
		else if (ext == ".png")
		{
			std::vector<uint8> rgba;
			if (!LoadPNGAsRGBA(it->second, loader->width, loader->height, rgba))
			{
				cemuLog_log(LogType::Force, "Texture replacement: {} rejected (PNG unreadable or not at the original resolution)", name);
				return false;
			}
			if (kind == PayloadKind::RGBA8)
			{
				memcpy(pixelData, rgba.data(), std::min((size_t)imageSize, rgba.size()));
			}
			else // BC1/BC3: compress on the fly
			{
				const uint32 blocks = ((loader->width + 3) / 4) * ((loader->height + 3) / 4);
				const uint32 expectedSize = blocks * (kind == PayloadKind::BC1 ? 8 : 16);
				if (expectedSize > imageSize)
				{
					cemuLog_log(LogType::Force, "Texture replacement: {} rejected (unexpected payload size {} vs {})", name, expectedSize, imageSize);
					return false;
				}
				CompressRGBAToBC(rgba.data(), loader->width, loader->height, kind, pixelData);
			}
		}
		else // .tga
		{
			if (kind != PayloadKind::RGBA8)
			{
				cemuLog_log(LogType::Force, "Texture replacement: {} is TGA but the texture is block-compressed; use a PNG or a DDS (DXT1/DXT5)", name);
				return false;
			}
			if (!LoadTGAAsRGBA(file, loader->width, loader->height, pixelData))
			{
				cemuLog_log(LogType::Force, "Texture replacement: {} rejected (TGA must be uncompressed 32-bit at the original resolution)", name);
				return false;
			}
		}
		cemuLog_log(LogType::TextureCache, "Texture replacement: applied {}{}", name, ext);
		return true;
	}

	void DumpForReplacement(LatteTexture* tex, LatteTextureLoaderCtx* loader, TextureDecoder* decoder,
							uint32 sliceIndex, uint32 mipIndex, const uint8* pixelData, uint32 imageSize)
	{
		if (tex->isDepth || imageSize == 0)
			return;
		if (mipIndex != 0)
			return; // replacements are authored from mip 0 only
		if (GetPayloadKind(tex->format) == PayloadKind::Unsupported)
			return;
		// same fingerprint->hash cache as Apply/OnTextureCreated, so an enabled
		// dump toggle doesn't hash the full decoded buffer on every reload either
		uint64 hash = 0;
		bool haveHash = false;
		uint64 fingerprint = 0;
		bool haveFingerprint = false;
		const uint32 rawSize = loader->computeAddrInfo.sliceBytes;
		if (loader->inputData && rawSize > 0 && rawSize <= 0x8000000)
		{
			fingerprint = FingerprintData(loader->inputData, rawSize);
			fingerprint ^= ((uint64)loader->width << 48) ^ ((uint64)loader->height << 32) ^ ((uint64)loader->pitch << 16) ^ (uint64)tex->format;
			fingerprint ^= ((uint64)sliceIndex << 12) ^ ((uint64)mipIndex << 8);
			haveFingerprint = true;
			std::lock_guard<std::mutex> lock(s_hashCacheMutex);
			const auto cacheIt = s_hashCache.find(fingerprint);
			if (cacheIt != s_hashCache.cend())
			{
				hash = cacheIt->second;
				haveHash = true;
			}
		}
		if (!haveHash)
		{
			hash = HashData(pixelData, imageSize);
			if (haveFingerprint)
			{
				std::lock_guard<std::mutex> lock(s_hashCacheMutex);
				if (s_hashCache.size() >= 65536)
					s_hashCache.clear();
				s_hashCache.emplace(fingerprint, hash);
			}
		}
		const std::string name = MakeName(hash, tex, loader, sliceIndex, mipIndex);

		{
			std::lock_guard<std::mutex> lock(s_dumpedMutex);
			if (s_dumpedNames.count(name))
				return; // already dumped this session, skip the disk check + decode entirely
		}

		std::error_code ec;
		const fs::path dumpDir = GetDumpDir();
		const fs::path outPath = dumpDir / (name + ".png");
		if (fs::exists(outPath, ec))
		{
			std::lock_guard<std::mutex> lock(s_dumpedMutex);
			s_dumpedNames.insert(name);
			return; // identical content was already dumped in a previous run
		}
		fs::create_directories(dumpDir, ec);

		// decode to RGBA via the per-pixel decoder (same approach as Cemu's stock dump)
		std::vector<uint8> rgba((size_t)loader->width * loader->height * 4);
		for (sint32 y = 0; y < loader->height; y++)
		{
			uint8* pixelOutput = rgba.data() + ((size_t)y * loader->width) * 4;
			for (sint32 x = 0; x < loader->width; x++)
			{
				uint8* blockData = LatteTextureLoader_GetInput(loader, x, y);
				decoder->decodePixelToRGBA(blockData, pixelOutput, x % loader->stepX, y % loader->stepY);
				pixelOutput += 4;
			}
		}
		// split interleaved RGBA into the separate RGB + alpha planes wxImage expects
		const size_t pixelCount = (size_t)loader->width * loader->height;
		wxImage img(loader->width, loader->height);
		img.SetAlpha();
		uint8* rgbOut = img.GetData();
		uint8* alphaOut = img.GetAlpha();
		for (size_t i = 0; i < pixelCount; i++)
		{
			rgbOut[i * 3 + 0] = rgba[i * 4 + 0];
			rgbOut[i * 3 + 1] = rgba[i * 4 + 1];
			rgbOut[i * 3 + 2] = rgba[i * 4 + 2];
			alphaOut[i] = rgba[i * 4 + 3];
		}
		wxLogNull noLog;
		img.SaveFile(outPath.wstring(), wxBITMAP_TYPE_PNG);

		// skip trivially small dumps (flat-color/near-empty textures aren't
		// worth authoring replacements for and just clutter the dump folder)
		constexpr uintmax_t kMinDumpSize = 16 * 1024;
		std::error_code sizeEc;
		const uintmax_t fileSize = fs::file_size(outPath, sizeEc);
		if (!sizeEc && fileSize <= kMinDumpSize)
			fs::remove(outPath, ec);

		std::lock_guard<std::mutex> lock(s_dumpedMutex);
		s_dumpedNames.insert(name);
	}
}
