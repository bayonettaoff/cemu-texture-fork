#include "Common/precompiled.h"
#include "Cafe/HW/Latte/Core/LatteGeometryAmp.h"

namespace LatteGeometryAmp
{
	static Config s_config;

	Config& GetConfig()
	{
		static bool s_envChecked = false;
		if (!s_envChecked)
		{
			s_envChecked = true;
			if (const char* v = std::getenv("CEMU_GEOAMP"))
				s_config.enabled = (v[0] == '1');
			if (const char* v = std::getenv("CEMU_GEOAMP_FACTOR"))
				s_config.factor = (float)std::atof(v);
			if (const char* v = std::getenv("CEMU_GEOAMP_TOLERANCE"))
				s_config.normalDetectTolerance = (float)std::atof(v);
		}
		return s_config;
	}
}
