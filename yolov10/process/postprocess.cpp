#include "postprocess.h"
#include <cmath>

void box2OriginalSize(std::vector<Detection>& res, float_t scale)
{
	for (size_t i = 0; i < res.size(); i++)
	{
		res[i].bbox.left = res[i].bbox.left / scale;
		res[i].bbox.top = res[i].bbox.top / scale;
		res[i].bbox.right = res[i].bbox.right / scale;
		res[i].bbox.bottom = res[i].bbox.bottom / scale;
	}
}

void box2OriginalSize(std::vector<Detection>& res, float_t scale, int dw, int dh)
{
	for (size_t i = 0; i < res.size(); i++)
	{
		res[i].bbox.left = (res[i].bbox.left - dw) / scale;
		res[i].bbox.top = (res[i].bbox.top - dh) / scale;
		res[i].bbox.right = (res[i].bbox.right - dw) / scale;
		res[i].bbox.bottom = (res[i].bbox.bottom - dh) / scale;
	}
}

void thresholdFilter(float_t* data, std::vector<Detection>& res, float_t scoreThres)
{
	for (int32_t i = 0; i < 300; i++)
	{
		const float_t* ptr = data + i * 6;
		if (ptr[4] < scoreThres) break;

		res.emplace_back(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5]);
	}
}




