#pragma once
#include <opencv2/opencv.hpp>
#include "utils/types.h"


void box2OriginalSize(std::vector<Detection>& res, float_t scale);

void box2OriginalSize(std::vector<Detection>& res, float_t scale, int dw, int dh);

void thresholdFilter(float_t *data, std::vector<Detection>& res, float_t scoreThres);



