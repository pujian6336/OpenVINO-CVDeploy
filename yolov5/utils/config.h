#pragma once
#include <string>

struct Config
{
	std::string model_path;

	float iou_threshold = 0.45f;
	float conf_threshold = 0.25f;

	int topK{ 300 };
	std::vector<float_t> anchors;
};