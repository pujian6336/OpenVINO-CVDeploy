#pragma once
#include <string>

struct Config
{
	std::string model_path;

	float iou_threshold = 0.45f;
	float conf_threshold = 0.25f;

	int batch_size = 1;
	int input_width = 640;
	int input_height = 640;

	int topK{ 300 };
};