#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/config.h"
#include "utils/types.h"

class YOLOV10
{
public:
	YOLOV10(Config cfg);
	~YOLOV10();

	bool init();

	void Run(const cv::Mat img, std::vector<Detection>& res);

public:
	void preprocess(const cv::Mat img);

	void infer();

	void postprocess(std::vector<Detection>& res);

private:
	Config m_cfg;

	ov::InferRequest m_infer_request;
	float_t m_scale;

	cv::Mat m_input_img;

	ov::element::Type m_input_element_type;
	ov::Shape m_input_shape;
	ov::Shape m_output_shape;
};
