#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/config.h"
#include "utils/types.h"

class YOLOV8_OBB
{
public:
	YOLOV8_OBB(Config cfg);
	~YOLOV8_OBB();

	bool init();

	void Run(const cv::Mat img, std::vector<OBBResult>& res);

public:
	void preprocess(const cv::Mat img);

	void infer();

	void postprocess(std::vector<OBBResult>& res);

private:
	Config m_cfg;

	ov::InferRequest m_infer_request;
	float_t* m_input_data;
	float_t m_scale;

	ov::element::Type m_input_element_type;
	ov::Shape m_input_shape;
	ov::Shape m_output_shape;
};
