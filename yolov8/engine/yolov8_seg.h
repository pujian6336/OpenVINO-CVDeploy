#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/config.h"
#include "utils/types.h"

class YOLOV8_SEG
{
public:
	YOLOV8_SEG(Config cfg);
	~YOLOV8_SEG();

	bool init();

	void Run(const cv::Mat img, std::vector<Detection>& res, std::vector<cv::Mat>& masks);

public:
	void preprocess(const cv::Mat img);

	void infer();

	void postprocess(std::vector<Detection>& res, std::vector<cv::Mat>& masks);
	void postprocess(std::vector<Detection>& res, cv::Mat& mask);

private:
	Config m_cfg;

	ov::InferRequest m_infer_request;
	float_t* m_input_data;
	std::vector<cv::Mat> m_imgs;
	float_t m_scale;

	ov::element::Type m_input_element_type;
	ov::Shape m_input_shape;
	ov::Shape m_output_det_shape;
	ov::Shape m_output_proto_shape;

};
