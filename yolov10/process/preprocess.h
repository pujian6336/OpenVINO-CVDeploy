#pragma once

#include <opencv2/opencv.hpp>

void imgResizePad(const cv::Mat img, cv::Mat &res, int32_t width, int32_t height, float_t& scale);

void imgResizePad(const cv::Mat img, cv::Mat& res, int32_t width, int32_t height, float_t& scale, int& dw, int& dh);