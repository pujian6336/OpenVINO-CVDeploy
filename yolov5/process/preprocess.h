#pragma once

#include <opencv2/opencv.hpp>

void img2blob(const cv::Mat img, cv::Mat &blob, int32_t width, int32_t height, float_t& scale);

void img2blob(const cv::Mat img, float_t *blob, int32_t width, int32_t height, float_t& scale);

void img2blob(const cv::Mat img, float_t* blob, int32_t width, int32_t height, float_t& scale, int& dw, int& dh);
