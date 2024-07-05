#include "preprocess.h"
#include "utils/utils.h"

void imgResizePad(const cv::Mat img, cv::Mat& res, int32_t width, int32_t height, float_t& scale)
{
	scale = std::min(static_cast<float>(width) / img.cols, static_cast<float>(height) / img.rows);

	cv::Mat resized;
	cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

	res = cv::Mat(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(res(cv::Rect(0, 0, resized.cols, resized.rows)));
}

void imgResizePad(const cv::Mat img, cv::Mat& res, int32_t width, int32_t height, float_t& scale, int& dw, int& dh)
{
	scale = std::min(static_cast<float>(width) / img.cols, static_cast<float>(height) / img.rows);

	cv::Mat resized;
	cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

	dw = (width - resized.cols) / 2;
	dh = (height - resized.rows) / 2;

	res = cv::Mat(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(res(cv::Rect(dw, dh, resized.cols, resized.rows)));
}
