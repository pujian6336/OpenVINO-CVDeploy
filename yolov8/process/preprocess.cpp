#include "preprocess.h"

void img2blob(const cv::Mat img, cv::Mat &blob, int32_t width, int32_t height, float_t& scale)
{
	scale = std::min(static_cast<float>(width) / img.cols, static_cast<float>(height) / img.rows);

	cv::Mat resized;
	cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

	cv::Mat res = cv::Mat::zeros(height, width, CV_8UC3);
	resized.copyTo(res(cv::Rect(0, 0, resized.cols, resized.rows)));

	blob = cv::dnn::blobFromImage(res, 1.0 / 255.0, cv::Size(width, height), cv::Scalar(), true);
}

void img2blob(const cv::Mat img, float_t* blob, int32_t width, int32_t height, float_t& scale)
{
	scale = std::min(static_cast<float>(width) / img.cols, static_cast<float>(height) / img.rows);

	cv::Mat resized;
	cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

	cv::Mat res(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(res(cv::Rect(0, 0, resized.cols, resized.rows)));

	for (size_t h = 0; h < height; h++) {
		for (size_t w = 0; w < width; w++) {
			const cv::Vec3b& pixel = res.at<cv::Vec3b>(h, w);
			size_t r_idx = h * width + w;
			size_t g_idx = r_idx + height * width;
			size_t b_idx = r_idx + height * width * 2;
			blob[r_idx] = static_cast<float_t>(pixel[2]) / 255.0f;// R
			blob[g_idx] = static_cast<float_t>(pixel[1]) / 255.0f; // G
			blob[b_idx] = static_cast<float_t>(pixel[0]) / 255.0f; // B
		}
	}
}

void img2blob(const cv::Mat img, float_t* blob, int32_t width, int32_t height, float_t& scale, int& dw, int& dh)
{
	scale = std::min(static_cast<float>(width) / img.cols, static_cast<float>(height) / img.rows);

	cv::Mat resized;
	cv::resize(img, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

	dw = (width - resized.cols) / 2;
	dh = (height - resized.rows) / 2;

	cv::Mat res(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(res(cv::Rect(dw, dh, resized.cols, resized.rows)));

	for (size_t h = 0; h < height; h++) {
		for (size_t w = 0; w < width; w++) {
			const cv::Vec3b& pixel = res.at<cv::Vec3b>(h, w);
			size_t r_idx = h * width + w;
			size_t g_idx = r_idx + height * width;
			size_t b_idx = r_idx + height * width * 2;
			blob[r_idx] = static_cast<float_t>(pixel[2]) / 255.0; // R
			blob[g_idx] = static_cast<float_t>(pixel[1]) / 255.0; // G
			blob[b_idx] = static_cast<float_t>(pixel[0]) / 255.0; // B
		}
	}
}
