#pragma once
#include <chrono>
#include <opencv2/opencv.hpp>
#include "types.h"

namespace utils {
	namespace dataSets
	{
		const std::vector<std::string> coco80 = {
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
			"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
			"hair drier", "toothbrush"
		};
	}

	namespace Colors
	{
		const std::vector<cv::Scalar> color20{
			cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
			cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
			cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
			cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
			cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)
		};
	};

	// 在图像上绘制检测框
	void DrawDetection(cv::Mat& img, const std::vector<Detection>& objects, const std::vector<std::string>& classNames);

	// 统计主机程序运行时间
	class HostTime {
	public:
		HostTime();
		float getUsedTime();
		~HostTime();

	private:
		std::chrono::high_resolution_clock::time_point t1;
		std::chrono::high_resolution_clock::time_point t2;
	};

	void save_txt(const std::vector<Detection>& objects, const std::string& savePath, cv::Mat& imgsBatch);

	void replace_root_extension(std::vector<std::string>& filePath,
		const std::string& rootPath, const std::string& newPath, const std::string& extension);

	std::string replacePathAndExtension(const std::string& originalPath, const std::string& newPathPart, const std::string& newExtension="");

}

