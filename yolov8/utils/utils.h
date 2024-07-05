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

		const std::vector<std::string> dotav1 = {
			"plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court", "ground track field",
			"harbor", "bridge", "large vehicle", "small vehicle", "helicopter", "roundabout", "soccer ball field" , "swimming pool"
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

		const std::vector<cv::Scalar> KeyPointsColor{
			cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
			cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(255, 128, 0),
			cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
			cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(51, 153, 255),
			cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
			cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255) };

		const std::vector<cv::Scalar> SkeletonColor{
			cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
			cv::Scalar(51, 153, 255), cv::Scalar(255,  51, 255),cv::Scalar(255,  51, 255),
			cv::Scalar(255,  51, 255),cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
			cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128,   0),
			cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),
			cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),
			cv::Scalar(0, 255,   0) };
	}

	const std::vector<cv::Point2i> skeleton{
		cv::Point2i(16, 14), cv::Point2i(14, 12), cv::Point2i(17, 15),
		cv::Point2i(15, 13), cv::Point2i(12, 13), cv::Point2i(6, 12),
		cv::Point2i(7, 13),  cv::Point2i(6, 7),   cv::Point2i(6, 8),
		cv::Point2i(7, 9),   cv::Point2i(8, 10),  cv::Point2i(9, 11),
		cv::Point2i(2, 3),   cv::Point2i(1, 2),   cv::Point2i(1, 3),
		cv::Point2i(2, 4),   cv::Point2i(3, 5),   cv::Point2i(4, 6),
		cv::Point2i(5, 7)
	};

	// ��ͼ���ϻ��Ƽ���
	void DrawDetection(cv::Mat& img, const std::vector<Detection>& objects, const std::vector<std::string>& classNames);

	void DrawSegmentation(cv::Mat& img, const std::vector<Detection>& dets, const std::vector<cv::Mat>& masks, const std::vector<std::string>& classNames);
	void DrawSegmentation(cv::Mat& img, const std::vector<Detection>& dets, const cv::Mat& masks, const std::vector<std::string>& classNames);

	void DrawKeyPoints(cv::Mat& img, const std::vector<KeyPointResult>& res, const std::string className, const float conf_thres = 0.5);

	void DrawOBB(cv::Mat& img, const std::vector<OBBResult>& objects, const std::vector<std::string>& classNames);

	// ͳ��������������ʱ��
	class HostTime {
	public:
		HostTime();
		float getUsedTime();
		~HostTime();

	private:
		std::chrono::high_resolution_clock::time_point t1;
		std::chrono::high_resolution_clock::time_point t2;
	};

	std::string replacePathAndExtension(const std::string& originalPath, const std::string& newPathPart, const std::string& newExtension = "");

	void save_txt(const std::vector<Detection>& objects, const std::string& savePath, cv::Mat& img);

	void replace_root_extension(std::vector<std::string>& filePath,
		const std::string& rootPath, const std::string& newPath, const std::string& extension);
}

