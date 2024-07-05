#pragma once
#include <opencv2/opencv.hpp>
#include "utils/types.h"

// ʹ��opencv�Դ���nmsʵ��
// void detect_post(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, 
// 	float_t scoreThres, float_t iouThres, float_t scale,int32_t topK);

void detect_decode_nms(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses,
	float_t scoreThres, float_t iouThres, int32_t topK);

void detect_decode_nms_multi_label(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses,
	float_t scoreThres, float_t iouThres);

void detect_decode(float* data, int32_t num_anchors, int32_t net_height, int32_t net_width, int32_t input_height, int32_t input_width, 
	int32_t num_classes, float_t score_thres, std::vector<float_t> anchors, std::vector<Detection> &dets);

void nms(std::vector<Detection> input, std::vector<Detection> &res, float_t iou_thres);

void seg_post(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses,
	float_t scoreThres, float_t iouThres, int32_t topK);

void segment_decode_nms(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK);

std::vector<cv::Mat> process_masks(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio);

cv::Mat segment_process_mask(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio);

void mask2OriginalSize(std::vector<cv::Mat>& masks, cv::Mat& imgs, float_t scale);

void mask2OriginalSize(cv::Mat& mask, cv::Mat& img, float_t scale);

void box2OriginalSize(std::vector<Detection>& res, float_t scale);

void box2OriginalSize(std::vector<Detection>& res, float_t scale, int dw, int dh);



