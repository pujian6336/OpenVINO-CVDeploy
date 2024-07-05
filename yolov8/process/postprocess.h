#pragma once
#include <opencv2/opencv.hpp>
#include "utils/types.h"

void opencv_post(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, float_t scale);

void detect_decode_nms(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK);

// map≤‚ ‘ multi_label∞Ê±æ
void detect_decode_nms_multi_label(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK);

void opencv_post_seg(float* data,std::vector<Detection>& res, int32_t numObjects, int32_t numClasses,
	float_t scoreThres, float_t iouThres);

void segment_decode_nms(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK);

std::vector<cv::Mat> process_masks(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio);

cv::Mat segment_process_mask(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio);

void mask2OriginalSize(std::vector<cv::Mat>& masks, cv::Mat& imgs, float_t scale);

void mask2OriginalSize(cv::Mat & mask, cv::Mat& img, float_t scale);

void box2OriginalSize(std::vector<Detection>& res, float_t scale);

void box2OriginalSize(std::vector<Detection>& res, float_t scale, int dw, int dh);

void pose_post(float* data, std::vector<KeyPointResult>& res, int32_t numObjects, int32_t num_keypoint, float_t scoreThres, float_t iouThres, float_t scale);

void pose_decode_post(float* data, std::vector<KeyPointResult>& res, int32_t numObjects, int32_t num_keypoint, float_t scoreThres, float_t iouThres, int32_t topK);

void pose2OriginalSize(std::vector<KeyPointResult>& res, float_t scale);

void obb_post(float* data, std::vector<OBBResult>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK);

void obb_decode_post(float* data, std::vector<OBBResult>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK);

void obb2OriginalSize(std::vector<OBBResult>& res, float_t scale);


