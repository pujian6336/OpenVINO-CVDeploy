#include "postprocess.h"
#include <cmath>

static float_t sigmoid(float_t x)
{
	return 1.0 / (1.0 + exp(-x));
}

static bool cmp(const Detection& a, const Detection& b) {
	return a.conf > b.conf;
}

static float_t iou(Bbox& lbox, Bbox& rbox)
{
	Bbox interBox;
	interBox.left = std::max(lbox.left, rbox.left);
	interBox.top = std::max(lbox.top, rbox.top);
	interBox.right = std::min(lbox.right, rbox.right);
	interBox.bottom = std::min(lbox.bottom, rbox.bottom);

	if (interBox.left > interBox.right || interBox.top > interBox.bottom) return 0.0f;

	float interBoxArea = (interBox.right - interBox.left) * (interBox.bottom - interBox.top);

	float unionBoxArea = (lbox.right - lbox.left) * (lbox.bottom - lbox.top) +
		(rbox.right - rbox.left) * (rbox.bottom - rbox.top) -
		interBoxArea;

	return interBoxArea / unionBoxArea;
}

static float iou_actual(Bbox lbox, Bbox& rbox) {
	Bbox interBox;
	interBox.left = (std::max)(lbox.left, rbox.left);
	interBox.top = (std::max)(lbox.top, rbox.top);
	interBox.right = (std::min)(lbox.right, rbox.right);
	interBox.bottom = (std::min)(lbox.bottom, rbox.bottom);

	if (interBox.left > interBox.right || interBox.top > interBox.bottom)
		return 0.0f;

	// (2,10)ʵ�ʳ�����9��������10-2=8
	float interBoxArea = (interBox.right - interBox.left + 1.0f) *
		(interBox.bottom - interBox.top + 1.0f);

	float unionBoxArea =
		(lbox.right - lbox.left + 1.0f) * (lbox.bottom - lbox.top + 1.0f) +
		(rbox.right - rbox.left + 1.0f) * (rbox.bottom - rbox.top + 1.0f) -
		interBoxArea;

	return interBoxArea / unionBoxArea;
}

// void detect_post(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, float_t scale, int32_t topK)
// {
// 	cv::Mat output_buffer(numObjects, numClasses + 5, CV_32F, data);

// 	std::vector<int> labels;
// 	std::vector<float> scores;
// 	std::vector<cv::Rect> boxes;

// 	for (int32_t i = 0; i < numObjects; i++)
// 	{
// 		float_t conf = output_buffer.at<float_t>(i, 4);
// 		if (conf < scoreThres) continue;

// 		cv::Mat classes_scores = output_buffer.row(i).colRange(5, numClasses + 5);
// 		cv::Point class_id;
// 		double maxClassScore;
// 		cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);
// 		maxClassScore *= conf;
// 		if (maxClassScore < scoreThres) continue;

// 		scores.push_back(maxClassScore);
// 		labels.push_back(class_id.x);
// 		float cx = output_buffer.at<float_t>(i, 0);
// 		float cy = output_buffer.at<float>(i, 1);
// 		float w = output_buffer.at<float>(i, 2);
// 		float h = output_buffer.at<float>(i, 3);

// 		int left = int(cx - 0.5 * w);
// 		int top = int(cy - 0.5 * h);
// 		int width = int(w);
// 		int height = int(h);
// 		boxes.push_back(cv::Rect(left, top, width, height));

// 		if (boxes.size() > topK) break;
// 	}

// 	std::vector<int> indices;
// 	//cv::dnn::NMSBoxes(boxes, scores, scoreThres, iouThres, indices);
// 	cv::dnn::NMSBoxesBatched(boxes, scores, labels, scoreThres, iouThres, indices);
// 	for (size_t i = 0; i < indices.size(); i++)
// 	{
// 		int index = indices[i];
// 		int class_id = labels[index];
// 		float_t score = scores[index];
// 		float_t left = boxes[index].x / scale;
// 		float_t top = boxes[index].y / scale;
// 		float_t right = left + boxes[index].width / scale;
// 		float_t bottom = top + boxes[index].height / scale;
// 		res.emplace_back(left, top, right, bottom, score, class_id);
// 	}
// }

void detect_decode_nms(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK)
{
	std::map<float, std::vector<Detection>> m;
	int32_t count = 0;

	int32_t object_width = numClasses + 5;
	for (int32_t i = 0; i < numObjects; i++)
	{
		float_t* pitem = data + i * object_width;

		float_t conf = pitem[4];
		if (conf < scoreThres) continue;

		int classId = 0;
		float_t maxClassScore = 0.0;
		for (uint32_t j = 0; j < numClasses; j++)
		{
			float_t prob = pitem[j + 5];
			if (prob > maxClassScore)
			{
				maxClassScore = prob;
				classId = j;
			}
		}

		float_t score = conf * maxClassScore;
		if (score < scoreThres) continue;

		float_t cx = *pitem++;
		float_t cy = *pitem++;
		float_t width = *pitem++;
		float_t height = *pitem++;
		Detection det;
		
		det.bbox.left = cx - width * 0.5f;
		det.bbox.top = cy - height * 0.5f;
		det.bbox.right = cx + width * 0.5f;
		det.bbox.bottom = cy + height * 0.5f;
		det.class_id = classId;
		det.conf = score;
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);

		count++;
		if (count > topK) break;
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (iou(item.bbox, dets[n].bbox) >= iouThres) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

void detect_decode_nms_multi_label(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres)
{
	std::map<float, std::vector<Detection>> m;
	int32_t count = 0;

	int32_t object_width = numClasses + 5;
	for (int32_t i = 0; i < numObjects; i++)
	{
		float_t* pitem = data + i * object_width;

		float_t conf = pitem[4];
		if (conf < scoreThres) continue;

		float_t cx = pitem[0];
		float_t cy = pitem[1];
		float_t width = pitem[2];
		float_t height = pitem[3];

		for (uint32_t j = 0; j < numClasses; j++)
		{
			float_t prob = pitem[j + 5] * conf;
			if (prob > scoreThres)
			{
				Detection det;
				det.bbox.left = cx - width * 0.5f;
				det.bbox.top = cy - height * 0.5f;
				det.bbox.right = cx + width * 0.5f;
				det.bbox.bottom = cy + height * 0.5f;
				det.class_id = j;
				det.conf = prob;
				if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
				m[det.class_id].push_back(det);
			}
		}
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (iou(item.bbox, dets[n].bbox) >= iouThres) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

void detect_decode(float* data, int32_t num_anchors, int32_t grid_height, int32_t grid_width, int32_t input_height, int32_t input_width,
	int32_t num_classes, float_t score_thres, std::vector<float_t> anchors, std::vector<Detection> &dets)
{
	int32_t cellSize = num_classes + 5;
	int grid_len = grid_height * grid_width;
	for (int a = 0; a < num_anchors; ++a)
	{
		for (int i = 0; i < grid_height; ++i)
		{
			for (int j = 0; j < grid_width; ++j)
			{
				float_t* pitem = data + a * grid_len * cellSize + i * grid_width + j ;

				float_t conf = sigmoid(pitem[4 * grid_len]);
				if (conf < score_thres) continue;

				int classId = 0;
				float_t maxClassScore = -10.0f;
				for (uint32_t ci = 0; ci < num_classes; ci++)
				{
					float_t prob = pitem[(ci + 5) * grid_len];

					if (prob > maxClassScore)
					{
						maxClassScore = prob;
						classId = ci;
					}
				}
				maxClassScore = sigmoid(maxClassScore);

				float_t score = conf * maxClassScore;
				if (score < score_thres) continue;

				float_t cx = *pitem;
				float_t cy = pitem[1 * grid_len];
				float_t width = pitem[2 * grid_len];
				float_t height = pitem[3 * grid_len];

				cx = (sigmoid(cx) * 2 - 0.5 + j) * input_width / grid_width;
				cy = (sigmoid(cy) * 2 - 0.5 + i) * input_height / grid_height;
				width = pow(sigmoid(width) * 2, 2) * anchors[a * 2];
				height = pow(sigmoid(height) * 2, 2) * anchors[a * 2 + 1];

				Detection det;

				det.bbox.left = cx - width * 0.5f;
				det.bbox.top = cy - height * 0.5f;
				det.bbox.right = cx + width * 0.5f;
				det.bbox.bottom = cy + height * 0.5f;
				det.class_id = classId;
				det.conf = score;
				dets.push_back(det);
			}
		}
	}
}

void nms(std::vector<Detection> input, std::vector<Detection> &res, float_t iou_thres)
{
	std::map<float, std::vector<Detection>> m;

	for (auto det : input)
	{
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (iou(item.bbox, dets[n].bbox) >= iou_thres) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

void seg_post(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK)
{
	cv::Mat output_buffer(numObjects, numClasses + 37, CV_32F, data);

	std::vector<int> labels;
	std::vector<float> scores;
	std::vector<cv::Rect> boxes;
	std::vector<cv::Mat> mask_confs;

	for (int32_t i = 0; i < numObjects; i++)
	{
		float_t conf = output_buffer.at<float_t>(i, 4);
		if (conf < scoreThres) continue;

		cv::Mat classes_scores = output_buffer.row(i).colRange(5, numClasses + 5);
		cv::Point class_id;
		double maxClassScore;
		cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);
		maxClassScore *= conf;
		if (maxClassScore < scoreThres) continue;

		scores.push_back(maxClassScore);
		labels.push_back(class_id.x);
		float cx = output_buffer.at<float_t>(i, 0);
		float cy = output_buffer.at<float>(i, 1);
		float w = output_buffer.at<float>(i, 2);
		float h = output_buffer.at<float>(i, 3);

		int left = int(cx - 0.5 * w);
		int top = int(cy - 0.5 * h);
		int width = int(w);
		int height = int(h);
		boxes.push_back(cv::Rect(left, top, width, height));
		cv::Mat mask_conf = output_buffer.row(i).colRange(numClasses + 5, numClasses + 37);
		mask_confs.push_back(mask_conf);

		if (boxes.size() > topK) break;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores, scoreThres, iouThres, indices);
	for (size_t i = 0; i < indices.size(); i++)
	{
		int index = indices[i];
		int class_id = labels[index];
		float_t score = scores[index];
		float_t left = boxes[index].x;
		float_t top = boxes[index].y;
		float_t right = left + boxes[index].width;
		float_t bottom = top + boxes[index].height;
		float mask[32];
		memcpy(&mask, mask_confs[index].ptr(), 32 * sizeof(float));
		res.emplace_back(left, top, right, bottom, score, class_id, mask);
	}
}

void segment_decode_nms(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK)
{
	std::map<float, std::vector<Detection>> m;
	int32_t count = 0;

	int32_t object_width = numClasses + 5;
	for (int32_t i = 0; i < numObjects; i++)
	{
		float_t* pitem = data + i * object_width;

		float_t conf = pitem[4];
		if (conf < scoreThres) continue;

		int classId = 0;
		float_t maxClassScore = 0.0;
		for (uint32_t j = 0; j < numClasses; j++)
		{
			float_t prob = pitem[j + 5];
			if (prob > maxClassScore)
			{
				maxClassScore = prob;
				classId = j;
			}
		}

		float_t score = conf * maxClassScore;
		if (score < scoreThres) continue;

		float_t cx = *pitem++;
		float_t cy = *pitem++;
		float_t width = *pitem++;
		float_t height = *pitem++;
		Detection det;

		det.bbox.left = cx - width * 0.5f;
		det.bbox.top = cy - height * 0.5f;
		det.bbox.right = cx + width * 0.5f;
		det.bbox.bottom = cy + height * 0.5f;
		det.class_id = classId;
		det.conf = score;
		memcpy(det.mask, pitem + numClasses + 1, 32 * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);

		count++;
		if (count > topK) break;
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (iou(item.bbox, dets[n].bbox) >= iouThres) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}


static cv::Rect get_downscale_rect(Bbox bbox, float scale, int proto_height, int proto_width) {
	float left = std::max(0.f, bbox.left / scale);
	float top = std::max(0.f, bbox.top / scale);
	float right = std::min((float)proto_width - 1, bbox.right / scale);
	float bottom = std::min((float)proto_height - 1, bbox.bottom / scale);
	return cv::Rect(round(left), round(top), round(right - left), round(bottom - top));
}

std::vector<cv::Mat> process_masks(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio)
{
	std::vector<cv::Mat> masks;
	for (size_t i = 0; i < dets.size(); i++)
	{
		cv::Mat mask = cv::Mat::zeros(proto_height, proto_width, CV_32FC1);
		auto r = get_downscale_rect(dets[i].bbox, mask_ratio, proto_height, proto_width);
		for (int x = r.x; x < r.x + r.width; x++)
		{
			for (int y = r.y; y < r.y + r.height; y++)
			{
				float e = 0.0f;
				for (int j = 0; j < 32; j++)
				{
					e += dets[i].mask[j] * proto[j * proto_width * proto_height + y * mask.cols + x];
				}
				e = 1.0f / (1.0f + expf(-e));
				mask.at<float>(y, x) = e;
			}
		}
		cv::resize(mask, mask, cv::Size(proto_width * mask_ratio, mask_ratio * proto_height));
		masks.push_back(mask > 0.5f);
	}
	return masks;
}

cv::Mat segment_process_mask(const float* proto, int proto_height, int proto_width, std::vector<Detection>& dets, int mask_ratio)
{
	cv::Mat mask = cv::Mat::zeros(proto_height, proto_width, CV_32FC1);
	for (size_t i = 0; i < dets.size(); i++)
	{
		auto r = get_downscale_rect(dets[i].bbox, mask_ratio, proto_height, proto_width);
		for (int x = r.x; x < r.x + r.width; x++)
		{
			for (int y = r.y; y < r.y + r.height; y++)
			{
				float e = 0.0f;
				for (int j = 0; j < 32; j++)
				{
					e += dets[i].mask[j] * proto[j * proto_width * proto_height + y * mask.cols + x];
				}
				e = sigmoid(e);
				mask.at<float>(y, x) = e;
			}
		}
	}
	cv::resize(mask, mask, cv::Size(proto_width * mask_ratio, mask_ratio * proto_height));
	return mask > 0.5f;
}

void mask2OriginalSize(std::vector<cv::Mat>& masks, cv::Mat& img, float_t scale)
{
	int32_t w, h;
	w = img.cols * scale;
	h = img.rows * scale;

	cv::Rect r(0, 0, w, h);
	for (size_t i = 0; i < masks.size(); i++)
	{
		cv::resize(masks[i](r), masks[i], img.size());
	}
}

void mask2OriginalSize(cv::Mat& mask, cv::Mat& img, float_t scale)
{
	int32_t w, h;
	w = img.cols * scale;
	h = img.rows * scale;

	cv::Rect r(0, 0, w, h);
	cv::resize(mask(r), mask, img.size());
}

void box2OriginalSize(std::vector<Detection>& res, float_t scale)
{
	for (size_t i = 0; i < res.size(); i++)
	{
		res[i].bbox.left = res[i].bbox.left / scale;
		res[i].bbox.top = res[i].bbox.top / scale;
		res[i].bbox.right = res[i].bbox.right / scale;
		res[i].bbox.bottom = res[i].bbox.bottom / scale;
	}
}

void box2OriginalSize(std::vector<Detection>& res, float_t scale, int dw, int dh)
{
	for (size_t i = 0; i < res.size(); i++)
	{
		res[i].bbox.left = (res[i].bbox.left - dw) / scale;
		res[i].bbox.top = (res[i].bbox.top - dh) / scale;
		res[i].bbox.right = (res[i].bbox.right - dw) / scale;
		res[i].bbox.bottom = (res[i].bbox.bottom - dh) / scale;
	}
}




