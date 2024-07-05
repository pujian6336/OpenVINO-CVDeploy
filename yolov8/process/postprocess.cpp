#include "postprocess.h"
#include <omp.h>

template <typename T>
static bool cmp(const T& a, const T& b) {
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


void opencv_post(float* data, std::vector<Detection> &res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, float_t scale)
{
	cv::Mat output_buffer(numClasses + 4, numObjects, CV_32F, data);
	cv::transpose(output_buffer, output_buffer);

	std::vector<int> labels;
	std::vector<float> scores;
	std::vector<cv::Rect> boxes;

	for (int32_t i = 0; i < numObjects; i++)
	{
		cv::Mat classes_scores = output_buffer.row(i).colRange(4, numClasses + 4);
		cv::Point class_id;
		double maxClassScore;
		cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > scoreThres)
		{
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
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores, scoreThres, iouThres, indices);
	for (size_t i = 0; i < indices.size(); i++)
	{
		int index = indices[i];
		int class_id = labels[index];
		float_t score = scores[index];
		float_t left = boxes[index].x / scale;
		float_t top = boxes[index].y / scale;
		float_t right = left + boxes[index].width / scale;
		float_t bottom = top + boxes[index].height / scale;
		res.emplace_back(left, top, right, bottom, score, class_id);
	}
}

void detect_decode_nms(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK)
{
	std::map<float, std::vector<Detection>> m;
	int32_t count = 0;

	for (int32_t i = 0; i < numObjects; i++)
	{
		int32_t class_id = 0;
		float_t score = 0.0;
		for (size_t idx = 0; idx < numClasses; idx++)
		{
			float_t prob = data[(4 + idx) * numObjects + i];
			if (prob > score)
			{
				score = prob;
				class_id = idx;
			}
		}

		if (score < scoreThres) continue;

		float_t cx = data[i];
		float_t cy = data[i + numObjects];
		float_t width = data[i + 2 * numObjects];
		float_t height = data[i + 3 * numObjects];
		
		Detection det;
		det.bbox.left = cx - width * 0.5f;
		det.bbox.top = cy - height * 0.5f;
		det.bbox.right = cx + width * 0.5f;
		det.bbox.bottom = cy + height * 0.5f;
		det.class_id = class_id;
		det.conf = score;

		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);

		count++;
		if (count > topK) break;
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp<Detection>);
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

void detect_decode_nms_multi_label(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK)
{
	std::map<float, std::vector<Detection>> m;

	for (int32_t i = 0; i < numObjects; i++)
	{
		int32_t class_id = 0;
		float_t score = 0.0;
		for (size_t idx = 0; idx < numClasses; idx++)
		{
			float_t prob = data[(4 + idx) * numObjects + i];
			if (prob > scoreThres)
			{
				score = prob;
				class_id = idx;

				float_t cx = data[i];
				float_t cy = data[i + numObjects];
				float_t width = data[i + 2 * numObjects];
				float_t height = data[i + 3 * numObjects];

				Detection det;
				det.bbox.left = cx - width * 0.5f;
				det.bbox.top = cy - height * 0.5f;
				det.bbox.right = cx + width * 0.5f;
				det.bbox.bottom = cy + height * 0.5f;
				det.class_id = class_id;
				det.conf = score;

				if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
				m[det.class_id].push_back(det);
			}
		}
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp<Detection>);
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

void opencv_post_seg(float* data, std::vector<Detection>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres)
{

	cv::Mat output_buffer(numClasses + 36, numObjects, CV_32F, data);
	cv::transpose(output_buffer, output_buffer);

	std::vector<int> labels;
	std::vector<float> scores;
	std::vector<cv::Rect> boxes;
	std::vector<cv::Mat> mask_confs;

	for (int32_t i = 0; i < numObjects; i++)
	{
		cv::Mat classes_scores = output_buffer.row(i).colRange(4, numClasses + 4);
		cv::Point class_id;
		double maxClassScore;
		cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > scoreThres)
		{
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
			cv::Mat mask_conf = output_buffer.row(i).colRange(numClasses + 4, numClasses + 36);
			mask_confs.push_back(mask_conf);
		}
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

	for (int32_t i = 0; i < numObjects; i++)
	{
		int32_t class_id = 0;
		float_t score = 0.0;
		for (size_t idx = 0; idx < numClasses; idx++)
		{
			float_t prob = data[(4 + idx) * numObjects + i];
			if (prob > score)
			{
				score = prob;
				class_id = idx;
			}
		}

		if (score < scoreThres) continue;

		float_t cx = data[i];
		float_t cy = data[i + numObjects];
		float_t width = data[i + 2 * numObjects];
		float_t height = data[i + 3 * numObjects];

		Detection det;
		det.bbox.left = cx - width * 0.5f;
		det.bbox.top = cy - height * 0.5f;
		det.bbox.right = cx + width * 0.5f;
		det.bbox.bottom = cy + height * 0.5f;
		det.class_id = class_id;
		det.conf = score;
		for (size_t maskidx = 0; maskidx < 32; ++maskidx)
		{
			det.mask[maskidx] = data[i + (numClasses + 4 + maskidx) * numObjects];
		}

		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);

		count++;
		if (count > topK) break;
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp<Detection>);
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

static float sigmoid_function(float a) {
	float b = 1. / (1. + exp(-a));
	return b;
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
				e = sigmoid_function(e);
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
				e = sigmoid_function(e);
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

void pose_post(float* data, std::vector<KeyPointResult>& res, int32_t numObjects, int32_t num_keypoint, float_t scoreThres, float_t iouThres, float_t scale)
{
	cv::Mat output_buffer(num_keypoint * 3 + 5, numObjects, CV_32F, data);
	cv::transpose(output_buffer, output_buffer);

	std::vector<float> scores;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<KeyPoint>> kps;

	for (int32_t i = 0; i < numObjects; i++)
	{
		float_t score = output_buffer.at<float>(i, 4);

		if (score > scoreThres)
		{
			scores.push_back(score);
			float_t cx = output_buffer.at<float_t>(i, 0);
			float_t cy = output_buffer.at<float_t>(i, 1);
			float_t w = output_buffer.at<float_t>(i, 2);
			float_t h = output_buffer.at<float_t>(i, 3);

			int left = int(cx - 0.5 * w);
			int top = int(cy - 0.5 * h);
			int width = int(w);
			int height = int(h);
			boxes.push_back(cv::Rect(left, top, width, height));
			
			std::vector<KeyPoint> vkp;
			for (int j = 0; j < num_keypoint; j++) {
				KeyPoint kp;
				kp.x = output_buffer.at<float>(i, j * 3 + 5);
				kp.y = output_buffer.at<float>(i, j * 3 + 6);
				kp.score = output_buffer.at<float>(i, j * 3 + 7);
				kp.id = j;
				vkp.push_back(kp);
			}
			kps.push_back(vkp);
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, scores, scoreThres, iouThres, indices);

	for (size_t i = 0; i < indices.size(); i++)
	{
		KeyPointResult tempres;

		int index = indices[i];
		tempres.conf = scores[index];
		tempres.bbox.left = boxes[index].x / scale;
		tempres.bbox.top = boxes[index].y / scale;
		tempres.bbox.right = tempres.bbox.left + boxes[index].width / scale;
		tempres.bbox.bottom = tempres.bbox.top + boxes[index].height / scale;

		KeyPoint kp;
		for (int i = 0; i < num_keypoint; i++) {
			kp.x = kps[index][i].x / scale;
			kp.y = kps[index][i].y / scale;
			kp.id = kps[index][i].id;
			kp.score = kps[index][i].score;
			tempres.keyPoints.push_back(kp);
		}
		res.push_back(tempres);
	}
}


void pose_decode_post(float* data, std::vector<KeyPointResult>& res, int32_t numObjects, int32_t num_keypoint, float_t scoreThres, float_t iouThres, int32_t topK)
{
	std::vector<KeyPointResult> temp;
	int32_t count = 0;

	for (int32_t i = 0; i < numObjects; i++)
	{
		float_t score = data[4 * numObjects + i];

		if (score < scoreThres) continue;

		float_t cx = data[i];
		float_t cy = data[i + numObjects];
		float_t width = data[i + 2 * numObjects];
		float_t height = data[i + 3 * numObjects];

		KeyPointResult kpres;
		kpres.bbox.left = cx - width * 0.5f;
		kpres.bbox.top = cy - height * 0.5f;
		kpres.bbox.right = cx + width * 0.5f;
		kpres.bbox.bottom = cy + height * 0.5f;
		kpres.conf = score;

		for (int j = 0; j < num_keypoint; j++) {
			KeyPoint kp;
			kp.x = data[i + (j * 3 + 5) * numObjects];
			kp.y = data[i + (j * 3 + 6) * numObjects];
			kp.score = data[i + (j * 3 + 7) * numObjects];
			kp.id = j;
			kpres.keyPoints.push_back(kp);
		}
		temp.push_back(kpres);

		count++;
		if (count > topK) break;
	}

	std::sort(temp.begin(), temp.end(), cmp<KeyPointResult>);
	for (size_t m = 0; m < temp.size(); ++m)
	{
		auto& item = temp[m];
		res.push_back(item);
		for (size_t n = m + 1; n < temp.size(); ++n)
		{
			if (iou(item.bbox, temp[n].bbox) >= iouThres) {
				temp.erase(temp.begin() + n);
				--n;
			}
		}
	}
}

void pose2OriginalSize(std::vector<KeyPointResult>& res, float_t scale)
{
	for (size_t i = 0; i < res.size(); i++)
	{
		res[i].bbox.left = res[i].bbox.left / scale;
		res[i].bbox.top = res[i].bbox.top / scale;
		res[i].bbox.right = res[i].bbox.right / scale;
		res[i].bbox.bottom = res[i].bbox.bottom / scale;

		for (size_t j = 0; j < res[i].keyPoints.size(); j++)
		{
			res[i].keyPoints[j].x = res[i].keyPoints[j].x / scale;
			res[i].keyPoints[j].y = res[i].keyPoints[j].y / scale;
		}
	}
}

static void convariance_matrix(float w, float h, float r, float& a, float& b, float& c)
{
	float a_val = w * w / 12.0f;
	float b_val = h * h / 12.0f;
	float cos_r = cosf(r);
	float sin_r = sinf(r);

	a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
	b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
	c = (a_val - b_val) * sin_r * cos_r;
}

static float box_probiou(
	float cx1, float cy1, float w1, float h1, float r1,
	float cx2, float cy2, float w2, float h2, float r2,
	float eps = 1e-7
)
{
	float a1, b1, c1, a2, b2, c2;
	convariance_matrix(w1, h1, r1, a1, b1, c1);
	convariance_matrix(w2, h2, r2, a2, b2, c2);

	float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
	float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
	float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) / (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) + eps);
	float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
	bd = fmaxf(fminf(bd, 100.0f), eps);
	float hd = sqrtf(1.0f - expf(-bd) + eps);
	return 1 - hd;
}

void obb_post(float* data, std::vector<OBBResult>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK)
{
	cv::Mat output_buffer(numClasses + 5, numObjects, CV_32F, data);
	cv::transpose(output_buffer, output_buffer);

	std::vector<OBBResult> tempres;

	for (int32_t i = 0; i < numObjects; i++)
	{
		cv::Mat classes_scores = output_buffer.row(i).colRange(4, numClasses + 4);
		cv::Point class_id;
		double maxClassScore;
		cv::minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > scoreThres)
		{
			float cx = output_buffer.at<float_t>(i, 0);
			float cy = output_buffer.at<float>(i, 1);
			float w = output_buffer.at<float>(i, 2);
			float h = output_buffer.at<float>(i, 3);
			float r = output_buffer.at<float>(i, numClasses+4);

			tempres.emplace_back(cx, cy, w, h, r, maxClassScore, class_id.x);

			if (tempres.size() > topK) break;
		}
	}

	std::map<float, std::vector<OBBResult>> m;

	for (int32_t i = 0; i < tempres.size(); i++)
	{
		if (m.count(tempres[i].class_id) == 0)
		{
			m.emplace(tempres[i].class_id, std::vector<OBBResult>());
		}
		m[tempres[i].class_id].push_back(tempres[i]);
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp<OBBResult>);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (box_probiou(item.cx, item.cy, item.width, item.height, item.angle,
					dets[n].cx, dets[n].cy, dets[n].width, dets[n].height, dets[n].angle) >= iouThres) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

void obb_decode_post(float* data, std::vector<OBBResult>& res, int32_t numObjects, int32_t numClasses, float_t scoreThres, float_t iouThres, int32_t topK)
{
	std::map<float, std::vector<OBBResult>> m;
	int32_t count = 0;

	for (int32_t i = 0; i < numObjects; i++)
	{
		int32_t class_id = 0;
		float_t score = 0.0;
		for (size_t idx = 0; idx < numClasses; idx++)
		{
			float_t prob = data[(4 + idx) * numObjects + i];
			if (prob > score)
			{
				score = prob;
				class_id = idx;
			}
		}

		if (score < scoreThres) continue;

		OBBResult obbres;
		obbres.cx = data[i];
		obbres.cy = data[i + numObjects];
		obbres.width = data[i + 2 * numObjects];
		obbres.height = data[i + 3 * numObjects];
		obbres.angle = data[i + (numClasses + 4) * numObjects];
		obbres.class_id = class_id;
		obbres.conf = score;

		if (m.count(obbres.class_id) == 0) m.emplace(obbres.class_id, std::vector<OBBResult>());
		m[obbres.class_id].push_back(obbres);

		count++;
		if (count > topK) break;
	}

	for (auto it = m.begin(); it != m.end(); it++)
	{
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp<OBBResult>);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto& item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (box_probiou(item.cx, item.cy, item.width, item.height, item.angle,
					dets[n].cx, dets[n].cy, dets[n].width, dets[n].height, dets[n].angle) >= iouThres) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

void obb2OriginalSize(std::vector<OBBResult>& res, float_t scale)
{
	for (size_t i = 0; i < res.size(); i++)
	{
		res[i].cx = res[i].cx / scale;
		res[i].cy = res[i].cy / scale;
		res[i].width = res[i].width / scale;
		res[i].height = res[i].height / scale;
	}
}



