#include "yolov8_seg.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

YOLOV8_SEG::YOLOV8_SEG(Config cfg) : m_cfg(cfg)
{
	m_scale = 0.0f;
	m_input_data = nullptr;
}

YOLOV8_SEG::~YOLOV8_SEG()
{
}

bool YOLOV8_SEG::init()
{
	ov::Core core;

	auto compiled_model = core.compile_model(m_cfg.model_path, "CPU");

	m_infer_request = compiled_model.create_infer_request();

	auto input_port = compiled_model.input();

	m_input_element_type = input_port.get_element_type();

	m_input_shape = input_port.get_shape();

	m_input_data = (float_t*)malloc(m_input_shape[1] * m_input_shape[2] * m_input_shape[3] * sizeof(float_t));

	m_imgs.resize(m_input_shape[0]);

	auto output_port = compiled_model.outputs();

	m_output_det_shape = output_port[0].get_shape();

	m_output_proto_shape = output_port[1].get_shape();

	return true;
}

void YOLOV8_SEG::Run(const cv::Mat img, std::vector<Detection>& res, std::vector<cv::Mat>& masks)
{
	preprocess(img);
	infer();
	postprocess(res, masks);
}

void YOLOV8_SEG::preprocess(const cv::Mat img)
{
	img2blob(img, m_input_data, m_input_shape[3], m_input_shape[2], m_scale);
	m_imgs[0] = img;
}

void YOLOV8_SEG::infer()
{
	ov::Tensor input_tensor(m_input_element_type, m_input_shape, m_input_data);
	m_infer_request.set_input_tensor(input_tensor);
	m_infer_request.infer();
}

void YOLOV8_SEG::postprocess(std::vector<Detection>& res,std::vector<cv::Mat> &masks)
{
	ov::Tensor output0 = m_infer_request.get_output_tensor(0);
	ov::Tensor output1 = m_infer_request.get_output_tensor(1);

	float* data = output0.data<float_t>();
	float* proto = output1.data<float_t>();
	//opencv_post_seg(data, res, m_output_det_shape[2], m_output_det_shape[1] - 36, m_cfg.conf_threshold, m_cfg.iou_threshold);
	segment_decode_nms(data, res, m_output_det_shape[2], m_output_det_shape[1] - 36, m_cfg.conf_threshold, m_cfg.iou_threshold, m_cfg.topK);
	masks = process_masks(proto, m_output_proto_shape[3], m_output_proto_shape[2], res, 4);

	mask2OriginalSize(masks, m_imgs[0], m_scale);
	box2OriginalSize(res, m_scale);
}

void YOLOV8_SEG::postprocess(std::vector<Detection>& res, cv::Mat& mask)
{
	ov::Tensor output0 = m_infer_request.get_output_tensor(0);
	ov::Tensor output1 = m_infer_request.get_output_tensor(1);

	float* data = output0.data<float_t>();
	float* proto = output1.data<float_t>();
	segment_decode_nms(data, res, m_output_det_shape[2], m_output_det_shape[1] - 36, m_cfg.conf_threshold, m_cfg.iou_threshold, m_cfg.topK);
	mask = segment_process_mask(proto, m_output_proto_shape[3], m_output_proto_shape[2], res, 4);

	mask2OriginalSize(mask, m_imgs[0], m_scale);
	box2OriginalSize(res, m_scale);
}
