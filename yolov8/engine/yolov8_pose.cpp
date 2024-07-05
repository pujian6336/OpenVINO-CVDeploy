#include "yolov8_pose.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

YOLOV8_POSE::YOLOV8_POSE(Config cfg) : m_cfg(cfg)
{
	m_scale = 0.0f;
	float_t* m_input_data = nullptr;
}

YOLOV8_POSE::~YOLOV8_POSE()
{
}

bool YOLOV8_POSE::init()
{
	ov::Core core;

	auto compiled_model = core.compile_model(m_cfg.model_path, "CPU");

	m_infer_request = compiled_model.create_infer_request();

	auto input_port = compiled_model.input();

	m_input_element_type = input_port.get_element_type();

	m_input_shape = input_port.get_shape();

	m_input_data = (float_t*)malloc(3 * m_input_shape[2] * m_input_shape[3] * sizeof(float_t));

	auto output_port = compiled_model.output();

	m_output_shape = output_port.get_shape();

	m_nkps = (m_output_shape[1] - 5) / 3;

	return true;
}

void YOLOV8_POSE::Run(const cv::Mat img, std::vector<KeyPointResult>& res)
{
	preprocess(img);
	infer();
	postprocess(res);
}

void YOLOV8_POSE::preprocess(const cv::Mat img)
{
	img2blob(img, m_input_data, m_input_shape[3], m_input_shape[2], m_scale);
}

void YOLOV8_POSE::infer()
{
	ov::Tensor input_tensor(m_input_element_type, m_input_shape, m_input_data);
	m_infer_request.set_input_tensor(input_tensor);
	m_infer_request.infer();
}

void YOLOV8_POSE::postprocess(std::vector<KeyPointResult>& res)
{
	ov::Tensor output = m_infer_request.get_output_tensor(0);
	float* data = output.data<float_t>();
	pose_decode_post(data, res, m_output_shape[2], m_nkps, m_cfg.conf_threshold, m_cfg.iou_threshold, m_cfg.topK);
	pose2OriginalSize(res, m_scale);
}
