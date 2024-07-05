#include "yolov8.h"
#include "process/preprocess.h"
#include "process/postprocess.h"


YOLOV8::YOLOV8(Config cfg) : m_cfg(cfg)
{
	m_scale = 0.0f;
	m_input_data = nullptr;
}

YOLOV8::~YOLOV8()
{
}

bool YOLOV8::init()
{
	ov::Core core;

	std::shared_ptr<ov::Model> model = core.read_model(m_cfg.model_path);

	if (model->is_dynamic())
	{
		model->reshape({ m_cfg.batch_size,3,static_cast<long int>(m_cfg.input_height),static_cast<long int>(m_cfg.input_width) });
	}

	auto compiled_model = core.compile_model(model, "CPU");

	m_infer_request = compiled_model.create_infer_request();

	auto input_port = compiled_model.input();

	m_input_element_type = input_port.get_element_type();

	m_input_shape =  input_port.get_shape();

	m_input_data = (float_t*)malloc(3 * m_input_shape[2] * m_input_shape[3] * sizeof(float_t));

	auto output_port = compiled_model.output();

	m_output_shape = output_port.get_shape();

	return true;
}

void YOLOV8::Run(const cv::Mat img, std::vector<Detection>& res)
{
	preprocess(img);
	infer();
	postprocess(res);
}

void YOLOV8::preprocess(const cv::Mat img)
{
	img2blob(img, m_input_data, m_input_shape[3], m_input_shape[2], m_scale);
}

void YOLOV8::infer()
{
	ov::Tensor input_tensor(m_input_element_type, m_input_shape, m_input_data);
	m_infer_request.set_input_tensor(input_tensor);
	m_infer_request.infer();
}

void YOLOV8::postprocess(std::vector<Detection>& res)
{
	ov::Tensor output = m_infer_request.get_output_tensor(0);
	float* data = output.data<float_t>();
	detect_decode_nms(data, res, m_output_shape[2], m_output_shape[1] - 4, m_cfg.conf_threshold, m_cfg.iou_threshold, m_cfg.topK);
	box2OriginalSize(res, m_scale);
}

void YOLOV8::postprocess_multi_label(std::vector<Detection>& res)
{
	ov::Tensor output = m_infer_request.get_output_tensor(0);
	float* data = output.data<float_t>();
	detect_decode_nms_multi_label(data, res, m_output_shape[2], m_output_shape[1] - 4, m_cfg.conf_threshold, m_cfg.iou_threshold, m_cfg.topK);
	box2OriginalSize(res, m_scale);
}

