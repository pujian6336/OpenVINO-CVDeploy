#include "yolov10.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

YOLOV10::YOLOV10(Config cfg) :m_cfg(cfg)
{
	m_scale = 0.0f;
}

YOLOV10::~YOLOV10()
{
}

bool YOLOV10::init()
{
	ov::Core core;

	std::shared_ptr<ov::Model> model = core.read_model(m_cfg.model_path);

	if (model->is_dynamic())
	{
		model->reshape({ m_cfg.batch_size,3,static_cast<long int>(m_cfg.input_height),static_cast<long int>(m_cfg.input_width) });
	}

	ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
	ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
	ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });
	ppp.input().model().set_layout("NCHW");
	ppp.output().tensor().set_element_type(ov::element::f32);
	model = ppp.build();

	auto compiled_model = core.compile_model(model, "CPU");

	m_infer_request = compiled_model.create_infer_request();

	auto input_port = compiled_model.input();

	m_input_element_type = input_port.get_element_type();

	m_input_shape = input_port.get_shape();

	auto output_port = compiled_model.output();

	m_output_shape = output_port.get_shape();

	return true;
}

void YOLOV10::Run(const cv::Mat img, std::vector<Detection>& res)
{
	preprocess(img);
	infer();
	postprocess(res);
}

void YOLOV10::preprocess(const cv::Mat img)
{
	imgResizePad(img, m_input_img, m_input_shape[2], m_input_shape[1], m_scale);
}

void YOLOV10::infer()
{
	float* data = (float*)m_input_img.data;
	ov::Tensor input_tensor(m_input_element_type, m_input_shape, data);
	m_infer_request.set_input_tensor(input_tensor);
	m_infer_request.infer();
}

void YOLOV10::postprocess(std::vector<Detection>& res)
{
	ov::Tensor output = m_infer_request.get_output_tensor(0);
	float_t* data = output.data<float_t>();
	thresholdFilter(data, res, m_cfg.conf_threshold);
	box2OriginalSize(res, m_scale);
}

