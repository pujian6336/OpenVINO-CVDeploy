#include "yolov5.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

YOLOV5::YOLOV5(Config cfg) :m_cfg(cfg)
{
	m_scale = 0.0f;
	m_input_data = nullptr;
}

YOLOV5::~YOLOV5()
{
}

bool YOLOV5::init()
{
	ov::Core core;

	auto compiled_model = core.compile_model(m_cfg.model_path, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

	m_infer_request = compiled_model.create_infer_request();

	auto input_port = compiled_model.input();

	m_input_element_type = input_port.get_element_type();

	m_input_shape = input_port.get_shape();

	m_input_data = (float_t*)malloc(3 * m_input_shape[2] * m_input_shape[3] * sizeof(float_t));

	auto output_port = compiled_model.outputs();

	for (int32_t i = 0; i < output_port.size(); i++)
	{
		m_output_shapes.push_back(output_port[i].get_shape());
	}

	return true;
}

void YOLOV5::Run(const cv::Mat img, std::vector<Detection>& res)
{
	preprocess(img);
	infer();
	postprocess(res);
}

void YOLOV5::preprocess(const cv::Mat img)
{
	img2blob(img, m_input_data, m_input_shape[3], m_input_shape[2], m_scale);
}

void YOLOV5::preprocess_center(const cv::Mat img)
{
	img2blob(img, m_input_data, m_input_shape[3], m_input_shape[2], m_scale, dw, dh);
}

void YOLOV5::infer()
{
	ov::Tensor input_tensor(m_input_element_type, m_input_shape, m_input_data);
	m_infer_request.set_input_tensor(input_tensor);
	m_infer_request.infer();
}

void YOLOV5::postprocess(std::vector<Detection>& res)
{
	if (m_output_shapes.size() == 1)
	{
		ov::Tensor output = m_infer_request.get_output_tensor(0);
		float_t* data = output.data<float_t>();
		detect_decode_nms(data, res, m_output_shapes[0][1], m_output_shapes[0][2] - 5, m_cfg.conf_threshold, m_cfg.iou_threshold, m_cfg.topK);
		box2OriginalSize(res, m_scale);
	}
	else
	{
		int32_t num_anchors = 3;
		std::vector<Detection> temp;
		for (int32_t i = 0; i < m_output_shapes.size(); i++)
		{
			ov::Tensor output = m_infer_request.get_output_tensor(i);
			float* data = output.data<float_t>();
			std::vector<float_t> anchors(m_cfg.anchors.begin() + i * num_anchors * 2, m_cfg.anchors.begin() + (i + 1) * num_anchors * 2);
			detect_decode(data, num_anchors, m_output_shapes[i][2], m_output_shapes[i][3], m_input_shape[2], m_input_shape[3], m_output_shapes[i][1] / 3 - 5,
				m_cfg.conf_threshold, anchors, temp);
		}
		nms(temp, res, m_cfg.iou_threshold);
		box2OriginalSize(res, m_scale);
	}
}

void YOLOV5::postprocess_multi_label(std::vector<Detection>& res)
{
	if (m_output_shapes.size() == 1)
	{
		ov::Tensor output = m_infer_request.get_output_tensor(0);
		float_t* data = output.data<float_t>();
		detect_decode_nms_multi_label(data, res, m_output_shapes[0][1], m_output_shapes[0][2] - 5, m_cfg.conf_threshold, m_cfg.iou_threshold);
		box2OriginalSize(res, m_scale, dw, dh);
	}
	else
	{
		std::cout << "暂不支持插件计算map" << std::endl;
	}
}
