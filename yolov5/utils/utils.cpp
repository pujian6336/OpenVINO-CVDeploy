#include "utils.h"
#include <fstream>

void utils::DrawDetection(cv::Mat& img, const std::vector<Detection>& objects, const std::vector<std::string>& classNames)
{
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::Point bbox_points[1][4];
    const cv::Point* bbox_point0[1] = { bbox_points[0] };
    int num_points[] = { 4 };
    if (!objects.empty())
    {
        for (auto& box : objects)
        {
            color = utils::Colors::color20[box.class_id % 20];

            cv::rectangle(img, cv::Point(box.bbox.left, box.bbox.top), cv::Point(box.bbox.right, box.bbox.bottom), color, 2, cv::LINE_AA);
            cv::String det_info;
            if (classNames.size() != 0)
            {
                det_info = classNames[box.class_id] + " " + cv::format("%.4f", box.conf);
            }
            else
            {
                det_info = cv::format("%i", box.class_id) + " " + cv::format("%.4f", box.conf);
            }
            // 在方框右上角绘制对应类别的底色
            bbox_points[0][0] = cv::Point(box.bbox.left, box.bbox.top);
            bbox_points[0][1] = cv::Point(box.bbox.left + det_info.size() * 11, box.bbox.top);
            bbox_points[0][2] = cv::Point(box.bbox.left + det_info.size() * 11, box.bbox.top - 15);
            bbox_points[0][3] = cv::Point(box.bbox.left, box.bbox.top - 15);
            cv::fillPoly(img, bbox_point0, num_points, 1, color);
            cv::putText(img, det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
}

void utils::DrawSegmentation(cv::Mat& img, const std::vector<Detection>& dets, const std::vector<cv::Mat>& masks, const std::vector<std::string>& classNames)
{
    cv::Point bbox_points[1][4];
    const cv::Point* bbox_point0[1] = { bbox_points[0] };
    int num_points[] = { 4 };

    if (!dets.empty())
    {
        for (size_t i = 0; i < dets.size(); i++)
        {
            cv::Scalar color = utils::Colors::color20[dets[i].class_id % 20];

            cv::Mat mask_bgr;

            cv::cvtColor(masks[i], mask_bgr, cv::COLOR_GRAY2BGR);
            mask_bgr.setTo(color, masks[i]);
            cv::addWeighted(mask_bgr, 0.45, img, 1.0, 0., img);

            cv::rectangle(img, cv::Point(dets[i].bbox.left, dets[i].bbox.top), cv::Point(dets[i].bbox.right, dets[i].bbox.bottom), color, 2, cv::LINE_AA);
            cv::String det_info;
            if (classNames.size() != 0)
            {
                det_info = classNames[dets[i].class_id] + " " + cv::format("%.4f", dets[i].conf);
            }
            else
            {
                det_info = cv::format("%i", dets[i].class_id) + " " + cv::format("%.4f", dets[i].conf);
            }

            // 在方框右上角绘制对应类别的底色
            bbox_points[0][0] = cv::Point(dets[i].bbox.left, dets[i].bbox.top);
            bbox_points[0][1] = cv::Point(dets[i].bbox.left + det_info.size() * 11, dets[i].bbox.top);
            bbox_points[0][2] = cv::Point(dets[i].bbox.left + det_info.size() * 11, dets[i].bbox.top - 15);
            bbox_points[0][3] = cv::Point(dets[i].bbox.left, dets[i].bbox.top - 15);
            cv::fillPoly(img, bbox_point0, num_points, 1, color);
            cv::putText(img, det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
}

void utils::DrawSegmentation(cv::Mat& img, const std::vector<Detection>& dets, const cv::Mat& mask, const std::vector<std::string>& classNames)
{
    cv::Point bbox_points[1][4];
    const cv::Point* bbox_point0[1] = { bbox_points[0] };
    int num_points[] = { 4 };

    if (!dets.empty())
    {
        for (size_t i = 0; i < dets.size(); i++)
        {
            cv::Scalar color = utils::Colors::color20[dets[i].class_id % 20];

            cv::Mat mask_bgr = cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
            float left = std::max(0.f, dets[i].bbox.left);
            float top = std::max(0.f, dets[i].bbox.top);
            float right = std::min((float)img.cols - 1, dets[i].bbox.right);
            float bottom = std::min((float)img.rows - 1, dets[i].bbox.bottom);
            cv::Rect roi(left, top, right - left, bottom - top);
            mask_bgr(roi).setTo(color, mask(roi));
            cv::addWeighted(mask_bgr, 0.45, img, 1.0, 0., img);

            cv::rectangle(img, cv::Point(dets[i].bbox.left, dets[i].bbox.top), cv::Point(dets[i].bbox.right, dets[i].bbox.bottom), color, 2, cv::LINE_AA);
            cv::String det_info;
            if (classNames.size() != 0)
            {
                det_info = classNames[dets[i].class_id] + " " + cv::format("%.4f", dets[i].conf);
            }
            else
            {
                det_info = cv::format("%i", dets[i].class_id) + " " + cv::format("%.4f", dets[i].conf);
            }

            // 在方框右上角绘制对应类别的底色
            bbox_points[0][0] = cv::Point(dets[i].bbox.left, dets[i].bbox.top);
            bbox_points[0][1] = cv::Point(dets[i].bbox.left + det_info.size() * 11, dets[i].bbox.top);
            bbox_points[0][2] = cv::Point(dets[i].bbox.left + det_info.size() * 11, dets[i].bbox.top - 15);
            bbox_points[0][3] = cv::Point(dets[i].bbox.left, dets[i].bbox.top - 15);
            cv::fillPoly(img, bbox_point0, num_points, 1, color);
            cv::putText(img, det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    }
}

utils::HostTime::HostTime()
{
    t1 = std::chrono::high_resolution_clock::now();
}

float utils::HostTime::getUsedTime()
{
    t2 = std::chrono::high_resolution_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.f;
    return time_used;  // ms
}

utils::HostTime::~HostTime() {}

void utils::save_txt(const std::vector<Detection>& objects, const std::string& savePath, cv::Mat& img)
{
    if (objects.empty()) { return; }

    int cur_width = img.cols;
    int cur_height = img.rows;

    std::ofstream file(savePath);
    if (file.is_open())
    {
        for (auto& box : objects)
        {
            float cx, cy, w, h;

            cx = (box.bbox.left + box.bbox.right) / 2 / cur_width;
            cy = (box.bbox.bottom + box.bbox.top) / 2 / cur_height;
            w = (box.bbox.right - box.bbox.left) / cur_width;
            h = (box.bbox.bottom - box.bbox.top) / cur_height;

            file << box.class_id << " " << cx << " " << cy << " " << w << " " << h << " " << box.conf << "\n";
        }
        file.close();
    }

}

void utils::replace_root_extension(std::vector<std::string>& filePath, const std::string& oldPath, const std::string& newPath, const std::string& extension)
{
    std::transform(filePath.begin(), filePath.end(), filePath.begin(), [&](std::string& str)
        {
            size_t pos = str.find(oldPath);
            if (pos != std::string::npos)
            {
                str.replace(pos, oldPath.length(), newPath);
            }

            size_t extensionPos = str.find_last_of(".");
            if (extensionPos != std::string::npos)
            {
                str.replace(extensionPos, str.length() - extensionPos, extension);
            }
            return str;
        });
}

std::string utils::replacePathAndExtension(const std::string& originalPath, const std::string& newPathPart, const std::string& newExtension)
{
    // 使用rfind找到最后一个路径分隔符'\'或'/'的位置
    size_t lastSeparatorPos = std::min(originalPath.rfind("/"), originalPath.rfind("\\"));

    if (lastSeparatorPos == std::string::npos) {
        lastSeparatorPos = 0; // 如果没有找到分隔符，假设整个字符串为文件名
    }
    else {
        ++lastSeparatorPos; // 从分隔符之后开始是文件名
    }

    // 找到文件名中的最后一个'.'来确定后缀的起始位置
    size_t extensionStartPos = originalPath.rfind('.');
    if (extensionStartPos == std::string::npos || extensionStartPos < lastSeparatorPos) {
        extensionStartPos = originalPath.length(); // 如果没有找到'.'，则认为没有后缀
    }

    // 提取文件名（不包括路径和后缀）
    std::string fileName;
    if (!newExtension.empty()) {
        fileName = originalPath.substr(lastSeparatorPos, extensionStartPos - lastSeparatorPos);
        fileName += "." + newExtension;
    }
    else
    {
        fileName = originalPath.substr(lastSeparatorPos, extensionStartPos);
    }
    // 构建新路径
    std::string newPath = newPathPart + "/" + fileName;

    return newPath;
}

