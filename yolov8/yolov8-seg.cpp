#include "engine/yolov8_seg.h"
#include "utils/utils.h"

#include <iostream>
#include <sys/stat.h>
#include <fstream>

bool isDirectoryExists(const std::string& folderPath) {
    struct stat info;
    if (stat(folderPath.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

int main(int argc, char** argv)
{
    const char* model_file = argv[1];
    const char* img_dirs = argv[2];
    const char* result_save_dirs = argv[3];

    if (!fileExists(model_file))
    {
        std::cout << "openvino模型文件不存在！" << std::endl;
        return -1;
    }

    Config cfg;
    cfg.model_path = model_file;

    YOLOV8_SEG yolov8_seg(cfg);
    yolov8_seg.init();

    std::vector<std::string> imgnames;
    cv::glob(img_dirs, imgnames);

    float total_time = 0;

    for (int i = 0; i < imgnames.size(); i += 1)
    {
        cv::Mat img = cv::imread(imgnames[i]);

        std::vector<Detection> res;
        std::vector<cv::Mat> masks;

        utils::HostTime t;

        yolov8_seg.Run(img, res, masks);

        float total = t.getUsedTime();

        total_time += total;

        if (!result_save_dirs) continue;

        if (isDirectoryExists(result_save_dirs))
        {
            std::string img_save_path = utils::replacePathAndExtension(std::string(imgnames[i]), result_save_dirs);
            utils::DrawSegmentation(img, res, masks, utils::dataSets::coco80);
            cv::imwrite(img_save_path, img);
        }
        else
        {
            std::cout << "文件保存路径不存在，请修改或创建文件保存路径！" << std::endl;
            return -1;
        }
    }

    std::cout << "avg_FPS: " << 1000.0f / (total_time / imgnames.size()) << std::endl;

    return 0;
}