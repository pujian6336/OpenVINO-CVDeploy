#include "engine/yolov10.h"
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

// 自定义比较函数，用于降序排序
bool compareByConf(const Detection& a, const Detection& b) {
    return a.conf > b.conf;
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
    cfg.conf_threshold = 0.000f;

    YOLOV10 yolov10(cfg);
    yolov10.init();

    std::vector<std::string> imgnames;
    cv::glob(img_dirs, imgnames);

    float total_time = 0;

    for (int i = 0; i < imgnames.size(); i += 1)
    {
        cv::Mat img = cv::imread(imgnames[i]);

        std::vector<Detection> res;

        utils::HostTime t;

        yolov10.preprocess(img);
        yolov10.infer();
        yolov10.postprocess(res);

        float total = t.getUsedTime();

        total_time += total;

        if (res.size() > 300)
        {
            std::sort(res.begin(), res.end(), compareByConf);
            res.resize(300);
        }

        if (!result_save_dirs)
        {
            std::cout << "未设置txt文件保存路径！" << std::endl;
            return -1;
        }

        if (isDirectoryExists(result_save_dirs))
        {
            std::string txt_save_path = utils::replacePathAndExtension(std::string(imgnames[i]), result_save_dirs, "txt");
            utils::save_txt(res, txt_save_path, img);
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