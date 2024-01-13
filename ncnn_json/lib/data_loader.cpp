#include "data_loader.h"
#include "json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

std::vector<std::vector<int>> get_inputs(const std::string path)
{
	json j;			// 创建 json 对象
	std::ifstream jfile(path);
	jfile >> j;		// 以文件流形式读取 json 文件
	std::vector<std::vector<int>> datas = j.at("datas");
	return datas;
}

std::vector<int> get_labels(const std::string path)
{
	json j;			// 创建 json 对象
	std::ifstream jfile(path);
	jfile >> j;		// 以文件流形式读取 json 文件
	std::vector<int> labels = j.at("labels");
	return labels;
}

int output_json(const std::string path, const std::string name, int label)
{
    // 创建 JSON 对象
    json j;
    j["label"] = std::to_string(label);
    j["name"] = name;

    // 打开文件流
    std::ofstream file(path);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "Error opening file\n";
        return 1;
    }

    // 将 JSON 对象写入文件
    file << std::setw(4) << j << std::endl;

    // 关闭文件流
    file.close();

    std::cout << "output to json done" << std::endl;

    return 0;
}

// /* 获取测试集输入 */
// std::vector<int> get_input(const std::string& filename)
// {
//     std::ifstream inputFile(filename.c_str());
//     std::vector<int> dataArray;

//     if (inputFile.is_open()) {
        
//         int value;
//         while (inputFile >> value) {
//             dataArray.push_back(value);
//         }
//     }

//     return dataArray;
// }

/* vector<int> -> ncnn::mat */
ncnn::Mat array2ncnnmat(std::vector<int> x)
{
    ncnn::Mat y = ncnn::Mat(x.size(), 1, 1);
    for (int i = 0; i < x.size(); i++)
    {
        y[i] = (float)x[i];
    }

    // 维度变换，以适应模型的输入 (1500, 1) -> (1, 1, 1500))
    y.reshape(1, 1, 1500);

    return y;
}