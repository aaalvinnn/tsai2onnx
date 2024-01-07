#include "data_loader.h"

/* 获取测试集输入 */
std::vector<int> get_input(const std::string& filename)
{
    std::ifstream inputFile(filename.c_str());
    std::vector<int> dataArray;

    if (inputFile.is_open()) {
        
        int value;
        while (inputFile >> value) {
            dataArray.push_back(value);
        }
    }

    return dataArray;
}

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