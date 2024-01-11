#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "net.h"
#include "data_loader.h"
#include "post_process.h"

// #define MODEL_RAW   // onnx模型中是否包含正则化层和softmax层
#define MODEL_PATH "../../models/traced_model.ncnn"

/* 生成随机输入 */
std::vector<int> generate_random_input()
{
    // 数组长度
    const int arrayLength = 1500;

    // 生成随机数组
    std::vector<int> randomArray;
    for (int i = 0; i < arrayLength; ++i) {
        randomArray.push_back(std::rand());
    }

    return randomArray;
}

/* 200次推理 */
std::vector<int> infer_200(ncnn::Net &net, const std::string data_input_path, std::vector<int> &labels)
{
    // 推理
    std::vector<int> preds;     // 定义预测值
    for(int i = 0; i < labels.size(); i++)
    {
        ncnn::Extractor extractor = net.create_extractor();     // 初始化ncnn推理器

        std::string x_path = data_input_path + std::to_string(i) + ".txt";
        std::vector<int> x = get_input(x_path);

        ncnn::Mat input = array2ncnnmat(x);     // 把vector<int>转换成ncnn的mat

#ifdef MODEL_RAW
        // 标准化预处理
        const float mean_vals[1] = {-23.61164f};
        // const float norm_vals[1] = {825.4822f};
        const float norm_vals[1] = {0.0012114131594842f};       // ncnn中这个函数的norm方差是实际数学公式中方差norm的倒数!
        input.substract_mean_normalize(mean_vals, norm_vals);
#endif
        // ncnn前向计算
        extractor.input("in0", input);
        ncnn::Mat output;
        extractor.extract("out0", output);
        // pretty_print(output);

#ifdef MODEL_RAW
        // 手动softmax
        // std::vector<double> result = softmax(output);
        // std::cout << result[0] << " " << result[1] << " " << result[2] << std::endl;
#endif

        preds.push_back(get_max_index(output));
    }

    return preds;
}

// 测试测试集的所有输入
int main(){
    // 获取输入
    std::vector<int> labels = get_input("../data/labels.txt");

    // 加载转换模型
    // ncnn::set_log_level(ncnn::LOG_NONE);    // 屏蔽输出日志信息
    ncnn::Net net;
    // net.opt.num_threads=1;
    char param_path[50];
    char bin_path[50];
    strcpy(param_path, MODEL_PATH);
    strcpy(bin_path, MODEL_PATH);

    net.load_param(strcat(param_path, ".param"));
    net.load_model(strcat(bin_path, ".bin"));


    // 获取函数开始执行的时间点
    auto start = std::chrono::high_resolution_clock::now();
    // 推理
    std::vector<int> preds = infer_200(net, "../data/", labels);
    // 获取函数结束执行的时间点
    auto end = std::chrono::high_resolution_clock::now();
    // 计算并打印函数的运行时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    
    // 输出
    // for(int i=0;i<preds.size();i++)
    // {
    //     std::cout << preds[i];
    // }

    // 统计结果
    int acc=0;
    for (int i=0;i<labels.size();i++)
    {
        if(labels[i] == preds[i])
        {
            acc ++;
        }
    }

    std::cout << "acc:" << (float)acc / preds.size() << std::endl ;
    std::cout << "done" << std::endl;
    return 0;
}

// test, 单独测试一个输入
int main1(int argc, char *argv[])
{
    // 处理异常输入
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <../data/N.txt>" << std::endl;
        return 1; // 返回非零值表示错误
    }

    // 获取输入
    std::vector<int> labels = get_input("../../data/labels.txt");

    // 加载转换模型
    // ncnn::set_log_level(ncnn::LOG_NONE);    // 屏蔽输出日志信息
    ncnn::Net net;
    // net.opt.num_threads=1;
    // net.load_param("../models/20240102InceptiontimePlus_zmj-sim-opt-fp16.param");
    // net.load_model("../models/20240102InceptiontimePlus_zmj-sim-opt-fp16.bin");
    net.load_param("../../models/traced_model.ncnn.param");
    net.load_model("../../models/traced_model.ncnn.bin");


    // 获取函数开始执行的时间点
    auto start = std::chrono::high_resolution_clock::now();
    // 一次推理
    ncnn::Extractor extractor = net.create_extractor();     // 初始化ncnn推理器
    std::string x_path = argv[1];
    std::vector<int> x = get_input(x_path);
    ncnn::Mat input = array2ncnnmat(x);     // 把vector<int>转换成ncnn的mat
    // 获取函数结束执行的时间点
    auto end = std::chrono::high_resolution_clock::now();
    // 计算并打印函数的运行时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    // // 标准化预处理
    // const float mean_vals[1] = {-23.61164f};
    // const float norm_vals[1] = {825.4822f};
    // input.substract_mean_normalize(mean_vals, norm_vals);

    // ncnn前向计算
    // extractor.input("TS_1_1_1500", input);
    extractor.input("in0", input);
    ncnn::Mat output;
    // extractor.extract("Leidian", output);
    extractor.extract("out0", output);

    std::cout << "input: " << argv[1] << std::endl;
    std::cout << "output: " ;

    pretty_print(output);
    std::cout << "pred: " << get_max_index(output) << std::endl;
    // std::cout << "\n" << "label: " << labels[std::stoi(argv[1][0])] << std::endl;

    return 0;
}