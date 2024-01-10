## 工作目录说明
```
-ncnn: ncnn推理工作路径
--data: cpp测试集
--lib: 一些预处理和后处理函数
--models: ncnn模型
--ncnn：ncnn框架源码，需要编译安装
--tools: numpy数组转txt脚本
--main.cpp: 主函数
```

## Steps
1. 编译
```
cd tsai2onnx/ncnn/ncnn
mkdir build
cd build
cmake .. -DNCNN_BENCHMARK=OFF -DNCNN_VULKAN=OFF
make -j8
make install
```
- vulkan是针对gpu的，如果想要ncnn能调用gpu做推理，那么选项需要打开:DHCNN_VULKAN=ON;
- -DNCNN_BENCHMARK=ON表示在Extractor.extract前向推理时，打印模型信息日志，调试时可以打开；
2. 测试ncnn是否安装成功
```
cd ../examples ../build/examples/squeezenet ../images/256-ncnn.png
output：
(NR_deploy) zsw@ubuntu:~/Projects/tsai2onnx/ncnn/ncnn/examples$ ../build/examples/squeezenet ../images/256-ncnn.png
532 = 0.165951
920 = 0.094098
716 = 0.062193
```
则正常安装

## 推理
`main.cpp` 中有2个主函数(其中一个为`main1.cpp`)，一个是测试所有输入，另一个是测试单个输入。
现有的为测试单个输入,示例：
```
cd ncnn/build
./main ../data/1.txt
```
控制台输出为：
```
output: 0.286551 0.418819 0.294631 
------------------------
pred: 1
```