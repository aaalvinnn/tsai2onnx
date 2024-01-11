## 工作目录说明
```
-ncnn: ncnn推理工作路径
--data: cpp测试集
--lib: 一些预处理和后处理函数
--models: ncnn模型
--ncnn：ncnn框架依赖，已经编译和取出必要库了
--tools: 
----npy2txt:numpy数组转txt脚本
----pnnx: .pt模型转换ncnn模型工具
--main.cpp: 主函数
```

## Steps
1. 编译
```
mkdir build
cd build
cmake ..
make -j8
```
2. 测试
```
cd build
./main
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