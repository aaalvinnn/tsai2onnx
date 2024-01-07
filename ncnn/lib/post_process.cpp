#include "post_process.h"
#include <cmath>

// softmax
// std::vector<float> softmax(const std::vector<float>& input)
// {
//     // 计算指数
//     std::vector<float> exp_values;
//     float max_val = *std::max_element(input.begin(), input.end());
//     float sum_exp = 0.0;

//     for (float val : input) {
//         float exp_val = std::exp(val - max_val);
//         exp_values.push_back(exp_val);
//         sum_exp += exp_val;
//     }

//     // 标准化
//     std::vector<float> softmax_values;
//     for (float exp_val : exp_values) {
//         softmax_values.push_back(exp_val / sum_exp);
//     }

//     return softmax_values;
// }

// 返回softmax输出的最大索引
int get_max_index(ncnn::Mat x)
{
    int index = 0;
    float max = x[0];
    for(int i=0;i<3;i++)
    {
        if(x[i] > max)
        {
            index = i;
            max = x[i];
        }
    }
    
    return index;
}

//这个函数是官方提供的用于打印输出的tensor
void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

float _max(const ncnn::Mat& m)
{
    float max = m[0];
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                if (m[x] > max) max = m[x];
            }
            ptr += m.w;
        }
    }
    return max;
}


// softmax
std::vector<double> softmax(const ncnn::Mat& m)
{
    double total = 0.0;
    float max = _max(m);
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                total += exp(ptr[x] - max);
            }
            ptr += m.w;
        }
    }
    std::vector<double> result;
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                result.push_back(exp(ptr[x] - max) / total);
            }
            ptr += m.w;
        }
    }
    return result;
}
