#ifndef _POST_PROCESS_H
#define _POST_PROCESS_H

#include "net.h"


int get_max_index(ncnn::Mat x);
void pretty_print(const ncnn::Mat& m);
std::vector<double> softmax(const ncnn::Mat& m);

#endif