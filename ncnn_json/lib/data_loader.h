#ifndef _DATA_LOADER_H
#define _DATA_LOADER_H

#include <vector>
#include <fstream>
#include "net.h"

std::vector<std::vector<int>> get_inputs(const std::string path);
std::vector<int> get_labels(const std::string path);
int output_json(const std::string path, const std::string name, int label);
// std::vector<int> get_input(const std::string& filename);
ncnn::Mat array2ncnnmat(std::vector<int> x);

#endif