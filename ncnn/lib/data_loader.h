#ifndef _DATA_LOADER_H
#define _DATA_LOADER_H

#include <vector>
#include <fstream>
#include "net.h"

std::vector<int> get_input(const std::string& filename);
ncnn::Mat array2ncnnmat(std::vector<int> x);

#endif