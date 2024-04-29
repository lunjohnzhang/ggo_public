#pragma once
#include <vector>
#include "common.h"
#include "States.h"
// #include<bits/stdc++.h>

namespace helper {
    std::tuple<double, double> mean_std(std::vector<double> v);
    void divide(std::vector<double> &v, double factor);
    double sum(std::vector<double> v);
    vector<int> longest_common_subpath(
        vector<Path> &paths, int simulation_time);
}