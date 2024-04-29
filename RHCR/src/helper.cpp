#include "helper.h"
#include <numeric>
#include <algorithm>
#include <math.h>
#include <tuple>

#define R (8417508174513LL)
#define X (165131LL)

#define MOD(x) ((x % R + R) % R)

namespace helper
{
    std::tuple<double, double> mean_std(std::vector<double> v)
    {
        if (v.size() == 0)
            return std::make_tuple(0, 0);
        double sum = helper::sum(v);
        double mean = sum / v.size();

        std::vector<double> diff(v.size());
        std::transform(v.begin(), v.end(), diff.begin(),
                       [mean](double x)
                       { return x - mean; });

        double sq_sum = std::inner_product(diff.begin(), diff.end(),
                                           diff.begin(), 0.0);
        double sigma = sqrt(sq_sum / v.size());
        return std::make_tuple(mean, sigma);
    }

    void divide(std::vector<double> &v, double factor)
    {
        for(int i = 0; i < v.size(); i+=1)
        {
            v[i] /= factor;
        }
    }

    double sum(std::vector<double> v)
    {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        return sum;
    }

    // **** Functions to get the longest common sub-path ****
    // Reference: https://leetcode.com/problems/longest-common-subpath/solutions/1319639/c-easy-solution-using-rolling-hash-with-explanation/

    // Return a tuple of <valid, start_idx, end_idx>, where valid indicates
    // whether there is a path of length L shared by all agents, and start/end
    // idx is the start and end index of the path in path[0].
    std::tuple<bool, int, int> is_valid(int L, std::vector<Path> &paths)
    {
        long long hash = 1;
        for (int i = 0; i < L - 1; ++i)
            // calculates hash of first path of length L in paths[0]
            hash = MOD(hash * X);

        // Map from hash value of a path to the number of the path.
        std::map<long long, int> mark;
        // Map from hash value to the start and end index of the path in path[0]
        std::map<long long, std::tuple<int, int>> marked_paths;
        auto &p0 = paths[0];
        long long v = 0;
        for (int i = 0; i < p0.size(); ++i)
        {
            v = MOD(v * X + p0[i].location); // calculates running hash
            if (i >= L - 1)
            {
                mark[v] = 1; // when L length subpath is found, hash it
                marked_paths[v] = make_tuple(i - L + 1, i);

                // here the value can be negative
                // that's why MOD is defined that way
                // subtract the previous part of hash to include the next part
                v = MOD(v - p0[i - L + 1].location * hash);
            }
        }

        for (int p = 1; p < paths.size(); ++p)
        {
            v = 0;
            // traverse all paths to check if any of the hash value is present
            auto &pth = paths[p];

            for (int i = 0; i < pth.size(); ++i)
            {
                v = MOD(v * X + pth[i].location);
                if (i >= L - 1)
                {
                    // only the hash which is present in all previous paths is
                    // increased
                    if (mark.count(v) > 0 && mark[v] == p)
                    {
                        mark[v] += 1;
                    }
                    // subtract the previous part of hash to include the next
                    // part
                    v = MOD(v - pth[i - L + 1].location * hash);
                }
            }
        }

        for (auto it : mark)
        {
            // a hash that is present in all paths
            if (it.second == paths.size())
            {
                int start_idx, end_idx;
                std::tie(start_idx, end_idx) = marked_paths[it.first];
                return make_tuple(true, start_idx, end_idx);
            }
        }
        return make_tuple(false, -1, -1);
    }

    static bool compare(Path a, Path b)
    {
        return (a.size() < b.size());
    }

    vector<int> longest_common_subpath(vector<Path> &paths, int simulation_time)
    {
        int ans_start, ans_end;
        bool has_valid = false;

        vector<Path> real_paths(paths.size());

        // Remove duplicates
        for (int i = 0; i < paths.size(); i++)
        {
            Path real_path;
            int t = 0;
            while(t < paths[i].size() && t < simulation_time)
            {
                State curr = paths[i][t];
                real_path.emplace_back(curr);
                // Skip waits
                while(t < paths[i].size() &&
                      curr.location == paths[i][t].location)
                    t += 1;
            }
            real_paths[i] = real_path;
        }

        // cout << "Real paths" << endl;
        // for (int i = 0; i < real_paths.size(); i++)
        // {
        //     cout << real_paths[i].size() << endl;
        // }

        // sort paths in increasing order of size
        sort(real_paths.begin(), real_paths.end(), compare);

        int l = 0, r = real_paths[0].size() + 1;
        while (r - l > 1)
        {
            // choose a length for subpath in smallest length path
            int mid = (r + l) / 2;
            bool valid;
            int start_idx, end_idx;
            std::tie(valid, start_idx, end_idx) = is_valid(mid, real_paths);
            if (valid)
            {
                // m will be an ans, but we want the largest subpath
                l = mid;
                ans_start = start_idx;
                ans_end = end_idx;
                has_valid = true;
            }
            else
                r = mid;
        }

        // We only care about the locations.
        vector<int> ans;
        if (has_valid)
        {
            // Construct the longest common subpath
            for (int i = ans_start; i <= ans_end; i++)
                ans.emplace_back(real_paths[0][i].location);
        }
        return ans;
    }

    // **** End Functions to get the longest common sub-path ****


}
