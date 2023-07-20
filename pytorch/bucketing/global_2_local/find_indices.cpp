
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <omp.h>


namespace py = pybind11;

std::vector<std::vector<int>> find_indices(const std::vector<int>& tensor, const std::vector<std::vector<int>>& nested_list) {
    std::unordered_map<int, int> tensor_index;
    for (int i = 0; i < tensor.size(); ++i) {
        tensor_index[tensor[i]] = i;
    }

    std::vector<std::vector<int>> indices(nested_list.size());

    #pragma omp parallel for
    for (int i = 0; i < nested_list.size(); ++i) {
        const auto& sublist = nested_list[i];
        std::vector<int> sublist_indices(sublist.size());

        for (int j = 0; j < sublist.size(); ++j) {
            sublist_indices[j] = tensor_index[sublist[j]];
        }

        indices[i] = sublist_indices;
    }

    return indices;
}

PYBIND11_MODULE(find_indices, m) {
    m.def("find_indices", &find_indices, "A function which finds the indices of a list of items in a tensor");
}