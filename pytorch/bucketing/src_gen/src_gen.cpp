#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace py = pybind11;

std::vector<long> remove_values(std::vector<long> data, std::vector<long> values_to_remove) {
    std::unordered_set<long> to_remove(values_to_remove.begin(), values_to_remove.end());
    std::vector<std::vector<long>> private_results(omp_get_max_threads());

    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        long value = data[i];
        if (!to_remove.count(value)) {
            private_results[omp_get_thread_num()].push_back(value);
        }
    }

    std::vector<long> result;
    for (auto& private_result : private_results) {
        result.insert(result.end(), private_result.begin(), private_result.end());
    }

    return result;
}

std::pair<std::vector<std::vector<long>>, std::vector<std::vector<long>>> src_gen(const std::vector<std::vector<std::vector<long>>>& local_in_edges_tensor_list, const std::vector<std::vector<long>>& global_batched_nids_list, const std::vector<long>& induced_src, const std::vector<long>& eids_global) {
    std::vector<std::vector<long>> tails_list(local_in_edges_tensor_list.size());
    std::vector<std::vector<long>> eids_list(local_in_edges_tensor_list.size());

    #pragma omp parallel for
    for (size_t i = 0; i < local_in_edges_tensor_list.size(); ++i) {
        const auto& local_in_edges_tensor = local_in_edges_tensor_list[i];
        const auto& global_output_nid = global_batched_nids_list[i];

        auto mini_batch_src_local = local_in_edges_tensor[0];
        std::unordered_set<long> set(mini_batch_src_local.begin(), mini_batch_src_local.end());
        mini_batch_src_local.assign(set.begin(), set.end());

        std::vector<long> mini_batch_src_global;
        for (const auto& src_local : mini_batch_src_local) {
            mini_batch_src_global.push_back(induced_src[src_local]);
        }

        auto eid_local_list = local_in_edges_tensor[2];
        std::vector<long> global_eid_tensor;
        for (const auto& eid_local : eid_local_list) {
            global_eid_tensor.push_back(eids_global[eid_local]);
        }

        auto r_ = remove_values(mini_batch_src_global, global_output_nid);

        #pragma omp critical
        {
            tails_list[i] = r_;
            eids_list[i] = global_eid_tensor;
        }
    }

    return std::make_pair(tails_list, eids_list);
}

PYBIND11_MODULE(src_gen, m) {
    m.def("src_gen", &src_gen, "A function that does something");
}
