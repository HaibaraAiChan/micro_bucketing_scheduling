#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>
#include <iostream>
#include <chrono>

namespace py = pybind11;

std::vector<int> remove_duplicates(std::vector<int> data) {
    std::unordered_set<int> unique_items;
    std::vector<int> result;

    for (const auto& item : data) {
        if (unique_items.find(item) == unique_items.end()) {
            result.push_back(item);
            unique_items.insert(item);
        }
    }

    return result;
}std::vector<long> remove_values(std::vector<long> data, std::vector<long> values_to_remove) {
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

std::vector<std::vector<long>> gen_tails(const std::vector<std::vector<long>>& src_long_list,
                                        const std::vector<std::vector<long>>& global_batched_nids_list) {
    std::vector<std::vector<long>> tails_list;
    tails_list.reserve(src_long_list.size());

    #pragma omp parallel for
    for (int i = 0; i < src_long_list.size(); ++i) {
        const auto& mini_batch_src_global = src_long_list[i];
        const auto& global_output_nid = global_batched_nids_list[i];

        auto r_ = remove_values(mini_batch_src_global, global_output_nid);

        #pragma omp critical
        tails_list.push_back(r_);
    }

    return tails_list;
}

std::pair<std::unordered_map<int, std::vector<int>>, std::unordered_map<int, std::vector<int>>>
src_gen_eid(const std::vector<std::vector<std::vector<int>>>& local_in_edges_tensor_list,
        const std::vector<std::vector<int>>& global_batched_nids_list,
        const std::vector<int>& induced_src_vec,
        const std::vector<int>& eids_global_vec) {
    auto start_total = std::chrono::high_resolution_clock::now();
    // Generate induced_src and eids_global mapping
    std::unordered_map<int, int> induced_src;
    for (size_t i = 0; i < induced_src_vec.size(); i++) {
        induced_src[i] = induced_src_vec[i];
    }

    std::unordered_map<int, int> eids_global;
    for (size_t i = 0; i < eids_global_vec.size(); i++) {
        eids_global[i] = eids_global_vec[i];
    }

    std::unordered_map<int, std::vector<int>> eids_list;
    std::unordered_map<int, std::vector<int>> src_long_list;

    auto start_parallel = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < local_in_edges_tensor_list.size(); ++i) {
        std::cout << "Thread ID: " << omp_get_thread_num() << "\n"; 
        auto start = std::chrono::high_resolution_clock::now();

        const auto& local_in_edges_tensor = local_in_edges_tensor_list[i];
        std::vector<int> mini_batch_src_local = remove_duplicates(local_in_edges_tensor.at(0));

        std::vector<int> mini_batch_src_global;
        for (const auto& local : mini_batch_src_local) {
            auto it = induced_src.find(local);
            if (it == induced_src.end()) {
                throw std::runtime_error("Key not found in induced_src: " + std::to_string(local));
            }
            mini_batch_src_global.push_back(it->second);
        }

        std::vector<int> eid_local_list = local_in_edges_tensor.at(2);

        std::vector<int> global_eid_tensor;
        for (const auto& local : eid_local_list) {
            auto it = eids_global.find(local);
            if (it == eids_global.end()) {
                throw std::runtime_error("Key not found in eids_global: " + std::to_string(local));
            }
            global_eid_tensor.push_back(it->second);
        }

        eids_list[i] = global_eid_tensor;
        src_long_list[i] = mini_batch_src_global;
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate and print the duration
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    }
    auto end_parallel = std::chrono::high_resolution_clock::now();

    // Calculate and print the duration
    auto duration_para = std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - start_parallel);
    std::cout << "Openmp Time taken by function: " << duration_para.count() << " microseconds" << std::endl;

    auto duration_total = std::chrono::duration_cast<std::chrono::microseconds>(end_parallel - start_total);
    std::cout << "gen-src C++ Time taken by function: " << duration_total.count() << " microseconds" << std::endl;

    return std::make_pair(eids_list, src_long_list);
}

std::pair<std::unordered_map<int, std::vector<int>>, std::unordered_map<int, std::vector<int>>>
src_gen(const std::vector<std::vector<std::vector<int>>>& local_in_edges_tensor_list,
        const std::vector<std::vector<int>>& global_batched_nids_list,
        const std::vector<int>& induced_src_vec,
        const std::vector<int>& eids_global_vec) {

    std::unordered_map<int, std::vector<int>> eids_list;
    std::unordered_map<int, std::vector<int>> src_long_list;
    std::tie(eids_list, src_long_list) = gen_src(local_in_edges_tensor_list,induced_src_vec, eids_global_vec);

    std::cout << "src_long_list " << src_long_list << std::endl;

    std::vector<std::vector<int>> src_long_vec;
    for (const auto& pair : src_long_list) {
        src_long_vec.push_back(pair.second);
    }

    std::vector<std::vector<int>> eids_list_vec;
    for (const auto& pair : eids_list) {
        eids_list_vec.push_back(pair.second);
    }
    // Call gen_tails with src_long_vec and global_batched_nids_list
    std::vector<std::vector<int>> tails_list = gen_tails(src_long_vec, global_batched_nids_list);
        
        
        
        
        
        
}



PYBIND11_MODULE(src_gen, m) {
    m.def("src_gen", &src_gen, py::arg("local_in_edges_tensor_list"),
    py::arg("global_batched_nids_list"), py::arg("induced_src"), py::arg("eids_global"));
}
