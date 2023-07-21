import src_gen

# Just some dummy data for testing
local_in_edges_tensor_list = [
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
]
global_batched_nids_list = [
    [19, 20, 21],
    [22, 23, 24]
]
induced_src = [25, 26, 27, 28, 29]
eids_global = [30, 31, 32, 33, 34]

# Call the function from the C++ extension
tails_list, eids_list = src_gen.src_gen(local_in_edges_tensor_list, global_batched_nids_list, induced_src, eids_global)

# Print the results
print("Tails List:")
print(tails_list)

print("\nEIDs List:")
print(eids_list)
