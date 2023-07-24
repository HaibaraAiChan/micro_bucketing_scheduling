import gen_src_tails
import numpy as np
def test_gen_src():
    local_in_edges_tensor_list = [  [np.random.randint(0, 10, size=(5,)).tolist() for _ in range(3)] for ii in range(3)]
    global_batched_nids_list = [np.random.randint(0, 10, size=(5,)).tolist() for _ in range(3)]

    induced_src = [np.random.randint(0, 50) for i in range(10)]
    eids_global = [np.random.randint(0, 50) for i in range(10)]
    

    result = gen_src_tails.gen_src_tails(local_in_edges_tensor_list, global_batched_nids_list, induced_src, eids_global)

    eids_list, tails_list = result

    print("EIDS List: ", eids_list)
    print("Tails List: ", tails_list)

if __name__ == "__main__":
    test_gen_src()
