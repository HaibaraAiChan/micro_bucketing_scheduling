import torch
import dgl
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../utils/')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils/')
import numpy
import time
import pickle
import io
from math import ceil
from math import floor
from math import ceil
from itertools import islice
from statistics import mean
from multiprocessing import Manager, Pool
from multiprocessing import Process, Value, Array


from my_utils import gen_batch_output_list
from memory_usage import see_memory_usage

from sortedcontainers import SortedList, SortedSet, SortedDict
from multiprocessing import Process, Queue
from collections import Counter, OrderedDict
import copy
from typing import Union, Collection
from my_utils import torch_is_in_1d

# import sys
sys.path.insert(0, './pybind')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/pybind')
import remove_values

sys.path.insert(0, './pybind_remove_duplicates')
sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing/pybind_remove_duplicates')
import remove_duplicates





import pdb
from multiprocessing import Pool
values_to_remove = set()

class OrderedCounter(Counter, OrderedDict):
	'Counter that remembers the order elements are first encountered'

	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)
#------------------------------------------------------------------------
# def unique_tensor_item(combined):
# 	uniques, counts = combined.unique(return_counts=True)
# 	return uniques.type(torch.long)




# def get_global_graph_edges_ids_block(raw_graph, block):

# 	edges=block.edges(order='eid', form='all')
# 	edge_src_local = edges[0]
# 	edge_dst_local = edges[1]
# 	# edge_eid_local = edges[2]
# 	induced_src = block.srcdata[dgl.NID]
# 	induced_dst = block.dstdata[dgl.NID]
# 	induced_eid = block.edata[dgl.EID]

# 	raw_src, raw_dst=induced_src[edge_src_local], induced_dst[edge_dst_local]
# 	# raw_src, raw_dst=induced_src[edge_src_local], induced_src[edge_dst_local]

# 	# in homo graph: raw_graph
# 	global_graph_eids_raw = raw_graph.edge_ids(raw_src, raw_dst)
# 	# https://docs.dgl.ai/generated/dgl.DGLGraph.edge_ids.html?highlight=graph%20edge_ids#dgl.DGLGraph.edge_ids
# 	# https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.edge_ids.html#dgl.DGLGraph.edge_ids

# 	return global_graph_eids_raw, (raw_src, raw_dst)





# def unique_tensor(tensor):
# 	_, unique_indices = torch.unique(tensor, return_inverse=True)
# 	unique_indices, _ = torch.unique(unique_indices, return_inverse=True)

# 	# Use the unique indices to get the unique elements in the original order
# 	unique_elements_tensor = tensor[unique_indices]
# 	return unique_elements_tensor

def remove_duplicated_values(my_dict):
	new_dict = {}
	seen_values = set()
	for k, v in my_dict.items():
		if v not in seen_values:
			seen_values.add(v)
			new_dict[k] = v
	return new_dict

# def remove_values_mp(item):
# 	global values_to_remove
# 	if item not in values_to_remove:
# 		return item
# 	else:
# 		return None

def check_connections_block(local_batched_nodes_list, current_layer_block):
	
	res=[]
	print('gen K hop neighbors check_connections_block*********************************')

	induced_src = current_layer_block.srcdata[dgl.NID]
	# induced_dst = current_layer_block.dstdata[dgl.NID]
	eids_global = current_layer_block.edata['_ID']
	
	# src_nid_list = induced_src.tolist()
	
	# dict_nid_2_local = dict(zip(src_nid_list, range(len(src_nid_list)))) # speedup 
	print('len of local_batched_nodes_list ', len(local_batched_nodes_list))
	print()
	# print(type(local_batched_nodes_list[0][0]))
	for step, local_output_nid in enumerate(local_batched_nodes_list):
		if step > 0: break
		print('step ', step)
		# in current layer subgraph, only has src and dst nodes,
		# and src nodes includes dst nodes, src nodes equals dst nodes.
		# if torch.is_tensor(local_output_nid): local_output_nid = local_output_nid.tolist()
		

		# local_output_nid = list(map(dict_nid_2_local.get, output_nid))
		
		local_in_edges_tensor = current_layer_block.in_edges(local_output_nid, form='all')
	
		# print('local_in_edges_tensor ', local_in_edges_tensor)
		mini_batch_src_local= list(local_in_edges_tensor)[0] # local (𝑈,𝑉,𝐸𝐼𝐷);
		
		time11=time.time()
		mini_batch_src_local = list(dict.fromkeys(mini_batch_src_local.tolist())) 
		time22=time.time()
		# print("remove duplicated spend dict ", time22-time11)
		
		
		mini_batch_src_global= induced_src[mini_batch_src_local].tolist() # map local src nid to global.
		mini_batch_dst_global= induced_src[local_output_nid].tolist() # map local src nid to global.

		eid_local_list = list(local_in_edges_tensor)[2] # local (𝑈,𝑉,𝐸𝐼𝐷); 
		global_eid_tensor = eids_global[eid_local_list] # map local eid to global.
		print("len(mini_batch_src_global) ", len(mini_batch_src_global))
		
		time1=time.time()
		r_ = remove_values.remove_values(mini_batch_src_global, mini_batch_dst_global)
		time2=time.time()
		print("len(r_) ", len(r_))
		print("remove values openmp spend ", time2-time1)

		src_nid = torch.tensor(mini_batch_dst_global + r_, dtype=torch.long)
		dst_nid = torch.tensor(mini_batch_dst_global, dtype=torch.long)

		res.append((src_nid, dst_nid, global_eid_tensor))
	print('one layer mini_batch_src_local collection stoped ')
	
	return res # global src nids , dst nids and global  eids



def generate_one_hop_neighbors(layer_block, local_batches_nid_list):

	check_connection_time = []

	t1= time.time()
	batches_temp_res_list = check_connections_block(local_batches_nid_list, layer_block)
	t2 = time.time()
	check_connection_time.append(t2-t1)

	src_list=[]
	dst_list=[]

	for step, (srcnid, dstnid, current_block_global_eid) in enumerate(batches_temp_res_list):
		print('generate_one_hop_neighbors global srcnid ',srcnid)
		print('generate_one_hop_neighbors global dstnid ',dstnid)
		src_list.append(srcnid)
		dst_list.append(dstnid)

	connection_time = sum(check_connection_time)
	

	return src_list, dst_list, (connection_time)




# def gen_grouped_dst_list(prev_layer_blocks):
# 	post_dst=[]
# 	for block in prev_layer_blocks:
# 		src_nids = block.srcdata['_ID']
# 		post_dst.append(src_nids)
# 	return post_dst # return next layer's dst nids(equals prev layer src nids)




def combine_list(list_of_lists):
    
	import itertools
	combined_list = list(itertools.chain(*list_of_lists))
	return combined_list

def cal_weights_list(batched_output_nid_list, len_dst_full):
    return [len(nids)/len_dst_full for nids in batched_output_nid_list]
    

def	generate_K_hop_neighbors(full_block_dataloader, args, local_batched_output_nid_list):
    # batched_output_nid_list can be the whole number of output nids
    # or it equals partial of the output nids
	
	connect_checking_time_list=[]
	
	final_src_list =[]
	for _,(src_full, dst_full, full_blocks) in enumerate(full_block_dataloader):
    
		dst_nids = dst_full
		num_batch=len(local_batched_output_nid_list)
		print(' the number of batches: ', num_batch)
		temp = combine_list(local_batched_output_nid_list )
		print('the ratio of the output nids to be processed: ', len(temp)/len(dst_full))
		weights_list = cal_weights_list(local_batched_output_nid_list, len(dst_full))
		print('K_Hop_neighbor: weights list of these split output nids: ', weights_list)

		for layer_id, layer_block in enumerate(reversed(full_blocks)):
			print('generate K hop neighbors layer id (from bottom to top)), ', layer_id )
			if layer_id == 0:
				src_list, dst_list, time_1 = generate_one_hop_neighbors( layer_block,  local_batched_output_nid_list)
				print('the bottom layer global src list ', src_list)
				print('the bottom layer global dst list ', dst_list)
				final_dst_list=dst_list
				if layer_id==args.num_layers-1:
					final_src_list=src_list
				prev_layer_src = src_list
			else:

				grouped_output_nid_list = prev_layer_src

				num_batch=len(grouped_output_nid_list)
				print('layer ', layer_id)
				print('gen k hop neighbor num of batch ',num_batch )
				src_list, dst_list, time_1 = generate_one_hop_neighbors( layer_block, grouped_output_nid_list)
				print(' the layer '+ str(layer_id) +' global src list '+ str(src_list) )
				print(' the layer '+ str(layer_id) +' global dst list ' + str(dst_list) )
				if layer_id==args.num_layers-1: # if current block is the final block, the src list will be the final src
					final_src_list=src_list
				prev_layer_src = src_list
				
			connection_time = time_1
			connect_checking_time_list.append(connection_time)

	print('generate k hop neighbors ', final_src_list)
	return  final_src_list, weights_list, sum(connect_checking_time_list)


if __name__=='__main__':
	

	my_list = [1, 2, 2, 3, 3, 4, 4, 5, 5]
	new_list_1 = remove_values.remove_values(my_list, [2,3])
	print(new_list_1)
	new_list = remove_duplicates.remove_duplicates(my_list)

	print(new_list)  # Outputs: [1, 2, 3, 4, 5]

