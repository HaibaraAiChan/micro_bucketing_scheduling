import numpy
import dgl
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../utils/')
# sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/bucketing')
# sys.path.insert(0,'/home/cc/Betty_baseline/pytorch/utils')
from numpy.core.numeric import Infinity
import multiprocessing as mp
import torch
import time
from statistics import mean
from my_utils import *
import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import cupy as cp

from collections import Counter
from math import ceil
from cpu_mem_usage import get_memory
from my_utils import torch_is_in_1d
import pdb


from gen_K_hop_neighbors import generate_K_hop_neighbors
from grouping_float import grouping_fanout_1, grouping_fanout_arxiv

def print_(list_):
    for ll in list_:
        print('length ', len(ll))
        print(ll )
        print()

def get_sum(list_idx, mem):
    res=0
    # print(mem)
    print(list_idx)
    for idx in list_idx:
        # print(idx)
        temp = mem[idx] 
        res += temp
    return res

def asnumpy(input):
	return input.cpu().detach().numpy()

def equal (x,y):
	return x == y

def nonzero_1d(input):
	x = torch.nonzero(input, as_tuple=False).squeeze()
	return x if x.dim() == 1 else x.view(-1)

def gather_row(data, row_index):
	return torch.index_select(data, 0, row_index.long())

def zerocopy_from_numpy(np_array):
	return torch.as_tensor(np_array)

def my_sort_1d(val):  # add new function here, to replace torch.sort()
		idx_dict = dict(zip(range(len(val)),val.tolist())) #####
		sorted_res = dict(sorted(idx_dict.items(), key=lambda item: item[1])) ######
		sorted_val = torch.tensor(list(sorted_res.values())).to(val.device)  ######
		idx = torch.tensor(list(sorted_res.keys())).to(val.device) ######
		return sorted_val, idx

class Bucket_Partitioner:  # ----------------------*** split the output layer block ***---------------------
	def __init__(self, layer_block, args, full_batch_dataloader):
		# self.balanced_init_ratio=args.balanced_init_ratio
		self.memory_constraint = args.mem_constraint
		self.dataset=args.dataset
		self.layer_block=layer_block # local graph with global nodes indices
		self.local=False
		self.output_nids=layer_block.dstdata['_ID'] # tensor type
		self.local_output_nids=[]
		self.local_src_nids=[]
		self.src_nids_tensor= layer_block.srcdata['_ID']
		self.src_nids_list= layer_block.srcdata['_ID'].tolist()
		self.full_src_len=len(layer_block.srcdata['_ID'])
		self.global_batched_seeds_list=[]
		self.local_batched_seeds_list=[]
		self.weights_list=[]
		# self.alpha=args.alpha
		# self.walkterm=args.walkterm
		self.num_batch=args.num_batch
		self.selection_method=args.selection_method
		self.batch_size=0
		self.ideal_partition_size=0

		# self.bit_dict={}
		self.side=0
		self.partition_nodes_list=[]
		self.partition_len_list=[]

		self.time_dict={}
		self.red_before=[]
		self.red_after=[]
		self.args=args
		self.full_batch_dataloader = full_batch_dataloader
		
		self.in_degrees = self.layer_block.in_degrees()
		self.K = args.num_batch

	



	def _bucketing(self, val):
		# val : local index degrees 
		# sorted_val, idx = torch.sort(val)
		# print('degrees val')
		# print(val)
		sorted_val, idx = val.sort(stable=True)
		# print('sorted_val ', sorted_val)
		# print('idx ', idx)
		# sorted_val, idx = my_sort_1d(val) # keep the nodes in global order

		unique_val = asnumpy(torch.unique(sorted_val))
		bkt_idx = []
		for v in unique_val:
			bool_idx = (sorted_val == v)
			eqidx = torch.nonzero(bool_idx, as_tuple=False).squeeze().view(-1)
			# eqidx = nonzero_1d(equal(sorted_val, v))
			# bkt_idx.append(gather_row(idx, eqidx))
			local_nids = torch.index_select(idx, 0, eqidx.long())
			bkt_idx.append(local_nids)
			
		def bucketor(data):
			bkts = [gather_row(data, idx) for idx in bkt_idx]
			return bkts
		return unique_val, bucketor

	def get_in_degree_bucketing(self):
		
		degs = self.layer_block.in_degrees()
		# print('dst global nid ', self.layer_block.dstdata['_ID'])
		# print('corresponding in degs', degs)
		nodes = self.layer_block.dstnodes() # local dst nid
		
		# degree bucketing
		unique_degs, bucketor = self._bucketing(degs)
		bkt_nodes = []
		for deg, node_bkt in zip(unique_degs, bucketor(nodes)):
			if deg == 0:
				# skip reduce function for zero-degree nodes
				continue
			bkt_nodes.append(node_bkt) # local nid idx
			print('len(bkt) ', len(node_bkt))

		return bkt_nodes  # local nid idx
	


	def get_src(self, seeds):
		in_ids=list(self.layer_block.in_edges(seeds))[0].tolist()
		src= list(set(in_ids+seeds))
		return src

	def get_nids_by_degree_bucket_ID(self, bucket_lists, bkt_dst_nodes_list):
		res=[]
		# print('bucket_lists ', bucket_lists)
		for bucket_l in bucket_lists:
			# print('bucket_l ', bucket_l)
			temp =[]
			for b in bucket_l:
				# print('b ', b)
				# print('bkt_dst_nodes_list[b] ', len(bkt_dst_nodes_list[b]))
				temp.append(bkt_dst_nodes_list[b])
			flattened_list = [element for sublist in temp for element in sublist]
			res.append(torch.tensor(flattened_list, dtype=torch.long))

		return res
	
	def gen_batches_seeds_list(self, bkt_dst_nodes_list):
		print('---||--'*20)
		if "bucketing" in self.selection_method :
			total_len = len(bkt_dst_nodes_list)
			tensor_lengths = [t.numel() for t in bkt_dst_nodes_list]
			print('')
			if 'fanout' in self.selection_method :
				print(len(bkt_dst_nodes_list))
				batches_nid_list = [t for t in bkt_dst_nodes_list]
				print(len(batches_nid_list))
				length = len(self.output_nids)
				self.weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
				self.local_batched_seeds_list = batches_nid_list
				print(self.weights_list)
				return
			if '50_backpack_' in self.selection_method:
				fanout_dst_nids = bkt_dst_nodes_list[-1]
				fanout = len(bkt_dst_nodes_list)
				if self.args.num_batch >= 1:
					fanout_batch_size = ceil(len(fanout_dst_nids)/(self.args.num_batch))
				indices = torch.arange(0,len(fanout_dst_nids)).long()
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
				src_list, weights_list, time_collection = generate_K_hop_neighbors(self.full_batch_dataloader, self.args, batches_nid_list)
				redundant_ratio = []
				for (i, input_nodes) in enumerate(src_list):
					print(len(input_nodes)/len(batches_nid_list[i])/fanout/0.411)
					redundant_ratio.append(len(input_nodes)/len(batches_nid_list[i])/fanout/0.411)
				print('the split redundant ratio ', )
				print(redundant_ratio)
				print(len(redundant_ratio))
				capacity = self.memory_constraint-mean(redundant_ratio)*55.22/self.args.num_batch
				print('capacity: ', capacity)
				return
				adjust =1000
				# estimated_mem = Estimate_MEM(bkt_dst_nodes_list)
				estimated_mem = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881, 0.6416539241286929, 0.6823511784373357, 0.666389745486164, 0.7496792492248849, 0.7371837931190246, 0.7577242599083827, 0.7889046908693763, 0.8683255342292655, 0.9311795745279405, 0.8477295250909833, 0.9436967117287708, 0.9945587138174034, 1.0309573992937635, 1.0749793136129961, 1.0747561831684673, 1.1274098691910925,1.2304586825034851, 1.1488268197006972, 1.3300050600793791, 1.2305013597063668, 1.339544299635952, 1.363191539881995, 1.501307503974184, 1.4590092047286807, 1.473764838436366]
				Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_1(adjust, estimated_mem, capacity=1.7)
				batches_nid_list =  batches_nid_list + G_BUCKET_ID_list
				self.weights_list = weights_list
				self.local_batched_seeds_list = batches_nid_list
			if '25_backpack_' in self.selection_method:
				# print_(bkt_dst_nodes_list)
				# print()
				time_split_start = time.time()
				fanout_dst_nids = bkt_dst_nodes_list[-1]
				fanout = len(bkt_dst_nodes_list)
				while(True):
					if self.args.num_batch >= 1:
						fanout_batch_size = ceil(len(fanout_dst_nids)/(self.K))
					indices = torch.arange(0,len(fanout_dst_nids)).long()
					map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
					# print('map_output_list')
					# print(map_output_list)
					split_batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
					# print('split_batches_nid_list')
					# print(split_batches_nid_list)
					ct = time.time()
					src_list, weights_list, time_collection = generate_K_hop_neighbors(self.full_batch_dataloader, self.args, split_batches_nid_list)
					print('generate_K_hop_neighbors time ', time.time()-ct)
					redundant_ratio = []
					for (i, input_nodes) in enumerate(src_list):
						print(len(input_nodes)/len(split_batches_nid_list[i])/fanout/0.411)
						redundant_ratio.append(len(input_nodes)/len(split_batches_nid_list[i])/fanout/0.411)
					print('the split redundant ratio ', )
					print(redundant_ratio)
					print(len(redundant_ratio))
					if self.memory_constraint <= max(redundant_ratio)*30.85/self.K:
						self.K = self.K + 1
					else:
						
						break
				
				capacity_est = self.memory_constraint-max(redundant_ratio)*30.85/self.K
				print('average memory estimation for each split fanout bucket batch', mean(redundant_ratio)*30.85/self.K)
				print('capacity est: ', capacity_est)
				time_split_end = time.time()
				print('split fanout degree bucket spend /sec: ', time_split_end - time_split_start)
				adjust =1000
				# estimated_mem = Estimate_MEM(bkt_dst_nodes_list)
				# estimated_mem = [0.03173831570355189, 0.05356346886621881, 0.04711123806907531, 0.07226774258244979, 0.09551787968431964, 0.1325150206414136, 0.16567610820741147, 0.18105303097399883, 0.2113949286027087, 0.2532854815945029, 0.28107834893923545, 0.2823118815649026, 0.33190986587898375, 0.3619426326234686, 0.3814523874739127, 0.3890698973198667, 0.40976734184772345, 0.4268743659042759, 0.4544031402175385, 0.44849912694596744, 0.49506335349881553, 0.5441759409753638, 0.5855775860088747, 0.5946814550216808]
				estimated_mem = [0.031600323026579925, 0.053446445057834434, 0.04691033726707499, 0.07212925883696267, 0.0954132446010461, 0.13250813817436047, 0.16562827234049787, 0.18126462923828512, 0.21130672298992675, 0.25300076929852366, 0.2809490893635299, 0.28129312471449885, 0.33190986587898375, 0.36230173630435075, 0.3834405979819673, 0.38852240658495635, 0.4104866247767621, 0.427057239492208, 0.45594087203866557, 0.4482479429953582, 0.494359802184077, 0.5455698065359045, 0.5838345744003708, 0.5952225418284881]
	
				print('sum(estimated_mem)')
				print(sum(estimated_mem))
				print(len(estimated_mem))
				# capacity_imp = min(max(estimated_mem), capacity_est)
				# capacity_imp = max(estimated_mem) + 0.01
				capacity_imp = 0.59523
				time_backpack_start = time.time()
				capacity_imp = 0.8
				if max(estimated_mem) > capacity_imp:
					print('max degree bucket (1-fanout-1) >capacity')
					print('we can reschedule split K-->K+1 ')
					self.K = self.K + 1

				Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_1(adjust, estimated_mem, capacity = capacity_imp)
				print("G_BUCKET_ID_list" , G_BUCKET_ID_list)
				# G_BUCKET_ID_list = [[0]]
				print(G_BUCKET_ID_list)
				# return
				print("G_BUCKET_ID_list length" , len(G_BUCKET_ID_list))
				g_bucket_nids_list=self.get_nids_by_degree_bucket_ID(G_BUCKET_ID_list, bkt_dst_nodes_list)
				# print(g_bucket_nids_list)
				# return 
				time_backpack_end = time.time()
				print('backpack scheduling spend ', time_backpack_end-time_backpack_start)

				print('split_batches_nid_list ', len(split_batches_nid_list))
				time_batch_gen_start = time.time()
				if len(split_batches_nid_list)>=len(g_bucket_nids_list):
					print('estimated_mem ', estimated_mem)
					for j in range(len(g_bucket_nids_list)):
						tensor_group = torch.tensor(g_bucket_nids_list[j], dtype=torch.long)
						current_group_mem = get_sum(G_BUCKET_ID_list[j], estimated_mem)
						print("current group_mem ", current_group_mem)
						
						split_batches_nid_list[j] = torch.cat((split_batches_nid_list[j],tensor_group))
						print('the new estimated memory cost(GB) for batch '+ str(j)+ " "+str(current_group_mem+ mean(redundant_ratio)*30.85/self.K))

				else:
					print("length of split fanout is smaller than len of group bucket_nids_list")
					print('-*_error_'*20)
					print()
					return
				print('the length of split_batches_nid_list ', len(split_batches_nid_list))
				# return
				time_batch_gen_end = time.time()
				print('batches output list generation spend ', time_batch_gen_end-time_batch_gen_start)
				self.weights_list = weights_list
				# split_batches_nid_list = g_bucket_nids_list #################
				self.local_batched_seeds_list = split_batches_nid_list
				return
			if 'arxiv_' in self.selection_method:
				
				print('memory_constraint: ', self.memory_constraint)
				
				adjust =1000
				estimated_mem = [3.6379471541090243, 3.519859292701475, 1.3139044948856264, 1.168023258447647, 1.1110345689631933, 1.0174509148766198, 0.939528083329737, 0.871675177344254, 0.8272551951869841, 0.7875044051387211, 0.7164326450037946, 0.7157693082792709, 0.6781556913286547, 0.6342095800024695, 0.623858655895532, 0.5609127309921086, 0.5670484961134443, 0.5411179518397851, 0.5307433498923407, 0.5061348172368614, 0.5054305008764547, 0.4282287766997373, 0.4350062972128945, 0.43397694928305497, 13.2]

				print('sum(estimated_mem)')
				print(sum(estimated_mem))
				print(len(estimated_mem))
				
				time_backpack_start = time.time()
				capacity_imp = self.memory_constraint
				if max(estimated_mem) > capacity_imp:
					print('max degree bucket (1-fanout-1) >capacity')
					print('we can reschedule split K-->K+1 ')
					self.K = self.K + 1
				print('self.K ', self.K)
				Groups_mem_list, G_BUCKET_ID_list = grouping_fanout_arxiv(adjust, estimated_mem, capacity = capacity_imp)
				print("G_BUCKET_ID_list" , G_BUCKET_ID_list)
				print("Groups_mem_list ", Groups_mem_list)
				
				print("G_BUCKET_ID_list length" , len(G_BUCKET_ID_list))
				g_bucket_nids_list=self.get_nids_by_degree_bucket_ID(G_BUCKET_ID_list, bkt_dst_nodes_list)
				
				time_backpack_end = time.time()
				print('backpack scheduling spend ', time_backpack_end-time_backpack_start)
				split_batches_nid_list =[]
				
				time_batch_gen_start = time.time()
				for j in range(len(g_bucket_nids_list)):
					tensor_group = torch.tensor(g_bucket_nids_list[j], dtype=torch.long)
					current_group_mem = get_sum(G_BUCKET_ID_list[j], estimated_mem)
					print("current group_mem ", current_group_mem)
					
					split_batches_nid_list.append(tensor_group) 
					
				
				
				time_batch_gen_end = time.time()
				print('batches output list generation spend ', time_batch_gen_end-time_batch_gen_start)
				length = len(self.output_nids)
				self.weights_list = [len(batch_nids)/length  for batch_nids in split_batches_nid_list]
				print('self.weights_list ', self.weights_list)
				
				self.local_batched_seeds_list = split_batches_nid_list
				return
						
			elif '25_group_' in self.selection_method:
				print('__ ')
				print(len(bkt_dst_nodes_list))

				print('group 1 start =========================')
				group1 = bkt_dst_nodes_list[:-1]
				group1 = torch.cat(group1)
				print()
				fanout_dst_nids = bkt_dst_nodes_list[-1]
				
				if self.args.num_batch > 1:
					fanout_batch_size = ceil(len(fanout_dst_nids)/(self.args.num_batch-1))
				indices = torch.arange(0,len(fanout_dst_nids)).long()
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
				
				batches_nid_list.insert(0, group1)
				

				print(len(batches_nid_list))
				length = len(self.output_nids)
				print('length ',length)
				print('group1 ', len(group1))

				self.weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
				self.local_batched_seeds_list = batches_nid_list
				print(self.weights_list)

				return
			
			elif '50_group_' in self.selection_method:
				print('__ ')
				print(len(bkt_dst_nodes_list))
				# for ii in range(len(bkt_dst_nodes_list)):
				# 	print(bkt_dst_nodes_list[ii][:10])
				print('group 1 start =========================')
				group1 = bkt_dst_nodes_list[:40]
				# for ii in range(len(group1)):
				# 	print(group1[ii][:10])
				group1 = torch.cat(group1)
				print()
				print('group 2 start=========================')
				group2 =  bkt_dst_nodes_list[40:49]
				# for ii in range(len(group2)):
				# 	print(group2[ii][:10])
				group2 = torch.cat(group2)
				# print('group 3 start=========================')
				# group3 =  bkt_dst_nodes_list[46:48]
				# group3 = torch.cat(group3)
    
				fanout_dst_nids = bkt_dst_nodes_list[-1]
				
				if self.args.num_batch > 2:
					fanout_batch_size = ceil(len(fanout_dst_nids)/(self.args.num_batch-2))
				indices = torch.arange(0,len(fanout_dst_nids)).long()
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
				# batches_nid_list.insert(0, group3)
				batches_nid_list.insert(0, group2)
				batches_nid_list.insert(0, group1)
				

				print(len(batches_nid_list))
				length = len(self.output_nids)
				print('length ',length)
				print('group1 ', len(group1))
				print('group2 ', len(group2))
				self.weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
				self.local_batched_seeds_list = batches_nid_list
				print(self.weights_list)
				# pdb.set_trace()
				return
			elif '100_group_' in self.selection_method:
				print('__ ')
				print(len(bkt_dst_nodes_list))
				
				print('group 1 start =========================')
				group1 = bkt_dst_nodes_list[:40]
				group1 = torch.cat(group1)
    
				print('group 2 start=========================')
				group2 =  bkt_dst_nodes_list[40:53]
				group2 = torch.cat(group2)
    
				print('group 3 start=========================')
				group3 =  bkt_dst_nodes_list[53:63]
				group3 = torch.cat(group3)

				print('group 4 start=========================')
				group4 =  bkt_dst_nodes_list[63:70]
				group4 = torch.cat(group4)

				print('group 5 start=========================')
				group5 =  bkt_dst_nodes_list[70:76]
				group5 = torch.cat(group5)
    
				print('group 6 start=========================')
				group6 =  bkt_dst_nodes_list[76:81]
				group6 = torch.cat(group6)
				print('group 7 start=========================')
				group7 =  bkt_dst_nodes_list[81:86]
				group7 = torch.cat(group7)
				print('group 8 start=========================')
				group8 =  bkt_dst_nodes_list[86:90]
				group8 = torch.cat(group8)
				print('group 9 start=========================')
				group9 =  bkt_dst_nodes_list[90:93]
				group9 = torch.cat(group9)
				print('group 10 start=========================')
				group10 =  bkt_dst_nodes_list[93:96]
				group10 = torch.cat(group10)    
				print('group 11 start=========================')
				group11 =  bkt_dst_nodes_list[96:99]
				group11 = torch.cat(group11)  
				fanout_dst_nids = bkt_dst_nodes_list[-1]
				
				if self.args.num_batch > 11:
					fanout_batch_size = ceil(len(fanout_dst_nids)/(self.args.num_batch-11))
				indices = torch.arange(0,len(fanout_dst_nids)).long()
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
				batches_nid_list.insert(0, group11)
				batches_nid_list.insert(0, group10)
				batches_nid_list.insert(0, group9)
				batches_nid_list.insert(0, group8)
				batches_nid_list.insert(0, group7)
				batches_nid_list.insert(0, group6)
				batches_nid_list.insert(0, group5)
				batches_nid_list.insert(0, group4)				
				batches_nid_list.insert(0, group3)
				batches_nid_list.insert(0, group2)
				batches_nid_list.insert(0, group1)
				

				print(len(batches_nid_list))
				length = len(self.output_nids)
				print('length ',length)
				print('group1 ', len(group1))
				print('group2 ', len(group2))
				print('group7 ', len(group7))
				self.weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
				self.local_batched_seeds_list = batches_nid_list
				print(self.weights_list)
				# pdb.set_trace()
				return
			
			elif 'custom' in self.selection_method:
				print('custom ')
				print(len(bkt_dst_nodes_list))
				for ii in range(len(bkt_dst_nodes_list)):
					print(bkt_dst_nodes_list[ii][:10])
				print('group 1=========================')
				group1 = bkt_dst_nodes_list[2:14]
				for ii in range(len(group1)):
					print(group1[ii][:10])
				group1 = torch.cat(group1)
				print('group 2=========================')
				group2 = bkt_dst_nodes_list[:2] + bkt_dst_nodes_list[14:24]
				for ii in range(len(group2)):
					print(group2[ii][:10])
				group2 = torch.cat(group2)
				split = bkt_dst_nodes_list[-1]
				num_split = int(len(split)/2)

				split_1 = split[:num_split]
				split_2 = split[num_split:]
				batches_nid_list = [group1,group2,split_1, split_2]

				print(len(batches_nid_list))
				length = len(self.output_nids)
				print('length ',length)
				print('group1 ', len(group1))
				self.weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
				self.local_batched_seeds_list = batches_nid_list
				print(self.weights_list)
				pdb.set_trace()
				return

			if tensor_lengths[-1] > max(tensor_lengths[:-1]):
				fanout_dst_nids = bkt_dst_nodes_list[-1]
				group_nids_list = bkt_dst_nodes_list[:-1]
			else:
				fanout_dst_nids = torch.cat(bkt_dst_nodes_list[int(total_len/2):])
				group_nids_list = bkt_dst_nodes_list[:int(total_len/2)]
			
			if self.args.num_batch <= 1:
				print('no need to split fanout degree, full batch train ')
				self.local_batched_seeds_list = bkt_dst_nodes_list
				return
			
			if self.args.num_batch > 1:
				fanout_batch_size = ceil(len(fanout_dst_nids)/(self.args.num_batch-1))
				# args.batch_size = batch_size
		

			if 'random' in self.selection_method:
				# print('before  shuffle ', fanout_dst_nids)
				indices = torch.randperm(len(fanout_dst_nids))
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				# print('after shuffle ', map_output_list)
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
			
    
			if 'range' in self.selection_method:   
				indices = torch.arange(0,len(fanout_dst_nids)).long()
				map_output_list = fanout_dst_nids.view(-1)[indices].view(fanout_dst_nids.size())
				batches_nid_list = [map_output_list[i:i + fanout_batch_size] for i in range(0, len(map_output_list), fanout_batch_size)]
			
			
			if len(group_nids_list) == 1 :
				batches_nid_list.insert(0, group_nids_list[0])
			else:
				group_tensor = torch.cat(group_nids_list)
				group_tensor_increase, _ = torch.sort(group_tensor)
				batches_nid_list.insert(0, group_tensor_increase)
			
			length = len(self.output_nids)
			self.weights_list = [len(batch_nids)/length  for batch_nids in batches_nid_list]
			# print('batches_nid_list ', batches_nid_list)
			# print('weights_list ', self.weights_list)
			

			self.local_batched_seeds_list = batches_nid_list
		return


	def get_src_len(self,seeds):
		in_nids=self.layer_block.in_edges(seeds)[0]
		src =torch.unique(in_nids)
		return src.size()



	def get_partition_src_len_list(self):
		partition_src_len_list=[]
		for seeds_nids in self.local_batched_seeds_list:
			partition_src_len_list.append(self.get_src_len(seeds_nids))

		self.partition_src_len_list=partition_src_len_list
		self.partition_len_list=partition_src_len_list
		return 


	def buckets_partition(self):
		t1 = time.time()
		bkt_dst_nodes_list = self.get_in_degree_bucketing()
		t2 = time.time()
		self.gen_batches_seeds_list(bkt_dst_nodes_list)
		t3 = time.time()
		print('bkt_dst_nodes_list = self.get_in_degree_bucketing() spend: ', t2-t1)
		print('self.gen_batches_seeds_list(bkt_dst_nodes_list) spend ', t3-t2 )

		# self.get_partition_src_len_list()

		return 



	def global_to_local(self):
		
		sub_in_nids = self.src_nids_list
		# print('src global')
		# print(sub_in_nids)#----------------
		# global_nid_2_local = {sub_in_nids[i]: i for i in range(0, len(sub_in_nids))}
		global_nid_2_local = dict(zip(sub_in_nids,range(len(sub_in_nids))))
		self.local_output_nids = list(map(global_nid_2_local.get, self.output_nids.tolist()))
		# print('dst local')
		# print(self.local_output_nids)#----------------
		# self.local_src_nids = list(map(global_nid_2_local.get, self.src_nids_list))
		
		self.local=True
		
		return

	
	def local_to_global(self):

		# print('src global ', self.src_nids_tensor)

		for local_seed_nids in self.local_batched_seeds_list:
			# print('local nid ', local_seed_nids)
			
			local_all = torch.tensor(list(range(len(self.src_nids_tensor))))
			
			eqidx = nonzero_1d(torch_is_in_1d(local_all, local_seed_nids))
			
			
			after_sort = gather_row(self.src_nids_tensor, eqidx)
				
			
			self.global_batched_seeds_list.append(after_sort)
				
			
		self.local=False
		print('len local_batched_seeds_list ', len(self.local_batched_seeds_list))
		return


	def init_partition(self):
		ts = time.time()
		
		self.global_to_local() # global to local           
		
		t2=time.time()
		print('self.global_to_local() spend sec: ', t2-ts)
		# Then, the graph_parition is run in block to graph local nids,it has no relationship with raw graph
		self.buckets_partition()  # generate  self.local_batched_seeds_list 
		t3=time.time()
		# after that, we transfer the nids of batched output nodes from local to global.
		self.local_to_global() # local to global         self.global_batched_seeds_list
		t_total=time.time()-ts
		print('partition total batch output list spend : ', t_total)
		print('self.buckets_partition() spend  sec: ', t3-t2)
		return self.global_batched_seeds_list, self.weights_list, t_total, self.partition_len_list