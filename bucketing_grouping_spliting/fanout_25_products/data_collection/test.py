import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys
import ast
from collections import Counter, OrderedDict

def data_collect(filename):
	src_nid_list =[]
	with open(filename) as f:
		for line in f:
			# line = line.strip()
			if line.startswith("before (mini_batch_src_local)  ") :
				_, list_str = line.strip().split("  ")
				list_data = ast.literal_eval(list_str)
				src_nid_list.append(list_data)
    
	print('the length of src list ',len(src_nid_list))
	# print(output_nid_list)
	for src in src_nid_list:
		print(len(src))
		# mini_batch_src_local = list(OrderedDict.fromkeys(src))
		mini_batch_src_local = list(dict.fromkeys(src))
		print(len(mini_batch_src_local))
		print(mini_batch_src_local)
		

	
    
if __name__=='__main__':

	# file = '../get_before_remove_duplicated____.log'
	file = '../get_before_remove_duplicated.log'
	# file = './test.log'
	data_collect(file)	

