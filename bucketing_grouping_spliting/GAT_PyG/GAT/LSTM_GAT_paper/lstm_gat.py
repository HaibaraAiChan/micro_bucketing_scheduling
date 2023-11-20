import torch
import torch.nn as nn
import dgl
from dgl.nn import GATConv
import random
import numpy as np
import time
import argparse
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../../pytorch/utils/')
sys.path.insert(0,'../../pytorch/micro_batch_train/')
sys.path.insert(0,'../../pytorch/models/')
from load_graph import load_reddit, inductive_split, load_cora, load_karate, prepare_data, load_pubmed
from load_graph import load_ogbn_dataset, load_ogb
from block_dataloader import generate_dataloader_block
import pickle



def load_block_subtensor(nfeat, labels, blocks, device,args):
	"""
	Extracts features and labels for a subset of nodes
	"""

	# if args.GPUmem:
		# see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = nfeat[blocks[0].srcdata[dgl.NID]].to(device)
	# if args.GPUmem:
		# see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = labels[blocks[-1].dstdata[dgl.NID]].to(device)
	# if args.GPUmem:
		# see_memory_usage("----------------------------------------after  batch labels to device")
	return batch_inputs, batch_labels


# Step 1: Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        print('lstm input x', x)
        out, _ = self.lstm(x)
        return out[:, -1, :] 

# Step 2: Define the GAT Model
class GATModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GATModel, self).__init__()
        self.gat_layer = GATConv(input_size, hidden_size, num_heads=1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, blocks, node_features):
        for i, block in enumerate(blocks):
            h = self.gat_layer(block, node_features)
            h = h.mean(1) 
        return self.fc(h)

# Step 3: Training Loop
def train_model(lstm_model, gat_model, graph, node_features, labels, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        lstm_output = lstm_model(node_features)
        gat_output = gat_model(graph, lstm_output)

        loss = criterion(gat_output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#### Entry point
def run(args, device, data):
	
	# Unpack data
	g, nfeats, labels, n_classes, train_nid, val_nid, test_nid = data
	in_feats = len(nfeats[0])
	print('in feats: ', in_feats)
	
	if args.selection_method =='metis':
		args.o_graph = dgl.node_subgraph(g, train_nid)


	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])
	full_batch_size = len(train_nid)
	

	args.num_workers = 0
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		sampler,
		# device='cpu',
		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	lstm_model = LSTMModel(in_feats, args.num_hidden, args.num_layers)
	gat_model = GATModel(args.num_hidden, args.num_hidden, n_classes)
	optimizer = torch.optim.Adam(list(lstm_model.parameters()) + list(gat_model.parameters()), lr=args.lr)
					
	loss_fcn = nn.CrossEntropyLoss()
	

	for epoch in range(args.num_epochs):
		if args.load_full_batch:
			full_batch_dataloader=[]
			file_name=r'./../../../dataset/fan_out_'+args.fan_out+'/'+args.dataset+'_'+str(epoch)+'_items.pickle'
			with open(file_name, 'rb') as handle:
				item=pickle.load(handle)
				full_batch_dataloader.append(item)
	
		block_dataloader, weights_list, time_collection = generate_dataloader_block(g, full_batch_dataloader, args)
		pseudo_mini_loss = torch.tensor([], dtype=torch.long)
		num_input_nids=0
		for step, (input_nodes, seeds, blocks) in enumerate(block_dataloader):
			print('step ', step)
			num_input_nids	+= len(input_nodes)
			batch_inputs, batch_labels = load_block_subtensor(nfeats, labels, blocks, device,args)
			blocks = [block.int().to(device) for block in blocks]#------------*
				
			print('batch_inputs size ', batch_inputs.size())
			lstm_output = lstm_model(batch_inputs)
			gat_output = gat_model(blocks, lstm_output)

			loss = loss_fcn(gat_output, batch_labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			print('loss ', loss)






# train_model(lstm_model, gat_model, graph, node_features, labels, optimizer, criterion, num_epochs)



def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)





def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	# argparser.add_argument('--dataset', type=str, default='ogbn-products')
	argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--aggre', type=str, default='mean')
	# argparser.add_argument('--selection-method', type=str, default='range')
	# argparser.add_argument('--selection-method', type=str, default='random')
	# argparser.add_argument('--selection-method', type=str, default='metis')
	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--num-batch', type=int, default=1)
	argparser.add_argument('--batch-size', type=int, default=0)

	argparser.add_argument('--re-partition-method', type=str, default='REG')
	# argparser.add_argument('--re-partition-method', type=str, default='random')
	argparser.add_argument('--num-re-partition', type=int, default=0)

	# argparser.add_argument('--balanced_init_ratio', type=float, default=0.2)
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=10)

	# argparser.add_argument('--num-hidden', type=int, default=512)
	argparser.add_argument('--num-hidden', type=int, default=1024)

	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')
	
	argparser.add_argument('--log-indent', type=float, default=0)
#--------------------------------------------------------------------------------------
	

	argparser.add_argument('--lr', type=float, default=5e-3)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=4,
		help="Number of sampling processes. Use 0 for no extra process.")
	

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	# if args.GPUmem:
		# see_memory_usage("-----------------------------------------before load data ")
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='pubmed':
		g, n_classes = load_pubmed()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset == 'ogbn-arxiv':
		data = load_ogbn_dataset(args.dataset,  args)
		device = "cuda:0"

	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset,args)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		# data = prepare_data_mag(device, args)
		data = load_ogbn_mag(args)
		device = "cuda:0"
		# run_mag(args, device, data)
		# return
	else:
		raise Exception('unknown dataset')
		
	
	best_test = run(args, device, data)
	

if __name__=='__main__':
	main()
