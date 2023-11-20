import dgl

import numpy as np
import torch

from dgl.data import CoraGraphDataset
	# load cora data
dataset = CoraGraphDataset()
g = dataset[0]

g = dgl.remove_self_loop(g)

num_nodes = g.number_of_nodes()



sequence_length = 3  # Desired sequence length
num_samples = 3  # Number of neighbors to sample for each node
batch_size = 32  # Define batch size (you can adjust this)

# Sample neighbors for each node in the graph
nodes = torch.arange(g.number_of_nodes())
random_walks = dgl.sampling.random_walk(g, nodes, length=sequence_length)
print('random_walk ', random_walks)
# Generate sequences
sequences = []
for walk in random_walks:
    if len(walk) >= sequence_length:
        sequence = walk[:sequence_length]
        sequences.append(sequence)
        print('sequence ', sequence)
print('-----sequences ', sequences)
max_sequence_length = max(len(seq) for seq in sequences)
print('max_sequence_length ', max_sequence_length)
padded_sequences = torch.zeros((len(sequences), max_sequence_length), dtype=torch.long)

for i, seq in enumerate(sequences):
    padded_sequences[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

# Convert sequences and features to PyTorch tensors
sequences_tensor = padded_sequences
node_features_tensor = sequences_tensor





print(sequences_tensor)
print('sequences_tensor size ', sequences_tensor.size())