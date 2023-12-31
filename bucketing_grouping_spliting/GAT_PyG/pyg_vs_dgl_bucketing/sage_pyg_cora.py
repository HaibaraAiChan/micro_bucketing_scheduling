import os.path as osp
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Reddit
# from torch_geometric.datasets import ogbn-arxiv
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from memory_usage import see_memory_usage
from torch_geometric.datasets import Planetoid

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'cora')
dataset = Planetoid(root=path, name='Cora')
# target_dataset = 'ogbn-arxiv'

# # This will download the ogbn-arxiv to the 'networks' folder
# dataset = PygNodePropPredDataset(name=target_dataset, root='path')

data = dataset[0]
print('dataset[0] length')
print(len(data.train_mask))

# train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
#                                sizes=[10], batch_size=len(data.train_mask), shuffle=True,
#                             #    num_workers=12)
# # train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
# #                                sizes=[25, 10], batch_size=1024, shuffle=True,
# #                                num_workers=12)
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[30,25,10], batch_size=len(data.train_mask), shuffle=True,
                               num_workers=12)

subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 3

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='lstm'))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='lstm'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='lstm'))

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 2048, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


# @torch.no_grad()
# def test():
#     model.eval()

#     out = model.inference(x)

#     y_true = y.cpu().unsqueeze(-1)
#     y_pred = out.argmax(dim=-1, keepdim=True)

#     results = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

#     return results

t_list = []
for epoch in range(1, 11):
    ts= time.time()
    loss, acc = train(epoch)
    te= time.time()
    see_memory_usage("----------------------------------------after train ")
    epoch_time = te-ts
    t_list.append(epoch_time)
    print('end to end time, ', epoch_time)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    # train_acc, val_acc, test_acc = test()
    # print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
    #       f'Test: {test_acc:.4f}')
print('epoch time list : ', t_list)
print('every epoch time ', sum(t_list[4:])/len(t_list[4:]))