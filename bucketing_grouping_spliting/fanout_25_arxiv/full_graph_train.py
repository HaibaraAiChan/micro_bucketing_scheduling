import argparse
import dgl
import dgl.function as fn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 num_layers,
                 dropout):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats, 'lstm', bias=False))
        self.bns.append(nn.BatchNorm1d(hidden_feats))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats, 'lstm', bias=False))
            self.bns.append(nn.BatchNorm1d(hidden_feats))
        # output layer
        self.layers.append(SAGEConv(hidden_feats, out_feats, 'lstm', bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, g, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(g, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](g, x)

        return x.log_softmax(dim=-1)

def train(model, g, feats, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(g, feats)[train_idx]
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, g, feats, y_true, split_idx, evaluator):
    model.eval()

    out = model(g, feats)
    y_pred = out.argmax(dim=-1, keepdim=True)

    # train_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['train']],
    #     'y_pred': y_pred[split_idx['train']],
    # })['acc']
    # valid_acc = evaluator.eval({
    #     'y_true': y_true[split_idx['valid']],
    #     'y_pred': y_pred[split_idx['valid']],
    # })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    # return train_acc, valid_acc, test_acc
    return  test_acc

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GraphSAGE Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=1)
    # parser.add_argument("--eval", action='store_true',
    #                     help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    g, labels = dataset[0]
    feats = g.ndata['feat']
    g = dgl.to_bidirected(g)
    g = g.int().to(device)
    feats, labels = feats.to(device), labels.to(device)
    train_idx = split_idx['train'].to(device)

    model = GraphSAGE(in_feats=feats.size(-1),
                      hidden_feats=args.hidden_channels,
                      out_feats=dataset.num_classes,
                      num_layers=args.num_layers,
                      dropout=args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(model, g, feats, labels, train_idx, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))
            if epoch == 80:
                new_lr = 1e-3  # Set the desired learning rate for the specific epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            # if not args.eval:
            #     continue
            print('epoch ', epoch)
            if epoch % args.log_steps == 0:
                result = test(model, g, feats, labels, split_idx, evaluator)
                # logger.add_result(run, result)
                print('epoch ', epoch)
                
                test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Test: {100 * test_acc:.2f}%')

    #     if args.eval:
    #         logger.print_statistics(run)
    # if args.eval:
    #     logger.print_statistics()


if __name__ == '__main__':
    main()