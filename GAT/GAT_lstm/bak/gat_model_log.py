from gatconv import GATConv,GATConv2
from torch import nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, 
                 in_size,
                 aggre,
                 hid_size, 
                 out_size, 
                 heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.aggre = aggre
        self.layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                self.aggre,
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.layers.append(
            GATConv2(
                hid_size * heads[0],
                out_size,
                heads[1],
                self.aggre,
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    # def forward(self, g, inputs):
    #     h = inputs
    #     for i, layer in enumerate(self.layers):
    #         h = layer(g, h)
    #         if i == 1:  # last layer
    #             h = h.mean(1)
    #         else:  # other layer(s)
    #             h = h.flatten(1)
    #     return h
    
    # def forward(self, blocks, x):
    #     h=x
    #     for i, (layer, block) in enumerate(zip(self.layers[:-1], blocks[:-1])):
    #         print('layer ', i)
    #         print('layer name ',layer)
    #         h = layer(block, h)
    #         h = h.flatten(1)
    #     h = self.layers[-1](blocks[-1], h)
    #     h = h.mean(1)
    #     return h
    
    def forward(self, blocks, x):
        h=x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            print('layer ', i)
            print('layer name ',layer)
            h = layer(block, h)
            print('h.size() after h = layer(block, h) ', h.size())
            if i == 1:  # last layer
                h = h.view(h.shape[0]*h.shape[1], h.shape[2])
                # h = h.mean(1)
                print('h.size() after h = h.mean(1)', h.size())
            else:
                print('h.size() ', h.size())
                h = h.flatten(1)
                print('after flatten h.size() ', h.size())
        return h