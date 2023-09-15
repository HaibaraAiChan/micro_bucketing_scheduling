from gatconv import GATConv
from torch import nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, 
                 in_size,
                 hid_size, 
                 out_size, 
                 heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                'lstm',
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.layers.append(
            GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                'lstm',
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
    
    def forward(self, blocks, x):
        for i, (layer, block) in enumerate(zip(self.layers[:-1], blocks[:-1])):
            x = layer(block, x)
            h = h.flatten(1)
        h = h.mean(1)
        return h