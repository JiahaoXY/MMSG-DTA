import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Dict, Optional
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.nn import GPSConv,GCNConv,GATConv,GINConv,global_max_pool as gmp, global_mean_pool as gep
from collections import OrderedDict
from torch_geometric.nn.resolver import normalization_resolver
from torch_geometric.utils import to_dense_batch

from torch.nn import (
    Linear,
    ModuleList,
    ReLU,
    Sequential,
    Dropout
)

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()
        self.project_x = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xt = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_st = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, x, xt,st):
        x = self.project_x(x)
        xt = self.project_xt(xt)
        st = self.project_st(st)
        a = torch.cat((x, xt, st), 1)
        a = torch.softmax(a, dim=1)
        return a

class GraphConv(nn.Module):
    def __init__(self, in_channel:int, local_conv: Optional[MessagePassing], heads: int, dropout: float):
        super().__init__()
        
        self.in_channel = in_channel
        self.local_conv = local_conv
        self.heads = heads
        self.dropout = dropout
        self.attn = nn.MultiheadAttention(in_channel, heads, batch_first= True)
        self.mlp = Sequential(
            Linear(in_channel, in_channel * 2),
            ReLU(),
            Dropout(dropout),
            Linear(in_channel * 2, in_channel),
            Dropout(dropout)
        )
        
        self.norm1 = normalization_resolver('batch_norm', in_channel)
        self.norm2 = normalization_resolver('batch_norm', in_channel)
        self.norm3 = normalization_resolver('batch_norm', in_channel)
        
    def forward(self, x: Tensor,  edge_index: Adj, batch: Optional[torch.Tensor] = None)->Tensor:
        
        hs = []
        h = self.local_conv(x,edge_index)
        h = F.dropout(h, p=self.dropout, training= self.training)
        h = h + x
        h = self.norm1(h)
        hs.append(h)
        
        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = h[mask]
        h = F.dropout(h,p=self.dropout, training= self.training)
        h = h + x
        h = self.norm2(h)
        hs.append(h)
        
        # Combine local and global outputs.
        out = sum(hs)
        out = out + self.mlp(out)
        out = self.norm3(out)
        
        return out    

class molGraphRepresentation(nn.Module):
    def __init__(self, node_dim, embedding_dim, num_layers,dropout):
        super().__init__()
        self.convs = ModuleList()
        self.node_linear = Sequential(
            Linear(node_dim,embedding_dim),
            ReLU(),
            Linear(embedding_dim,embedding_dim),
        )
        
        for _ in range(num_layers):
            nn = Sequential(
                Linear(embedding_dim, embedding_dim * 2),
                ReLU(),
                Linear(embedding_dim * 2, embedding_dim),
            )
            conv = GPSConv(embedding_dim, GINConv(nn), heads=4, dropout=dropout)
            self.convs.append(conv)

        self.global_fc = Sequential(
            Linear(embedding_dim, 1024),
            ReLU(),
            Dropout(dropout),
            Linear(1024, embedding_dim),
            #Dropout(dropout)
        )
    def forward(self,data):
        data.x = self.node_linear(data.x)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        x = gep(x,batch)
        x = self.global_fc(x)
        return x

class proGraphRepresentation(nn.Module):
    def __init__(self, num_features_pro,dropout,output_dim):
        super().__init__()
        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=4, dropout=dropout, concat=False))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=4, dropout=dropout, concat=False))
        self.pro_out_feats = num_features_pro * 4
        self.pro_seq_fc1 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_seq_fc2 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_bias = nn.Parameter(torch.rand(1, num_features_pro * 4))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)
        self.global_fc = Sequential(
            Linear(num_features_pro * 4, 1024),
            ReLU(),
            Dropout(dropout),
            Linear(1024, output_dim),
            Dropout(dropout)
        )
        self.relu = nn.ReLU()
    
    def forward(self,data):
        # get protein input
        x, edge_index, weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        n = x.size(0)
        for i in range(len(self.pro_conv)):
            if i == 0:
                xc = self.pro_conv[i](x, edge_index, weight)
            else:
                xc = self.pro_conv[i](x, edge_index)
            if i < len(self.pro_conv) - 1:
                xc = self.relu(xc)
            if i == 0:
                x = xc
                continue
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xc) + self.pro_seq_fc2(x) + self.pro_bias.expand(n, self.pro_out_feats))
            x = pro_z * xc + (1 - pro_z) * x

        x = gmp(x, batch)  # global pooling
        # flatten
        x = self.global_fc(x)
        return x
class Conv1dReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class LinearReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)


class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        return self.inc(x).squeeze(-1)


class proSequenceRePresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        self.linear = nn.Linear(3 * 96, 128)
        
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]

        x = torch.cat(feats, -1)
        x = self.linear(x)

        return x

class MGNNDTA(torch.nn.Module):
    def __init__(self,num_features_pro=33, num_features_mol=88, embed_dim=128, dropout=0.2):
        super(MGNNDTA, self).__init__()

        print('MGNNDTA Loading ...')
        
        self.ligand_encoder = molGraphRepresentation(num_features_mol,num_layers = 4, embedding_dim=embed_dim,dropout=dropout)
        self.progra_encoder = proGraphRepresentation(num_features_pro,dropout,embed_dim)
        self.proseq_encoder = proSequenceRePresentation(block_num=3,vocab_size = 22, embedding_num=128)
        
        self.attention = Attention(embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(3 * embed_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, data_mol, data_pro):
        data_pro_seq = data_pro.pro_emb
        mol_x = self.ligand_encoder(data_mol)
        pro_x = self.progra_encoder(data_pro)
        pro_s = self.proseq_encoder(data_pro_seq)
        
        a = self.attention(mol_x, pro_x, pro_s)
        emb = torch.stack([mol_x, pro_x, pro_s], dim=1)
        a = a.unsqueeze(dim=2)
        fused_emb = (a * emb).reshape(-1, 3 * 128)

        xc = self.fc1(fused_emb)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

