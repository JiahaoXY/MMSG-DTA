import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Any, Dict, Optional
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.nn import GPSConv, GCNConv, GATConv, GINConv, global_max_pool as gmp, global_mean_pool as gep
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
    """
    This class implements a multi-input attention mechanism.
    It computes attention scores for three different inputs.
    """

    def __init__(self, in_size, hidden_size=64):
        """
        Initializes the Attention module.
        
        Parameters:
            in_size (int): Input feature size.
            hidden_size (int): Hidden layer size for attention projection.
        """
        super(Attention, self).__init__()
        
        # Define layers to project the inputs into a common attention space
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

    def forward(self, x, xt, st):
        """
        Forward pass for the Attention mechanism.
        
        Parameters:
            x (Tensor): Input for first attention part.
            xt (Tensor): Input for second attention part.
            st (Tensor): Input for third attention part.
        
        Returns:
            Tensor: Attention scores.
        """
        # Project inputs to attention space
        x = self.project_x(x)
        xt = self.project_xt(xt)
        st = self.project_st(st)
        
        # Concatenate the projected inputs
        a = torch.cat((x, xt, st), 1)
        
        # Apply softmax to calculate attention weights
        a = torch.softmax(a, dim=1)
        return a


class GraphConv(nn.Module):
    """
    This class implements a graph convolution layer using both local graph convolutions
    and global attention (transformer-style) for node feature propagation.
    """

    def __init__(self, in_channel:int, local_conv: Optional[MessagePassing], heads: int, dropout: float):
        """
        Initializes the GraphConv module.
        
        Parameters:
            in_channel (int): The input feature size of each node.
            local_conv (MessagePassing): A local graph convolution operation (e.g., GCN, GAT).
            heads (int): The number of attention heads for multi-head attention.
            dropout (float): The dropout rate for regularization.
        """
        super().__init__()
        
        self.in_channel = in_channel
        self.local_conv = local_conv
        self.heads = heads
        self.dropout = dropout

        # Multi-head attention layer for global feature aggregation
        self.attn = nn.MultiheadAttention(in_channel, heads, batch_first=True)

        # MLP (Multi-layer perceptron) for final transformation of the node features
        self.mlp = Sequential(
            Linear(in_channel, in_channel * 2),
            ReLU(),
            Dropout(dropout),
            Linear(in_channel * 2, in_channel),
            Dropout(dropout)
        )
        
        # Normalization layers to stabilize learning
        self.norm1 = normalization_resolver('batch_norm', in_channel)
        self.norm2 = normalization_resolver('batch_norm', in_channel)
        self.norm3 = normalization_resolver('batch_norm', in_channel)

    def forward(self, x: Tensor,  edge_index: Adj, batch: Optional[torch.Tensor] = None) -> Tensor:
        """
        Forward pass for the GraphConv layer.
        
        Parameters:
            x (Tensor): Node features.
            edge_index (Adj): Edge indices describing the graph structure.
            batch (Tensor, optional): Batch indices for graph batching.

        Returns:
            Tensor: Transformed node features.
        """
        hs = []
        
        # Apply local graph convolution
        h = self.local_conv(x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Skip connection
        h = self.norm1(h)
        hs.append(h)
        
        # Apply global attention (transformer-style)
        h, mask = to_dense_batch(x, batch)
        h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Skip connection
        h = self.norm2(h)
        hs.append(h)
        
        # Combine both local and global features
        out = sum(hs)
        out = out + self.mlp(out)  # Apply MLP transformation
        out = self.norm3(out)
        
        return out    


class molGraphRepresentation(nn.Module):
    """
    This class handles the graph representation of molecular data.
    It uses multiple layers of graph convolutions to learn molecular features.
    """

    def __init__(self, node_dim, embedding_dim, num_layers, dropout):
        """
        Initializes the molecular graph representation model.
        
        Parameters:
            node_dim (int): The number of input features for each node.
            embedding_dim (int): The dimensionality of the node embedding.
            num_layers (int): The number of graph convolution layers.
            dropout (float): The dropout rate for regularization.
        """
        super().__init__()
        
        self.convs = ModuleList()
        
        # Initial linear layer for the nodes
        self.node_linear = Sequential(
            Linear(node_dim, embedding_dim),
            ReLU(),
            Linear(embedding_dim, embedding_dim),
        )
        
        # Add multiple graph convolution layers (GPSConv + GINConv)
        for _ in range(num_layers):
            nn = Sequential(
                Linear(embedding_dim, embedding_dim * 2),
                ReLU(),
                Linear(embedding_dim * 2, embedding_dim),
            )
            conv = GPSConv(embedding_dim, GINConv(nn), heads=4, dropout=dropout)
            self.convs.append(conv)

        # Final global layer
        self.global_fc = Sequential(
            Linear(embedding_dim, 1024),
            ReLU(),
            Dropout(dropout),
            Linear(1024, embedding_dim),
        )

    def forward(self, data):
        """
        Forward pass for the molecular graph representation.
        
        Parameters:
            data: Data object containing graph structure and node features.
        
        Returns:
            Tensor: Graph-level features after pooling.
        """
        # Apply initial node transformation
        data.x = self.node_linear(data.x)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        
        # Apply global mean pooling to aggregate node-level features to graph-level
        x = gep(x, batch)
        
        # Apply final global layer
        x = self.global_fc(x)
        return x


class proGraphRepresentation(nn.Module):
    """
    This class represents protein graphs and performs graph-based learning on protein structures.
    """

    def __init__(self, num_features_pro, dropout, output_dim):
        """
        Initializes the protein graph representation model.
        
        Parameters:
            num_features_pro (int): The number of input features for each protein node.
            dropout (float): The dropout rate for regularization.
            output_dim (int): The output dimension of the final protein embedding.
        """
        super().__init__()
        
        # Define layers for protein graph convolution
        self.pro_conv = nn.ModuleList([])
        self.pro_conv.append(GCNConv(num_features_pro, num_features_pro * 4))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=4, dropout=dropout, concat=False))
        self.pro_conv.append(GATConv(num_features_pro * 4, num_features_pro * 4, heads=4, dropout=dropout, concat=False))
        
        self.pro_out_feats = num_features_pro * 4
        
        # Linear layers for protein sequence transformation
        self.pro_seq_fc1 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        self.pro_seq_fc2 = nn.Linear(num_features_pro * 4, num_features_pro * 4)
        
        # Bias for protein sequence transformation
        self.pro_bias = nn.Parameter(torch.rand(1, num_features_pro * 4))
        torch.nn.init.uniform_(self.pro_bias, a=-0.2, b=0.2)
        
        # Global layer for protein features
        self.global_fc = Sequential(
            Linear(num_features_pro * 4, 1024),
            ReLU(),
            Dropout(dropout),
            Linear(1024, output_dim),
            Dropout(dropout)
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, data):
        """
        Forward pass for the protein graph representation.
        
        Parameters:
            data: Data object containing graph structure and
        """
        Forward pass for the protein graph representation.
        
        Parameters:
            data: Data object containing graph structure and protein node features.
        
        Returns:
            Tensor: The final protein graph-level representation.
        """
        # Get the protein input features
        x, edge_index, weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        n = x.size(0)
        
        # Apply each convolution layer
        for i in range(len(self.pro_conv)):
            if i == 0:
                xc = self.pro_conv[i](x, edge_index, weight)
            else:
                xc = self.pro_conv[i](x, edge_index)
            
            # Apply ReLU activation to intermediate layers
            if i < len(self.pro_conv) - 1:
                xc = self.relu(xc)
            
            # Skip connection for the first layer
            if i == 0:
                x = xc
                continue
            
            # Protein sequence transformation with gating mechanism
            pro_z = torch.sigmoid(
                self.pro_seq_fc1(xc) + self.pro_seq_fc2(x) + self.pro_bias.expand(n, self.pro_out_feats))
            
            # Combine the transformed features with the original features
            x = pro_z * xc + (1 - pro_z) * x

        # Apply global max pooling to aggregate node-level features into a graph-level feature
        x = gmp(x, batch)
        
        # Pass through the final global layer
        x = self.global_fc(x)
        return x


class Conv1dReLU(nn.Module):
    """
    This class defines a simple 1D convolution layer followed by ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initializes the Conv1dReLU module.
        
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding to apply to the input.
        """
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass for the Conv1dReLU module.
        
        Parameters:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Transformed output after convolution and ReLU activation.
        """
        return self.inc(x)


class LinearReLU(nn.Module):
    """
    This class defines a linear layer followed by ReLU activation.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the LinearReLU module.
        
        Parameters:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include a bias term.
        """
        super().__init__()
        self.inc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass for the LinearReLU module.
        
        Parameters:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output after linear transformation and ReLU activation.
        """
        return self.inc(x)


class StackCNN(nn.Module):
    """
    This class defines a stack of convolutional layers followed by ReLU activation, 
    with an adaptive pooling layer at the end.
    """

    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initializes the StackCNN module.
        
        Parameters:
            layer_num (int): Number of convolution layers in the stack.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for each convolution layer.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Stride of the convolution.
            padding (int): Padding to apply to the input.
        """
        super().__init__()

        # Create a stack of convolution layers
        self.inc = nn.Sequential(OrderedDict([('conv_layer0',
                                               Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1),
                                Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=padding))

        # Adaptive max pooling to reduce the output to a fixed size
        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        """
        Forward pass for the StackCNN module.
        
        Parameters:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output after all convolutional layers and pooling.
        """
        return self.inc(x).squeeze(-1)


class proSequenceRePresentation(nn.Module):
    """
    This class represents the protein sequence as a series of CNN layers.
    It takes the protein sequence as input and applies a series of convolutions.
    """

    def __init__(self, block_num, vocab_size, embedding_num):
        """
        Initializes the protein sequence representation model.
        
        Parameters:
            block_num (int): The number of convolution blocks to apply.
            vocab_size (int): Size of the protein sequence vocabulary (e.g., number of amino acids).
            embedding_num (int): The size of the embedding for each amino acid.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        
        # Create the convolution blocks
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx + 1, embedding_num, 96, 3)
            )

        # Final linear layer to map the features to a fixed size
        self.linear = nn.Linear(3 * 96, 128)
        
    def forward(self, x):
        """
        Forward pass for the protein sequence representation.
        
        Parameters:
            x (Tensor): The protein sequence as input.
        
        Returns:
            Tensor: The protein sequence's learned representation.
        """
        x = self.embed(x).permute(0, 2, 1)  # Embed and reshape sequence
        feats = [block(x) for block in self.block_list]  # Apply each block

        # Concatenate the features from each block
        x = torch.cat(feats, -1)
        
        # Final linear transformation
        x = self.linear(x)

        return x


class MGNNDTA(torch.nn.Module):
    """
    This is the main model for Multimodal Graph Neural Network-based Drug-Target Affinity prediction (MGNNDTA).
    It combines molecular graph representations, protein graph representations, and protein sequences 
    to predict the affinity between a drug and a target.
    """

    def __init__(self, num_features_pro=33, num_features_mol=88, embed_dim=128, dropout=0.2):
        """
        Initializes the MGNNDTA model.
        
        Parameters:
            num_features_pro (int): The number of features in the protein graph.
            num_features_mol (int): The number of features in the molecular graph.
            embed_dim (int): The embedding dimension for node features.
            dropout (float): The dropout rate for regularization.
        """
        super(MGNNDTA, self).__init__()

        print('MGNNDTA Loading ...')
        
        # Initialize encoders for each modality
        self.ligand_encoder = molGraphRepresentation(num_features_mol, num_layers=4, embedding_dim=embed_dim, dropout=dropout)
        self.progra_encoder = proGraphRepresentation(num_features_pro, dropout, embed_dim)
        self.proseq_encoder = proSequenceRePresentation(block_num=3, vocab_size=22, embedding_num=128)
        
        # Attention layer to fuse the outputs of different modalities
        self.attention = Attention(embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Combined fully connected layers to make the final prediction
        self.fc1 = nn.Linear(3 * embed_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, data_mol, data_pro):
        """
        Forward pass for the MGNNDTA model.
        
        Parameters:
            data_mol: Data object containing molecular graph data.
            data_pro: Data object containing protein graph and sequence data.
        
        Returns:
            Tensor: The predicted affinity between the drug and target.
        """
        # Encode molecular graph, protein graph, and protein sequence
        data_pro_seq = data_pro.pro_emb
        mol_x = self.ligand_encoder(data_mol)
        pro_x = self.progra_encoder(data_pro)
        pro_s = self.proseq_encoder(data_pro_seq)
        
        # Compute attention weights and fuse the embeddings
        a = self.attention(mol_x, pro_x, pro_s)
        emb = torch.stack([mol_x, pro_x, pro_s], dim=1)
        a = a.unsqueeze(dim=2)  # Reshape attention scores
        fused_emb = torch.sum(a * emb, dim=1)  # Weighted sum of embeddings
        
        # Apply final fully connected layers
        out = self.fc1(fused_emb)
        out = self.fc2(out)
        out = self.out(out)
        
        return out
