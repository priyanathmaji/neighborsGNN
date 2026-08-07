import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GCN2Conv, SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch import Tensor
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix

class H2GCNNeighborhoodConv(MessagePassing):
    """
    Symmetrically normalized adjacency propagation WITHOUT self-loops.
    Computes: D^{-1/2} A D^{-1/2} X
    Strictly isolates ego features from neighbor features per H2GCN literature.
    """
    def __init__(self, **kwargs):
        super().__init__(aggr='add', **kwargs)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        row, col = edge_index
        deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=x.dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

class GAT_Naive(nn.Module):
    """
    Naive GAT baseline (no explicit ego/neighbor separation).
    Standard GATConv with self-loops enabled (attention handles self-information
    implicitly through the self-loop, as in vanilla GAT), plus a simple residual
    skip once shapes align - directly comparable to GCN (Naive) as the
    architecture-only control against GAT-sep.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout))
        else:
            # Input Layer
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
            self.lns.append(nn.LayerNorm(hidden_channels * heads))

            # Hidden Layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))
                self.lns.append(nn.LayerNorm(hidden_channels * heads))

            # Output Layer
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.num_layers == 1:
            return self.convs[0](x, edge_index)

        for i in range(self.num_layers - 1):
            h = self.convs[i](x, edge_index)
            h = self.lns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + x if h.shape == x.shape else h  # residual once dims align, same convention as GCN (Naive)

        return self.convs[-1](x, edge_index)
    
class GAT(nn.Module):
    """
    Upgraded GAT-sep Variant with Multi-Head Attention, Residual Skips,
    Stochastic Attention Dropout, and Individual LayerNorm Tracking.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        self.convs = nn.ModuleList()
        self.ego_lins = nn.ModuleList()
        self.lns = nn.ModuleList()

        if num_layers == 1:
            # 1-layer fallback uses multi-head averaging directly to match output dimensions
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout, add_self_loops=False))
            self.ego_lins.append(nn.Linear(in_channels, out_channels))
        else:
            # Input Layer (Concatenates parallel attention heads)
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, add_self_loops=False))
            self.ego_lins.append(nn.Linear(in_channels, hidden_channels * heads))
            self.lns.append(nn.LayerNorm(hidden_channels * heads))
            
            # Hidden Layers
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout, add_self_loops=False))
                self.ego_lins.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))
                self.lns.append(nn.LayerNorm(hidden_channels * heads))
                
            # Output Layer (Averages attention heads to map down to target classes)
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, add_self_loops=False))
            self.ego_lins.append(nn.Linear(hidden_channels * heads, out_channels))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.num_layers == 1:
            # Residual skip pass for 1-layer configurations
            return self.convs[0](x, edge_index) + self.ego_lins[0](x)
            
        for i in range(self.num_layers - 1):
            x_neighbor = self.convs[i](x, edge_index)
            x_ego = self.ego_lins[i](x)
            
            # Explicit residual skip connection handles heterophily smoothly
            x = x_neighbor + x_ego
            x = self.lns[i](x) # Layer-wise independent normalization
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return self.convs[-1](x, edge_index) + self.ego_lins[-1](x)


class GAT_EgoOnly(nn.Module):
    """Ablation: drops the neighbor/attention branch entirely. Tests how much
    a node's own features alone explain, with zero graph structure."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        # Single linear layer, in -> out, matching GAT's layers=1 case exactly minus the conv
        self.ego_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None):  # edge_index accepted but unused
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.ego_lin(x)

class GAT_NeighborOnly(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, heads=8, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout, add_self_loops=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv(x, edge_index)
    
class GCN(nn.Module):
    """ Standard Graph Convolutional Network baseline (Platonov-aligned). """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        self.fc_final = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            h = self.convs[i](x, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + x if h.shape == x.shape else h  # residual once dims align (from layer 2 on)
        return self.fc_final(x)

class GCNsep(nn.Module):
    """
    Upgraded GCN-sep Variant featuring Explicit Residual Skip Connections,
    Layer-wise LayerNorm Regularization, and Isolated Root Node Projections.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.ego_lins = nn.ModuleList()
        self.lns = nn.ModuleList()

        if num_layers == 1:
            # 1-layer fallback includes direct ego-skip projection to match target output space
            self.convs.append(GCNConv(in_channels, out_channels, add_self_loops=False))
            self.ego_lins.append(nn.Linear(in_channels, out_channels))
        else:
            # Input Layer
            self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            self.ego_lins.append(nn.Linear(in_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            
            # Hidden Layer Stacks
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
                self.ego_lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.lns.append(nn.LayerNorm(hidden_channels))
                
            # Output Classification Layer
            self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))
            self.ego_lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            return self.convs[0](x, edge_index) + self.ego_lins[0](x)
            
        for i in range(self.num_layers - 1):
            x_neighbor = self.convs[i](x, edge_index)
            x_ego = self.ego_lins[i](x)
            
            # Explicit decoupled combination step protects dense text features from diluting
            x = x_neighbor + x_ego
            x = self.lns[i](x) # Individual channel normalization
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return self.convs[-1](x, edge_index) + self.ego_lins[-1](x)


class GCNsepBatchNorm(nn.Module):
    """
    Upgraded GCN-sep Variant featuring Explicit Residual Skip Connections,
    Layer-wise BatchNorm Regularization, and Isolated Root Node Projections.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.ego_lins = nn.ModuleList()
        self.lns = nn.ModuleList()

        if num_layers == 1:
            # 1-layer fallback includes direct ego-skip projection to match target output space
            self.convs.append(GCNConv(in_channels, out_channels, add_self_loops=False))
            self.ego_lins.append(nn.Linear(in_channels, out_channels))
        else:
            # Input Layer
            self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False))
            self.ego_lins.append(nn.Linear(in_channels, hidden_channels))
            self.lns.append(nn.BatchNorm1d(hidden_channels))
            
            # Hidden Layer Stacks
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False))
                self.ego_lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.lns.append(nn.BatchNorm1d(hidden_channels))
                
            # Output Classification Layer
            self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False))
            self.ego_lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            return self.convs[0](x, edge_index) + self.ego_lins[0](x)
            
        for i in range(self.num_layers - 1):
            x_neighbor = self.convs[i](x, edge_index)
            x_ego = self.ego_lins[i](x)
            
            # Explicit decoupled combination step protects dense text features from diluting
            x = x_neighbor + x_ego
            x = self.lns[i](x) # Individual channel normalization
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return self.convs[-1](x, edge_index) + self.ego_lins[-1](x)


class GCN2(nn.Module):
    """
    True GCNII implementation.
    Preserves variance across ultra-deep stacks without BatchNorm layers.
    Topology: Linear(in) -> [Dropout -> GCN2Conv -> ReLU]*num_layers -> Linear(out)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=64, dropout=0.5, alpha=0.1, theta=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha=alpha, theta=theta, layer=i + 1))

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x_0 = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, x_0, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)

class GraphSAGE(nn.Module):
    """
    Inductive GraphSAGE baseline enhanced with individual LayerNorm layers.
    Natively preserves ego-node identities via internal structural concatenation.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            # Input Layer
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            
            # Hidden Stacks
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.lns.append(nn.LayerNorm(hidden_channels))
                
            # Output Layer
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        if self.num_layers == 1:
            return self.convs[0](x, edge_index)
            
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.lns[i](x) # Replaced BatchNorm with LayerNorm tracking
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return self.convs[-1](x, edge_index)


class H2GCN(nn.Module):
    """
    Algorithmic implementation of H2GCN (K=2 hops).
    Strictly separates structural hops and skips self-loops to handle high heterophily.
    Topology: Concat(Ego, Hop1, Hop2) -> Linear -> out
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.fc_embed = nn.Linear(in_channels, hidden_channels)
        self.hop1_conv = H2GCNNeighborhoodConv()
        self.hop2_conv = H2GCNNeighborhoodConv()
        self.fc_final = nn.Linear(hidden_channels + hidden_channels + hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h_ego = F.relu(self.fc_embed(x))
        h_ego = F.dropout(h_ego, p=self.dropout, training=self.training)
        h_hop1 = self.hop1_conv(h_ego, edge_index)
        h_hop2 = self.hop2_conv(h_hop1, edge_index)
        out_features = torch.cat([h_ego, h_hop1, h_hop2], dim=-1)
        return self.fc_final(out_features)

class H2GCNMod(nn.Module):
    """
    True Structural Implementation of H2GCN (Variable K-Hops).
    Self-contained within models.py - requires ZERO changes to main.py.
    Topology: Concat(Ego, Hop1, ..., HopK) -> Linear -> Out
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, edge_index=None, num_nodes=None):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.fc_embed = nn.Linear(in_channels, hidden_channels)
        self.hop_convs = nn.ModuleList([H2GCNNeighborhoodConv() for _ in range(num_layers)])
        
        total_concat_dimension = hidden_channels * (1 + num_layers)
        self.fc_final = nn.Linear(total_concat_dimension, out_channels)
        
        # Permanent storage cache container for your strict multi-hop tensor matrices
        self.hop_edge_indices = []
        
        # --- ONE-TIME PRECOMPUTATION BLOCKS INSIDE INITIALIZATION ---
        if edge_index is not None and num_nodes is not None:
            # Hop 1 is your standard graph topology tensor mask
            self.hop_edge_indices.append(edge_index)
            
            # Convert to an optimized scipy CSR matrix format for fast structural calculations
            adj_1hop = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes)
            
            current_adj_power = adj_1hop
            accumulated_reachable = adj_1hop.copy()
            accumulated_reachable.setdiag(1) # Include self-loops in reachability checks
            
            for k in range(2, num_layers + 1):
                # 1. Compute next step spatial propagation matrix pass
                current_adj_power = current_adj_power @ adj_1hop
                
                # 2. Strict Isolation: Remove self-loops and lower-order links
                strict_khop_adj = current_adj_power.copy()
                strict_khop_adj.setdiag(0)
                strict_khop_adj = strict_khop_adj - strict_khop_adj.multiply(accumulated_reachable > 0)
                strict_khop_adj.eliminate_zeros()
                
                # 3. Re-convert back to native PyG PyTorch tensors on the proper device
                khop_edge_index, _ = from_scipy_sparse_matrix(strict_khop_adj)
                self.hop_edge_indices.append(khop_edge_index.to(edge_index.device))
                
                accumulated_reachable = accumulated_reachable + (strict_khop_adj > 0)

    def forward(self, x, edge_index):
        h_ego = F.relu(self.fc_embed(x))
        h_ego = F.dropout(h_ego, p=self.dropout, training=self.training)
        
        representations_list = [h_ego]
        
        # If pre-computed hops exist in cache, deploy them; fallback to base edge_index otherwise
        has_cache = len(self.hop_edge_indices) == self.num_layers
        
        for k in range(self.num_layers):
            target_edges = self.hop_edge_indices[k] if has_cache else edge_index
            
            # Every hop processes INDEPENDENTLY from root h_ego to prevent feature dilution!
            h_k = self.hop_convs[k](h_ego, target_edges)
            representations_list.append(h_k)
            
        out_features = torch.cat(representations_list, dim=-1)
        out_features = F.dropout(out_features, p=self.dropout, training=self.training)
        
        return self.fc_final(out_features)

class MLP(nn.Module):
    """
    Feature-Only Baseline (0-hop GNN).
    Completely isolates raw text features by bypassing graph topology.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()

        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            # Input Projection
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            
            # Hidden Transforming Layers
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.lns.append(nn.LayerNorm(hidden_channels))
                
            # Out mapping layer
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index=None):
        if self.num_layers == 1:
            return self.lins[0](x)
            
        for i in range(self.num_layers - 1):
            x = self.lins[i](x)
            x = self.lns[i](x) # Stable node-level semantic token tracking
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return self.lins[-1](x)


def get_model(name: str, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, **kwargs):
    """
    Centralized mapping to resolve string queries into explicit GNN configurations.
    Safely handles architecture-specific kwargs across the hyperparameter loop.
    """
    name_lower = name.lower().strip()
    
    # Clean up kwargs copy so arguments don't pollute incompatible constructors
    local_kwargs = kwargs.copy()
    data_obj = local_kwargs.pop('data', None)

    model_args = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'out_channels': out_channels,
        'num_layers': num_layers,
        'dropout': dropout
    }
    
    if name_lower == "gcn":
        local_kwargs.pop('heads', None) # GCN doesn't use attention heads
        return GCN(**model_args, **local_kwargs)

    elif name_lower == "gcn-sep":
        local_kwargs.pop('heads', None) # GCN doesn't use attention heads
        return GCNsep(**model_args, **local_kwargs)

    elif name_lower == "gcn-with-batchnorm":
        local_kwargs.pop('heads', None) # GCN doesn't use attention heads
        return GCNsepBatchNorm(**model_args, **local_kwargs)
    elif name_lower == "gat-naive":
        return GAT_Naive(**model_args, **local_kwargs)
    
    elif name_lower == "gat":
        return GAT(**model_args, **local_kwargs)

    elif name_lower == "gat-egoonly":
            return GAT_EgoOnly(**model_args, **local_kwargs)

    elif name_lower == "gat-neighboronly":
            return GAT_NeighborOnly(**model_args, **local_kwargs)
        
    elif name_lower in ["graphsage", "sage"]:
        local_kwargs.pop('heads', None) # GraphSAGE doesn't use attention heads
        return GraphSAGE(**model_args, **local_kwargs)
        
    elif name_lower in ["gcn2", "gcnii"]:
        local_kwargs.pop('heads', None) # GCNII doesn't use attention heads
        return GCN2(**model_args, **local_kwargs)
        
    elif name_lower == "h2gcn":
        local_kwargs.pop('heads', None) # H2GCN doesn't use attention heads
        return H2GCN(**model_args, **local_kwargs)

    elif name_lower == "h2gcnmod":
        local_kwargs.pop('heads', None) # H2GCN doesn't use attention heads
        if data_obj is not None:
            local_kwargs['edge_index'] = data_obj.edge_index
            local_kwargs['num_nodes'] = data_obj.num_nodes
        else:
            print("WARNING: h2gcnmod built without data — hops degrade to 1-hop fallback!")
        model = H2GCNMod(**model_args, **local_kwargs)
        assert len(model.hop_edge_indices) == model.num_layers, f"Cache not populated: {len(model.hop_edge_indices)} vs {model.num_layers}"
        return model
       
    elif name_lower == "mlp":
        local_kwargs.pop('heads', None) # MLP doesn't use attention heads
        return MLP(**model_args, **local_kwargs)
        
    else:
        raise ValueError(f"Unsupported model architecture: '{name}'")

