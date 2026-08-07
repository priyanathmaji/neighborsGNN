import os
import sys
import numpy as np
import torch
import torch_geometric.utils as utils
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid

# Ensure local custom Datasets package path resolves cleanly
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from Datasets.garesearch_dataset import GAResearchDataset
except ImportError:
    # Fallback to handle direct local execution directory context
    from Datasets.garesearch_dataset import GAResearchDataset

# ==============================================================================
# 1. STRUCTURAL METRICS PROFILER
# ==============================================================================
def compute_heterophily_metrics(data):
    """
    Computes structural homophily parameters to characterize class connections.
    """
    edge_index = data.edge_index
    y = data.y
    num_nodes = data.num_nodes
    
    # Edge Homophily
    row, col = edge_index
    same_class_edges = (y[row] == y[col]).sum().item()
    h_edge = same_class_edges / edge_index.size(1) if edge_index.size(1) > 0 else 0.0
    
    # Node Homophily
    deg = utils.degree(row, num_nodes=num_nodes)
    same_class_mask = (y[row] == y[col]).float()
    same_class_deg = torch.zeros(num_nodes, device=y.device)
    same_class_deg.scatter_add_(0, row, same_class_mask)
    h_node = (same_class_deg / torch.clamp(deg, min=1)).mean().item()
    
    return {
        "edge_homophily": round(h_edge, 4),
        "node_homophily": round(h_node, 4)
    }

# ==============================================================================
# 2. EXPERIMENT DATA FACTORY
# ==============================================================================
def get_clean_heterophily_data(dataset_type: str, name: str, root_path: str = "tmp"):
    """
    Coordinates data downloading via GAResearchDataset or HeterophilousGraphDataset.
    Extracts underlying graph data structure and verifies split tensor constraints.
    """
    dataset_type_lower = dataset_type.lower().strip()
    
    if dataset_type_lower == "garesearch":
        # Loads your custom dataset format
        dataset = GAResearchDataset(root=os.path.join(root_path, name), name=name)
    elif dataset_type_lower == "heterophilous":
        # Loads PyG native baseline configurations (e.g., 'Amazon-ratings', 'Roman-empire')
        dataset = HeterophilousGraphDataset(root=os.path.join(root_path, name), name=name)
    elif dataset_type_lower == "webkb":
        # Loads PyG native WebKB dataset (e.g., 'Cornell', 'Texas', 'Wisconsin')
        dataset = WebKB(root=os.path.join(root_path, name), name=name)
    elif dataset_type.lower() in ["planetoid", "citeseer", "cora", "pubmed"]:
        resolved_name = "Cora" if "cora" in name else ("PubMed" if "pubmed" in name else "CiteSeer")
        dataset = Planetoid(root=root_path, name=resolved_name, split='full')
        data = dataset[0] # Extract the underlying Data object
        num_classes = dataset.num_classes
        num_nodes = data.num_nodes
        train_masks, val_masks, test_masks = [], [], []
        
        if data.x is not None:
            # Casts sparse arrays or alternative bit-weights straight to 
            # standard PyTorch floating-point tensors before GPU allocation passes
            data.x = data.x.to(torch.float32)
            if hasattr(data.x, 'to_dense'):
                try:
                    data.x = data.x.to_dense()
                except Exception:
                    pass

        train_masks, val_masks, test_masks = [], [], []
        rng = np.random.default_rng(seed=42)

        # --- GENERATE THE 10 FOLDS MANUALLY ---
        for fold in range(10):
            # Enforce standard cross-validation literature split distribution weights:
            # 60% Train / 20% Val / 20% Test partitions
            indices = rng.permutation(num_nodes)
            
            train_boundary = int(num_nodes * 0.60)
            val_boundary = int(num_nodes * 0.80)
            
            train_idx = indices[:train_boundary]
            val_idx = indices[train_boundary:val_boundary]
            test_idx = indices[val_boundary:]
            
            t_mask = torch.zeros(num_nodes, dtype=torch.bool)
            v_mask = torch.zeros(num_nodes, dtype=torch.bool)
            te_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            t_mask[train_idx] = True
            v_mask[val_idx] = True
            te_mask[test_idx] = True
            
            # Add an extra dimension so we can concatenate them into columns later
            train_masks.append(t_mask.unsqueeze(1))
            val_masks.append(v_mask.unsqueeze(1))
            test_masks.append(te_mask.unsqueeze(1))
            
        # Cat columns along dimension 1 to standardize into the 2D matrix shape [num_nodes, 10]
        data.train_mask = torch.cat(train_masks, dim=1)
        data.val_mask = torch.cat(val_masks, dim=1)
        data.test_mask = torch.cat(test_masks, dim=1)
        
        return data, 10
    else:
        raise ValueError(f"Unknown dataset wrapper context type: '{dataset_type}'")
        
    data = dataset[0]
    
    # Validate multi-split mask presence [num_nodes, 10]
    if not hasattr(data, 'train_mask') or data.train_mask.dim() < 2:
        raise AttributeError(
            f"Dataset {name} is missing a multi-split 10-fold execution matrix mask layout. "
            f"Verify your dataset build configuration outputs [Nodes, Splits] dimensions."
        )
        
    num_splits = data.train_mask.size(1)
    return data, num_splits
