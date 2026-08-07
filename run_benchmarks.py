import os
# Force hardware workspace allocation at Line 1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import json
import copy
import torch
import random
import numpy as np
import argparse

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import main # Import reference directly to configure global scope states
from main import run_complete_experiment

TUNING_LEDGER_PATH = "results/json/optimal_hyperparams.json"

def enforce_strict_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic 10-Fold Cross Validation Benchmarking Pipeline.")
    parser.add_argument('--small', action='store_true', help="Enforces concise display logs.")
    cmd_args = parser.parse_args()
    
    # Secure state routing to main execution module
    main.IS_SMALL_REPORT_MODE = cmd_args.small
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(TUNING_LEDGER_PATH):
        raise FileNotFoundError(f"Tuning ledger mapping is missing at {TUNING_LEDGER_PATH}.")
        
    with open(TUNING_LEDGER_PATH, 'r') as f:
        tuned_matrix = json.load(f)

    target_benchmarks = [
        {"type": "garesearch", "name": "roman-transformer", "features": "BERT Auto"},
        {"type": "garesearch", "name": "roman-fasttext", "features": "FastText Native"},
        {"type": "garesearch", "name": "roman-roberta", "features": "RoBERTa Embeds"},
        {"type": "garesearch", "name": "roman-custom-transformer", "features": "Custom Transformer"},
        
        {"type": "garesearch", "name": "amazon-fasttext", "features": "FastText Custom"},
        {"type": "garesearch", "name": "amazon-mpnet-base-v2", "features": "MPNet Base Contextual"},
        {"type": "garesearch", "name": "amazon", "features": "SBERT Custom Embeds"},
        
        {"type": "webkb", "name": "cornell", "features": "Native Web Attributes"},
        {"type": "webkb", "name": "wisconsin", "features": "Native Web Attributes"},
        {"type": "webkb", "name": "texas", "features": "Native Web Attributes"},

        {"type": "planetoid", "name": "pubmed","features": "Native TF-IDF Embeds"},
        {"type": "planetoid", "name": "cora","features": "Native TF-IDF Embeds"},

    ]
    
    architectures = ["gcn","gat-naive","gat", "gcn-with-batchNorm","h2gcnmod", "graphsage" ,"gcn2", "mlp"] # ,"gcn","gcn-with-batchNorm","h2gcnmod", "graphsage" ,"gcn2", "mlp" ["h2gcn", "graphsage", "gcn", "gat", "gcn2", "mlp", "gcn-sep", "gat-egoonly", "gat-neighboronly","gcn-with-batchNorm","h2gcnmod"]
    
    
    for dataset in target_benchmarks:
        d_name = dataset['name']
        
        for arch in architectures:
            if d_name not in tuned_matrix or arch not in tuned_matrix[d_name]:
                continue
                
            optimal_hyperparams = tuned_matrix[d_name][arch]["params"]
            
            run_config = {
                'seed': 42, 'device': device, 'dataset_type': dataset['type'], 'dataset_name': d_name,
                'root_path': 'tmp', 'node_feature_type': dataset['features'], 'model_name': arch,
                'opt': 'adamw', 'epochs': 600, 'patience': 50,
                'num_layers': 3 if arch != "gcn2" else 64, # Default allocation fallback
            }
            run_config.update(optimal_hyperparams)
            
            # Extract detailed keys to print out in the dashboard layout
            layers_val = run_config.get('num_layers', 'N/A')
            dim_val = run_config.get('hidden_dim', 'N/A')
            lr_val = run_config.get('lr', 'N/A')
            wd_val = run_config.get('weight_decay', 'N/A')
            drop_val = run_config.get('dropout', 'N/A')
            heads_val = run_config.get('heads', 1 if arch not in ["gat", "gat-naive"] else 8)

            # ------------------------------------------------------------------
            # MASTER HYPERPARAMETER DISPLAY BLOCK
            # ------------------------------------------------------------------
            print("\n" + "="*85)
            print(f"RUNNING OFFICIAL 10-FOLD CV BENCHMARK: {arch.upper()} on {d_name.upper()}")
            print("-"*85)
            print(f" Parameters -> Layers: {layers_val:<2} | Dim: {dim_val:<4} | LR: {lr_val:<5} | WD: {wd_val:<6} | Dropout: {drop_val:<4} | Heads: {heads_val}")
            print("="*85)
            
            try:
                enforce_strict_determinism(run_config['seed'])
                run_complete_experiment(run_config)
            except Exception as e:
                print(f"[CRASH ERROR] Failed execution loop for {arch} on {d_name} due to error: {e}")
                continue
