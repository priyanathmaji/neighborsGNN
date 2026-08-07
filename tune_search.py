import os
# CRITICAL LAYER 1: Force hardware workspace allocation at Line 1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import json
import copy
import torch
import numpy as np
import random
import time

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from main import train_epoch, evaluate_model, EarlyStopping
from datasets import get_clean_heterophily_data
from models import get_model

# Establish permanent absolute project path maps to protect against cluster node shifts
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TUNING_LEDGER_PATH = os.path.join(PROJECT_ROOT, "results/json/optimal_hyperparams.json")

def enforce_strict_determinism(seed=42):
    """ Completely freezes all pseudo-random sequence pipelines for the upcoming block. """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def execute_fast_validation_trial(config, data, in_channels, out_channels):
    """ Executes a fast optimization run over Split 1 validation masks. """
    device = config['device']
    enforce_strict_determinism(config['seed'])
    
    model = get_model(
        name=config['model_name'],
        in_channels=in_channels,
        hidden_channels=config['hidden_dim'],
        out_channels=out_channels,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        heads=config.get('heads', 1),
        data=data
    ).to(device)
    
    filter_fn = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(filter_fn, lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    
    tuning_ckpt = os.path.join(PROJECT_ROOT, f"tuning_tmp_{config['model_name']}.pth")
    # Tighter patience window during sweeps to discard failing trials quickly
    early_stopper = EarlyStopping(patience=20, path=tuning_ckpt)
    
    train_mask = data.train_mask[:, 0]  # Lock strictly to Split 1 for tuning speed
    val_mask = data.val_mask[:, 0]
    
    for epoch in range(1, config['epochs'] + 1):
        _ = train_epoch(model, data, train_mask, optimizer, criterion)
        val_f1, _, _, _, _ = evaluate_model(model, data, val_mask)
        
        early_stopper(val_f1, epoch, model)
        if early_stopper.early_stop:
            break
            
    best_val_f1 = early_stopper.best_score
    if os.path.exists(tuning_ckpt):
        os.remove(tuning_ckpt)
        
    return best_val_f1


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(TUNING_LEDGER_PATH), exist_ok=True)
    
    if os.path.exists(TUNING_LEDGER_PATH):
        with open(TUNING_LEDGER_PATH, 'r') as f:
            hyperparam_ledger = json.load(f)
    else:
        hyperparam_ledger = {}

    target_benchmarks = [
        {"type": "garesearch", "name": "roman-transformer"},
        {"type": "garesearch", "name": "roman-fasttext"},
        {"type": "garesearch", "name": "roman-roberta"},
        {"type": "garesearch", "name": "roman-custom-transformer"},

        {"type": "garesearch", "name": "amazon-fasttext"},
        {"type": "garesearch", "name": "amazon-mpnet-base-v2"},
        {"type": "garesearch", "name": "amazon"},

        {"type": "webkb", "name": "cornell"},
        {"type": "webkb", "name": "wisconsin"},
        {"type": "webkb", "name": "texas"},

        {"type": "planetoid", "name": "pubmed"},
        {"type": "planetoid", "name": "cora"},
    ]
    
    architectures = ["gat", "gat-naive", "gcn", "gcn-with-batchNorm", "h2gcnmod", "graphsage", "gcn2", "mlp"]
    

    for dataset in target_benchmarks:
        d_name = dataset['name']
        if d_name not in hyperparam_ledger:
            hyperparam_ledger[d_name] = {}
            
        for arch in architectures:
            if arch in hyperparam_ledger[d_name]:
                print(f"[SKIPPED] {arch.upper()} on {d_name} already has tuned params.")
                continue

            config_str = f"\n=== Configuring Customized Sweep Space: {arch.upper()} on {d_name.upper()} ==="
            print(config_str)
            txt_log_path = os.path.join(PROJECT_ROOT, "results/json/tuning_terminal_logs.txt")
            try:
                with open(txt_log_path, "a") as txt_file:
                    txt_file.write(config_str + "\n")
            except Exception as txt_err:
                pass
            current_heads = 1
            
            # ------------------------------------------------------------------
            # DYNAMIC ARCHITECTURE-AWARE SEARCH SPACE GRID CUSTOMIZATION
            # ------------------------------------------------------------------
            if arch in ["gcn2", "gcnii"]:
                #layer_options = [32,64]            # Lock to native deep residual scale
                layer_options = [10]                #to speed up tuning, we can lock to 10 layers for GCNII
                dropout_options = [0.2]     # Bypasses the 0.5 parameter collapse threshold
                weight_decays = [1e-4]           # Fixed weight decay to optimize compute
                learning_rates = [0.001, 0.005]
                hidden_dims = [64, 128, 256]     # Deep models require smaller hidden channels footprint
            elif arch in ["gcn","gcn-with-batchNorm"]: #gcn-sep but with batchnorm
                layer_options = [2,3,4,5]           # Allows standard models to find their shallow sweet-spot
                dropout_options = [0.1, 0.2, 0.5]
                weight_decays = [1e-4, 5e-4, 1e-3]
                learning_rates = [0.005, 0.001, 0.01] 
                hidden_dims = [64, 128, 256]

            #Delete 
            elif arch == "gat-naive":
                layer_options = [1,2]           # 1,2 Allows standard models to find their shallow sweet-spot
                dropout_options = [0.2]
                weight_decays = [1e-4]
                learning_rates = [0.01, 0.001] 
                hidden_dims = [64,128]
                if arch == "gat-naive":
                    current_heads = 8

            elif arch == "gat":
                layer_options = [3,5]           # 1,2 Allows standard models to find their shallow sweet-spot
                dropout_options = [0.1, 0.2]
                weight_decays = [1e-4, 5e-4]
                learning_rates = [0.005, 0.001, 0.01] 
                hidden_dims = [64, 128]

                if arch == "gat":
                    current_heads = 8
                if arch == "gat-naive":
                    current_heads = 8
            elif arch == "h2gcnmod":
                layer_options = [2]              # Locked to explicit 2-hop radius design
                dropout_options = [0.2, 0.5]
                weight_decays = [1e-4, 1e-3]
                learning_rates = [0.001, 0.005]
                hidden_dims = [128, 256]
            else:  # GraphSAGE and MLP configurations
                layer_options = [2, 3]
                dropout_options = [0.1, 0.2, 0.5]
                weight_decays = [1e-4, 5e-4, 1e-3]
                learning_rates = [0.005, 0.001, 0.01] 
                hidden_dims = [64, 128, 256]
            # ------------------------------------------------------------------
            
            base_args = {
                'seed': 42, 'device': device, 'dataset_type': dataset['type'],
                'dataset_name': d_name, 'root_path': 'tmp', 'epochs': 400, 'patience': 35,
                'model_name': arch, 'opt': 'adamw', 'heads': current_heads
            }
            
            data, _ = get_clean_heterophily_data(base_args['dataset_type'], base_args['dataset_name'], base_args['root_path'])
            
            in_channels, out_channels = data.num_features, int(data.y.max().item() + 1)
            data = data.to(device)
            
            best_score = -1.0
            best_config = {}
            
            trial_counter = 0
            total_trials = len(layer_options) * len(hidden_dims) * len(learning_rates) * len(weight_decays) * len(dropout_options)
            
            for layers in layer_options:
                for dim in hidden_dims:
                    for lr in learning_rates:
                        for wd in weight_decays:
                            for drop in dropout_options:
                                trial_counter += 1
                                
                                trial_config = copy.deepcopy(base_args)
                                trial_config.update({
                                    'num_layers': layers,
                                    'hidden_dim': dim,
                                    'lr': lr,
                                    'weight_decay': wd,
                                    'dropout': drop
                                })

                                combo_str = f"   [Combo {trial_counter}/{total_trials}] Running -> Layers: {layers} | Dim: {dim:<3} | LR: {lr:<5} | Drop: {drop:<4} | WD: {wd:<5}"
                                print(combo_str, end="", flush=True)
                                
                                start_trial_time = time.time()
                                val_score = execute_fast_validation_trial(trial_config, data, in_channels, out_channels)
                                duration = time.time() - start_trial_time

                                done_str = f" Done! Peak Val F1: {val_score*100:.2f}% (Took {duration:.1f}s)"
                                                                
                                print(done_str)

                                txt_log_path = os.path.join(PROJECT_ROOT, "results/json/tuning_terminal_logs.txt")
                                try:
                                    with open(txt_log_path, "a") as txt_file:
                                        txt_file.write(combo_str + done_str + "\n")
                                except Exception as txt_err:
                                    pass

                                if val_score > best_score:
                                    best_score = val_score
                                    best_config = {'num_layers': layers, 'hidden_dim': dim, 'lr': lr, 'weight_decay': wd, 'dropout': drop}
                                    
                                    # Atomic flush to protect completed results ledger instantly
                                    hyperparam_ledger[d_name][arch] = {
                                        "best_val_macro_f1": round(best_score * 100, 2),
                                        "params": best_config
                                    }
                                    try:
                                        with open(TUNING_LEDGER_PATH, 'w') as f:
                                            json.dump(hyperparam_ledger, f, indent=4)
                                            f.flush()
                                            os.fsync(f.fileno())
                                    except Exception as write_err:
                                        print(f"\n[WARNING] Disk write failure: {write_err}")
            
            print(f"--> [FINALIZED SELECTION] Best Config for {arch}: {best_config}\n")
