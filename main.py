import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score

IS_SMALL_REPORT_MODE = False
SKIP_DETAILED_LEDGER = False

# Ensure local custom Datasets package path and working directory resolve cleanly
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import dependencies from your local workspace modules
from models import get_model
from datasets import get_clean_heterophily_data, compute_heterophily_metrics
from logger_and_plots import ComprehensivePaperLogger

# Try importing your custom tracker safely
try:
    from results_manager import ExperimentTracker
except ImportError:
    class ExperimentTracker:
        def __init__(self, dataset_name, node_feature_type): pass
        def log_epoch(self, **kwargs): pass
        def save_results(self, **kwargs): pass

# ==============================================================================
# 1. ENFORCE TOTAL REPRODUCIBILITY (BACKEND ATOMIC CONTROLS)
# ==============================================================================
def enforce_strict_determinism(seed=42):
    """ Locks all systemic pseudo-random states and anchors PyG CUDA environments. """
    import os
    import random
    import numpy as np
    import torch

    # 1. Establish structural shell masks first
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Lock runtime random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 3. Secure backend engine operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. Trigger deterministic checking last, once all settings are ready
    torch.use_deterministic_algorithms(True, warn_only=True)


# ==============================================================================
# 2. PROACTIVE EARLY STOPPING CONTROLS
# ==============================================================================
class EarlyStopping:
    """
    Monitors validation performance metrics. Automatically halts training loop
    if optimization plateaus, protecting the weights from target-split leakage.
    """
    def __init__(self, patience=20, path="checkpoint.pth"):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.optimal_epoch = 0

    def __call__(self, val_metric, epoch, model):
        if self.best_score is None:
            self.best_score = val_metric
            self.optimal_epoch = epoch
            self.save_checkpoint(model)
        elif val_metric <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.optimal_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# ==============================================================================
# 3. CORE TRAIN & EVALUATION PIPELINE FUNCTIONS
# ==============================================================================
def train_epoch(model, data, mask, optimizer, criterion):
    """ Executes a single optimization training step over the selected split. """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_model(model, data, mask):
    """ Evaluates the model, generating macro statistics for training loop feedback. """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1).cpu().numpy()
    y = data.y[mask].cpu().numpy()
    
    f1 = f1_score(y, pred, average='macro', zero_division=0)
    acc = accuracy_score(y, pred)
    bal_acc = balanced_accuracy_score(y, pred)
    
    return f1, acc, bal_acc, y, pred

# ==============================================================================
# 4. OPTIMIZER SEARCH FACTORY
# ==============================================================================
def build_optimizer_suite(args, model_params):
    filter_fn = filter(lambda p: p.requires_grad, model_params)
    lr = args.get('lr', 0.001)
    weight_decay = args.get('weight_decay', 1e-4)
    opt_type = args.get('opt', 'adamw').lower()
    
    if opt_type == 'adam':
        optimizer = torch.optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        optimizer = torch.optim.AdamW(filter_fn, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(filter_fn, lr=lr, momentum=0.95, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer selection: {opt_type}")
    return None, optimizer

# ==============================================================================
# 5. DYNAMIC EXPERIMENT ENGINE (ITERATES OVER 10 SPLITS & PLOTS PER RUN)
# ==============================================================================
def run_complete_experiment(args):
    enforce_strict_determinism(args['seed'])
    device = args['device']
    
    print(f"\n===== Running Grid: {args['model_name'].upper()} | Data: {args['dataset_name'].upper()} | Hidden Dim: {args['hidden_dim']} =====")
    
    data, num_splits = get_clean_heterophily_data(
        dataset_type=args['dataset_type'], 
        name=args['dataset_name'], 
        root_path=args['root_path']
    )
    
    num_features = data.num_features
    num_classes = int(data.y.max().item() + 1)
    
    data = data.to(device)
    
    split_records = []
    y_trues_all_splits = []
    y_preds_all_splits = []
    
    tracker = ExperimentTracker(dataset_name=args['dataset_name'], node_feature_type=args['node_feature_type'])
    paper_logger = ComprehensivePaperLogger()
    
    for split_idx in range(num_splits):
        print(f"   -> Processing Split {split_idx + 1}/{num_splits}")
        
        model = get_model(
            name=args['model_name'],
            in_channels=num_features,
            hidden_channels=args['hidden_dim'],
            out_channels=num_classes,
            num_layers=args['num_layers'],
            dropout=args['dropout'],
            heads=args.get('heads', 1),
            data=data
        ).to(device)
        
        _, optimizer = build_optimizer_suite(args, model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        split_ckpt_path = f"temp_ckpt_{args['model_name']}_{args['dataset_name']}_dim{args['hidden_dim']}_s{split_idx}.pth"
        early_stopper = EarlyStopping(patience=args['patience'], path=split_ckpt_path)
        
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
        
        train_losses, val_f1s = [], []
        split_train_start = time.time()
        
        for epoch in range(1, args['epochs'] + 1):
            loss = train_epoch(model, data, train_mask, optimizer, criterion)
            val_f1, _, _, _, _ = evaluate_model(model, data, val_mask)
            
            train_losses.append(loss)
            val_f1s.append(val_f1)
            
            early_stopper(val_f1, epoch, model)
            if early_stopper.early_stop:
                break
                
        split_train_dur = time.time() - split_train_start
        
        # Load optimal weight parameters back for final test generation
        model.load_state_dict(torch.load(split_ckpt_path, map_location=device))
        
        inf_start = time.time()
        test_f1, test_acc, test_bal_acc, y_true, y_pred = evaluate_model(model, data, test_mask)
        inf_dur = time.time() - inf_start
        
        if os.path.exists(split_ckpt_path):
            os.remove(split_ckpt_path)
            
        split_records.append({
            "split": split_idx + 1,
            "optimal_epoch": early_stopper.optimal_epoch,
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "test_balanced_acc": float(test_bal_acc),
            "train_time_sec": split_train_dur,
            "inference_time_sec": inf_dur
        })
        
        y_trues_all_splits.append(y_true)
        y_preds_all_splits.append(y_pred)
        
        # Generate the Times New Roman Appendix figures for Split 1
        if split_idx == 0:
            paper_logger.generate_all_plots(
                args['model_name'], args['dataset_name'], args['hidden_dim'], 
                split_idx + 1, train_losses, val_f1s, y_true, y_pred, num_classes
            )
            
    # Process aggregates and write directly to the comprehensive JSON Ledger
    paper_logger.commit_experiment_run(args, split_records, y_trues_all_splits, y_preds_all_splits, num_classes)


def run_single_diagnostic_test():
    """
    Optional helper function to run a single, quick test run.
    Can be manually called if debugging main.py independently.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    run_config = {
        'seed': 42,
        'device': device,
        'dataset_type': 'garesearch',
        'dataset_name': 'roman-transformer',
        'root_path': 'tmp',
        'node_feature_type': 'BERT Auto / Native',
        'model_name': 'h2gcn',
        'num_layers': 3,
        'hidden_dim': 256,
        'dropout': 0.3,
        'epochs': 600,
        'patience': 50,
        'opt': 'adamw',
        'lr': 0.001,
        'weight_decay': 1e-4,
    }
    
    # Force environmental hardware safety inside the diagnostic test
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    print("\n>>> Launching Standalone Diagnostic Test Execution <<<")
    run_complete_experiment(run_config)

if __name__ == "__main__":
    # Commented out to prevent accidental activation during master experiment swings.
    # To run a standalone single-shot test from main.py, uncomment the line below:
    # run_single_diagnostic_test()
    pass

