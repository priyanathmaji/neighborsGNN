import os
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SMALL_LEDGER_PATH = os.path.join(PROJECT_ROOT, "results/json/small_metrics_ledger.json")
DETAILED_LEDGER_PATH = os.path.join(PROJECT_ROOT, "results/json/detailed_metrics_ledger.json")

def print_synchronized_global_metrics():
    if not os.path.exists(SMALL_LEDGER_PATH) or not os.path.exists(DETAILED_LEDGER_PATH):
        print("[ERROR] Database file layers are missing. Ensure both ledger files exist.")
        return

    # 1. Load the small metrics ledger (which holds the true hyperparameters)
    with open(SMALL_LEDGER_PATH, 'r') as f:
        small_records = json.load(f)
        
    # 2. Load the detailed metrics ledger (which holds accuracy and balanced accuracy metrics)
    with open(DETAILED_LEDGER_PATH, 'r') as f:
        detailed_records = json.load(f)

    # Map the detailed records using their timestamp as a primary lookup key
    detailed_map = {run.get('timestamp'): run for run in detailed_records if run.get('timestamp')}

    # Define the precise production baseline architectures array
    target_models = ["GCN", "GAT-NAIVE","GAT", "GCN-WITH-BATCHNORM", "H2GCNMOD", "GRAPHSAGE", "GCN2", "MLP", "H2GCN", "GCN-SEP", "GAT-EGOONLY", "GAT-NEIGHBORONLY"]
    dataset_groups = sorted(list(set(r['dataset_name'] for r in small_records)))

    print("\n" + "="*155)
    print("                      ACCURATE AGGREGATED AVERAGES PERFORMANCE MATRIX (CROSS-LEDGER SYNCHRONIZED)")
    print("="*155)

    for dataset in dataset_groups:
        print(f"\n📍 DATASET FEATURE SPACE: {dataset.upper()}")
        print("-" * 155)
        print(f" {'Model Architecture':<22} | {'Lyr':<3} | {'Dim':<4} | {'LR':<5} | {'Drop':<4} | {'Macro F1-Score':<18} | {'Overall Accuracy':<18} | {'Balanced Accuracy':<18}")
        print("-" * 155)

        # Track the top scoring run for each model in this dataset group
        best_model_runs = {}
        for s_run in small_records:
            if s_run['dataset_name'] == dataset:
                arch_upper = s_run['model_architecture'].upper()
                if arch_upper in target_models:
                    # Keep the record with the highest Macro F1 score
                    if arch_upper not in best_model_runs or s_run['macro_f1_mean'] > best_model_runs[arch_upper]['macro_f1_mean']:
                        best_model_runs[arch_upper] = s_run

        for model_name in target_models:
            if model_name not in best_model_runs:
                continue
                
            s_run = best_model_runs[model_name]
            ts = s_run.get('timestamp')
            hp = s_run.get('hyperparameters', {})
            
            # Extract the real, accurate hyperparameter values from the small ledger entries
            lyr = hp.get('layers', hp.get('num_layers', 'N/A'))
            dim = hp.get('hidden_dim', s_run.get('hidden_dimension', 'N/A'))
            lr = hp.get('lr', 'N/A')
            drop = hp.get('dropout', 'N/A')

            # Pull global score averages from the small ledger
            f1_mean = s_run.get('macro_f1_mean', 0.0)
            f1_std = s_run.get('macro_f1_std', 0.0)
            f1_str = f"{f1_mean:.2f}% \u00b1 {f1_std:.2f}%"

            # Cross-reference the detailed ledger to extract accuracy and balanced accuracy values
            acc_str = "N/A"
            bal_str = "N/A"
            
            if ts in detailed_map:
                d_run = detailed_map[ts]
                d_avg = d_run.get('global_aggregated_averages', {})
                acc_str = f"{d_avg.get('accuracy_mean', 0.0):.2f}% \u00b1 {d_avg.get('accuracy_std', 0.0):.2f}%"
                bal_str = f"{d_avg.get('balanced_accuracy_mean', 0.0):.2f}% \u00b1 {d_avg.get('balanced_accuracy_std', 0.0):.2f}%"

            print(f" {model_name:<22} | {lyr:<3} | {dim:<4} | {lr:<5} | {drop:<4} | {f1_str:<18} | {acc_str:<18} | {bal_str:<18}")
        print("-" * 155)

if __name__ == "__main__":
    print_synchronized_global_metrics()
