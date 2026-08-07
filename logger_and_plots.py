import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ==============================================================================
# GLOBAL NEURIPS EMBEDDED TYPOGRAPHY STYLING CONFIGURATION
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "text.usetex": False,            # Set to True only if system has a working LaTeX engine
    "axes.edgecolor": "#111111",     # Crisp dark borders
    "axes.linewidth": 0.8,
    "grid.color": "#cccccc",
    "grid.alpha": 0.5,
    "xtick.direction": "in",         # Academic inward tick marks
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4
})

# Complete mapping array list to ground the 18 Roman Empire class names to text strings
ROMAN_CLASS_MAP = [
    "pobj", "prep", "det", "amod", "conj", "nsubj", "cc", "compound", "ROOT",
    "dobj", "advmod", "aux", "auxpass", "appos", "nsubjpass", "poss", "relcl", "other"
]

class ComprehensivePaperLogger:
    """
    Handles high-density JSON logging and generates high-resolution figures 
    formatted in clean, academic NeurIPS-style Times New Roman layouts.
    """
    def __init__(self, json_path="results/json/detailed_metrics_ledger.json"):
        self.json_path = json_path
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs("results/plots/learning_curves", exist_ok=True)
        os.makedirs("results/plots/confusion_matrices", exist_ok=True)
        os.makedirs("results/plots/per_class_metrics", exist_ok=True)
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    self.ledger = json.load(f)
            except Exception:
                self.ledger = []
        else:
            self.ledger = []

    def compute_all_metrics(self, y_true, y_pred, num_classes):
        """ Computes global macro statistics alongside granular, per-class metrics. """
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
        per_class_prec = precision_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
        per_class_rec = recall_score(y_true, y_pred, average=None, labels=range(num_classes), zero_division=0)
        
        return {
            "global_macro": {
                "f1": round(float(macro_f1) * 100, 2),
                "precision": round(float(macro_prec) * 100, 2),
                "recall": round(float(macro_rec) * 100, 2)
            },
            "per_class": {
                "f1s": [round(float(x) * 100, 2) for x in per_class_f1],
                "precisions": [round(float(x) * 100, 2) for x in per_class_prec],
                "recalls": [round(float(x) * 100, 2) for x in per_class_rec]
            }
        }

    def generate_all_plots(self, model_name, dataset_name, dim, split, train_losses, val_f1s, y_true, y_pred, num_classes):
        """ Generates individual split performance trajectory metadata. """
        base_fn = f"{model_name}_{dataset_name}_dim{dim}_split{split}"
        
        fig, ax1 = plt.subplots(figsize=(6.5, 4.0), facecolor='white')
        epochs_range = range(1, len(train_losses) + 1)
        
        ax1.set_facecolor('white')
        ax1.set_xlabel('Training Epochs', fontsize=11)
        ax1.set_ylabel('Cross Entropy Loss', color='#b2182b', fontsize=11)
        ax1.plot(epochs_range, train_losses, color='#b2182b', alpha=0.9, linewidth=1.2, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor='#b2182b', labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Validation Macro F1 Score', color='#2166ac', fontsize=11)
        ax2.plot(epochs_range, val_f1s, color='#2166ac', alpha=0.9, linewidth=1.2, linestyle="--", label='Val F1')
        ax2.tick_params(axis='y', labelcolor='#2166ac', labelsize=10)
        
        plt.title(f"Learning Dynamics: {model_name.upper()} on {dataset_name.upper()} (Dim {dim} | Split {split})", fontsize=12, pad=10)
        fig.tight_layout()
        plt.savefig(f"results/plots/learning_curves/{base_fn}_learning_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_aggregated_plots(self, model_name, dataset_name, dim, all_y_true, all_y_pred, mean_class_f1s, std_class_f1s, num_classes):
        """ Generates 10-fold consolidated statistics graphs for cross-validation summary blocks. """
        base_fn = f"{model_name}_{dataset_name}_dim{dim}_10fold_aggregated"
        
        # Resolve class names based on dataset context rules cleanly
        if "roman" in dataset_name.lower():
            class_labels = [ROMAN_CLASS_MAP[c] if c < len(ROMAN_CLASS_MAP) else f"class_{c}" for c in range(num_classes)]
        else:
            class_labels = [f"Class {c}" for c in range(num_classes)]
            
        # ----------------------------------------------------------------------
        # FIGURE 2: 10-Fold Globally Aggregated Confusion Matrix Heatmap
        # ----------------------------------------------------------------------
        plt.figure(figsize=(8.5, 7.5), facecolor='white')
        flat_y_true = np.concatenate(all_y_true)
        flat_y_pred = np.concatenate(all_y_pred)
        
        cm = confusion_matrix(flat_y_true, flat_y_pred, labels=range(num_classes))
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        
        ax = sns.heatmap(cm_norm, annot=num_classes <= 12, fmt=".2f", cmap='Blues', cbar=True, square=True,
                         annot_kws={"size": 8, "family": "serif"}, cbar_kws={"shrink": 0.8},
                         xticklabels=class_labels, yticklabels=class_labels)
        ax.set_facecolor('white')
        plt.title(f"Aggregated Confusion Matrix: {model_name.upper()} ({dataset_name.upper()} | 10 Splits)", fontsize=11, pad=12, fontweight='bold')
        plt.xlabel('Predicted Category Classes', fontsize=11, labelpad=8)
        plt.ylabel('True Target Ground Labels', fontsize=11, labelpad=8)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.savefig(f"results/plots/confusion_matrices/{base_fn}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ----------------------------------------------------------------------
        # FIGURE 3: Ordered Dot-and-Whisker (Lollipop Variant) Plot - NEURIPS SPECIALTY
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.5, len(class_labels) * 0.28 + 1.5), facecolor='white')
        ax.set_facecolor('white')
        
        # Sort indices incrementally from lowest F1 performance to highest
        sorted_indices = np.argsort(mean_class_f1s)
        y_positions = np.arange(num_classes)
        
        # Draw background guideline markers to make scanning long naming lists effortlessly clean
        ax.grid(True, axis='x', linestyle=':', color='#cccccc', alpha=0.6)
        
        # Plot structural lollipop lines
        for idx, y_pos in enumerate(y_positions):
            orig_idx = sorted_indices[idx]
            ax.plot([0, mean_class_f1s[orig_idx]], [y_pos, y_pos], color='#bbbbbb', linewidth=0.8, zorder=1)
            
        # Superimpose clean, variance-bounded error points
        ax.errorbar(
            mean_class_f1s[sorted_indices], y_positions, 
            xerr=std_class_f1s[sorted_indices], 
            fmt='o', color='#2166ac', ecolor='#111111',
            elinewidth=0.9, capsize=2.5, markersize=4.5, zorder=2,
            label=r'Mean F1 ($\pm$ Cross-Fold Std)'
        )
        
        plt.yticks(y_positions, [class_labels[i] for i in sorted_indices], fontsize=9.5)
        plt.xticks(fontsize=9.5)
        plt.xlabel('Macro F1-Score Percentage (%)', fontsize=10.5, labelpad=6)
        plt.ylabel('Linguistic Syntax Category Role', fontsize=10.5, labelpad=6)
        plt.title(f"Per-Class Structural Stability Range: {model_name.upper()} ({dataset_name.upper()})", fontsize=11, pad=12, fontweight='bold')
        plt.xlim(0, 105)
        
        # Clear top and right spines to follow modern publication styling guidelines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"results/plots/per_class_metrics/{base_fn}_class_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()

    def commit_experiment_run(self, run_metadata, split_records, y_trues_all_splits, y_preds_all_splits, num_classes):
        """ Collects split-level lists, computes mean standard deviations, and updates logs. """
        # Import the global flag context state natively from main
        import main
        
        all_splits_f1s, all_splits_precs, all_splits_recs = [], [], []
        for s_idx in range(len(split_records)):
            metrics = self.compute_all_metrics(y_trues_all_splits[s_idx], y_preds_all_splits[s_idx], num_classes)
            all_splits_f1s.append(metrics["per_class"]["f1s"])
            all_splits_precs.append(metrics["per_class"]["precisions"])
            all_splits_recs.append(metrics["per_class"]["recalls"])
            
        all_splits_f1s = np.array(all_splits_f1s)
        all_splits_precs = np.array(all_splits_precs)
        all_splits_recs = np.array(all_splits_recs)
        
        mean_class_f1s = np.mean(all_splits_f1s, axis=0)
        std_class_f1s = np.std(all_splits_f1s, axis=0)
        mean_class_precs = np.mean(all_splits_precs, axis=0)
        mean_class_recs = np.mean(all_splits_recs, axis=0)
        
        f1_list = [r['test_f1'] * 100 for r in split_records]
        acc_list = [r['test_acc'] * 100 for r in split_records]
        bal_acc_list = [r['test_balanced_acc'] * 100 for r in split_records]
        
        f1_mean = round(float(np.mean(f1_list)), 2)
        f1_std = round(float(np.std(f1_list)), 2)
        
        # 1. Standard full-breadth research logging payload
        if not main.SKIP_DETAILED_LEDGER:
            master_payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_architecture": run_metadata['model_name'],
                "dataset_name": run_metadata['dataset_name'],
                "hidden_dimension": run_metadata['hidden_dim'],
                "global_aggregated_averages": {
                    "macro_f1_mean": f1_mean,
                    "macro_f1_std": f1_std,
                    "accuracy_mean": round(float(np.mean(acc_list)), 2),
                    "accuracy_std": round(float(np.std(acc_list)), 2),
                    "balanced_accuracy_mean": round(float(np.mean(bal_acc_list)), 2),
                    "balanced_accuracy_std": round(float(np.std(bal_acc_list)), 2)
                },
                "per_class_cross_validation_averages": {
                    f"class_{c}": {
                        "f1_mean": round(float(mean_class_f1s[c]), 2),
                        "f1_std": round(float(std_class_f1s[c]), 2),
                        "precision_mean": round(float(mean_class_precs[c]), 2),
                        "recall_mean": round(float(mean_class_recs[c]), 2),
                        "latex_f1": f"${mean_class_f1s[c]:.2f} \\pm {std_class_f1s[c]:.2f}$"
                    } for c in range(num_classes)
                },
                "split_level_raw_logs": split_records
            }
            
            # Commit to main database ledger file
            self.ledger.append(master_payload)
            with open(self.json_path, "w") as f:
                json.dump(self.ledger, f, indent=4)
        else:
            print("[INFO] --smallwodetailed active: Bypassing detailed_metrics_ledger.json write.")

        # ------------------------------------------------------------------
        # NEW: SAVE CLEAN COMPACT LEDGER MATRIX TO DISK
        # ------------------------------------------------------------------
        small_json_path = os.path.join(os.path.dirname(self.json_path), "small_metrics_ledger.json")
        small_history = []
        if os.path.exists(small_json_path):
            try:
                with open(small_json_path, "r") as sf:
                    small_history = json.load(sf)
            except Exception:
                pass
                
        small_payload = {
            "timestamp": master_payload["timestamp"],
            "model_architecture": run_metadata['model_name'],
            "dataset_name": run_metadata['dataset_name'],
            "macro_f1_mean": f1_mean,
            "macro_f1_std": f1_std,
            "accuracy_mean": round(float(np.mean(acc_list)), 2),
            "accuracy_std": round(float(np.std(acc_list)), 2),
            "hyperparameters": {
                "layers": run_metadata.get('num_layers', 'N/A'),
                "hidden_dim": run_metadata['hidden_dim'],
                "lr": run_metadata.get('lr', 'N/A'),
                "dropout": run_metadata.get('dropout', 'N/A')
            }
        }
        small_history.append(small_payload)
        with open(small_json_path, "w") as sf:
            json.dump(small_history, sf, indent=4)
        # ------------------------------------------------------------------
            
        # Trigger aggregated chart execution
        self.generate_aggregated_plots(
            run_metadata['model_name'], run_metadata['dataset_name'], run_metadata['hidden_dim'],
            y_trues_all_splits, y_preds_all_splits, mean_class_f1s, std_class_f1s, num_classes
        )
        
        # ------------------------------------------------------------------
        # CONDITIONALLY TRAFFIC CONSOLE REPORTERS BLOCK
        # ------------------------------------------------------------------
        if main.IS_SMALL_REPORT_MODE:
            # Output only the high-utility concise summary line to terminal
            print(f"[CONCISE REPORT] Architecture: {run_metadata['model_name'].upper():<10} | Dataset: {run_metadata['dataset_name']:<25} | Final Macro F1 Score: {f1_mean}% \u00b1 {f1_std}%")
            print("="*85)
        else:
            # Full verbose report including minority class metrics listing blocks
            dataset_name_lower = run_metadata['dataset_name'].lower()
            print("\n" + "="*75)
            print(f"GRANULAR MINORITY CLASS CROSS-VALIDATION SUMMARY (10 SPLITS)")
            print(f"Model: {run_metadata['model_name'].upper()} | Dataset: {run_metadata['dataset_name'].upper()}")
            print("-"*75)
            print(f"{'Class ID':<10}{'Dependency Role':<18}{'Mean Precision':<16}{'Mean Recall':<13}{'Mean F1 (LaTeX)':<22}")
            print("-"*75)
            for c in range(num_classes):
                role_str = ROMAN_CLASS_MAP[c] if ("roman" in dataset_name_lower and c < len(ROMAN_CLASS_MAP)) else f"class_{c}"
                print(f"{c:<10}{role_str:<18}{mean_class_precs[c]:<16.2f}{mean_class_recs[c]:<13.2f}${mean_class_f1s[c]:.2f} \\pm {std_class_f1s[c]:.2f}$")
            print("="*75)

