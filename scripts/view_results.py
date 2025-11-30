#!/usr/bin/env python3
"""
View training results
"""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
RESULTS_FILE = project_root / "experiments" / "model_results.json"

def main():
    if not RESULTS_FILE.exists():
        print(f"No results file found at {RESULTS_FILE}")
        print("Run 'make train-all' first to train all models.")
        return
    
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("MODEL TRAINING RESULTS")
    print("="*80)
    print(f"Last updated: {data.get('last_updated', 'Unknown')}")
    print()
    
    if 'models' not in data:
        print("No model results found.")
        return
    
    # Print table header
    print(f"{'Model':<35} {'Accuracy':<12} {'AUC-ROC':<12} {'F1':<12} {'Time (min)':<12} {'Status':<12}")
    print("-"*80)
    
    # Sort by accuracy (if available)
    models = data['models']
    sorted_models = sorted(
        models,
        key=lambda x: x.get('val_accuracy', x.get('val_hamming_accuracy', 0)) or 0,
        reverse=True
    )
    
    for result in sorted_models:
        model = result.get('model', 'unknown')
        acc = result.get('val_accuracy', result.get('val_hamming_accuracy', None))
        auc = result.get('val_auc_roc_macro', result.get('val_auc_roc', None))
        f1 = result.get('val_f1_macro', result.get('val_f1', None))
        time_min = result.get('training_time', None)
        status = result.get('status', 'unknown')
        
        # Format values
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        time_str = f"{time_min:.1f}" if time_min is not None else "N/A"
        
        print(f"{model:<35} {acc_str:<12} {auc_str:<12} {f1_str:<12} {time_str:<12} {status:<12}")
    
    print("="*80)

if __name__ == "__main__":
    main()

