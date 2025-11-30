#!/usr/bin/env python3
"""
Train all models and log accuracy results
"""
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Models to train
MODELS = [
    "efficientnet_b3",
    "resnet50",
    "vit_base",
    "swin_base_patch4_window7_224",
    "autoencoder",
    "vae",
]

# Results file
RESULTS_FILE = project_root / "experiments" / "model_results.json"
LOG_FILE = project_root / "experiments" / "training_log.txt"

def log_message(message, log_file=LOG_FILE):
    """Log message to file and print"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(message)
    with open(log_file, "a") as f:
        f.write(log_entry)

def train_model(model_name, epochs=10):
    """Train a single model and return results"""
    log_message(f"\n{'='*60}")
    log_message(f"Training model: {model_name}")
    log_message(f"{'='*60}")
    
    start_time = time.time()
    
    # Run training
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "train.py"),
        "--model", model_name,
        "--epochs", str(epochs),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 6,  # 6 hour timeout per model
        )
        
        elapsed_time = time.time() - start_time
        
        # Parse output for metrics
        output = result.stdout + result.stderr
        metrics = parse_training_output(output, model_name)
        metrics['training_time'] = elapsed_time / 60  # minutes
        metrics['status'] = 'completed' if result.returncode == 0 else 'failed'
        metrics['return_code'] = result.returncode
        
        if result.returncode != 0:
            log_message(f"ERROR: Training failed for {model_name}")
            log_message(f"Error output: {result.stderr[:500]}")
        else:
            log_message(f"Completed {model_name} in {elapsed_time/60:.1f} minutes")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        log_message(f"TIMEOUT: {model_name} exceeded 6 hour limit")
        return {
            'model': model_name,
            'status': 'timeout',
            'training_time': 360,
        }
    except Exception as e:
        log_message(f"ERROR: Exception training {model_name}: {str(e)}")
        return {
            'model': model_name,
            'status': 'error',
            'error': str(e),
        }

def parse_training_output(output, model_name):
    """Parse training output to extract metrics"""
    metrics = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Look for validation metrics
    lines = output.split('\n')
    for i, line in enumerate(lines):
        # Look for accuracy
        if 'Val Accuracy:' in line:
            try:
                acc = float(line.split('Val Accuracy:')[1].strip())
                metrics['val_accuracy'] = acc
            except:
                pass
        
        if 'Val Hamming Accuracy:' in line:
            try:
                hamm_acc = float(line.split('Val Hamming Accuracy:')[1].strip())
                metrics['val_hamming_accuracy'] = hamm_acc
            except:
                pass
        
        if 'Val AUC-ROC' in line:
            try:
                auc = float(line.split(':')[1].strip())
                if 'macro' in line:
                    metrics['val_auc_roc_macro'] = auc
                else:
                    metrics['val_auc_roc'] = auc
            except:
                pass
        
        if 'Val F1' in line:
            try:
                f1 = float(line.split(':')[1].strip())
                if 'macro' in line:
                    metrics['val_f1_macro'] = f1
                else:
                    metrics['val_f1'] = f1
            except:
                pass
        
        # Look for best model message
        if 'New best model saved!' in line:
            try:
                # Extract metric value
                parts = line.split('(')[1].split(')')[0]
                if ':' in parts:
                    metric_name, value = parts.split(':')
                    metrics['best_metric_name'] = metric_name.strip()
                    metrics['best_metric_value'] = float(value.strip())
            except:
                pass
        
        # Look for final training message
        if 'Training completed!' in line:
            try:
                if 'Best' in line:
                    parts = line.split('Best')[1].strip()
                    if ':' in parts:
                        metric_name, value = parts.split(':')
                        metrics['final_metric_name'] = metric_name.strip()
                        metrics['final_metric_value'] = float(value.strip())
            except:
                pass
    
    return metrics

def save_results(all_results):
    """Save results to JSON file"""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if any
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}
    
    # Update with new results
    existing['last_updated'] = datetime.now().isoformat()
    existing['models'] = all_results
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(existing, f, indent=2)
    
    log_message(f"\nResults saved to {RESULTS_FILE}")

def print_summary(all_results):
    """Print summary of all results"""
    log_message(f"\n{'='*60}")
    log_message("TRAINING SUMMARY")
    log_message(f"{'='*60}")
    
    # Create summary table
    log_message(f"\n{'Model':<30} {'Accuracy':<12} {'AUC-ROC':<12} {'F1':<12} {'Status':<12}")
    log_message("-" * 80)
    
    for result in all_results:
        model = result.get('model', 'unknown')
        acc = result.get('val_accuracy', result.get('val_hamming_accuracy', 'N/A'))
        auc = result.get('val_auc_roc_macro', result.get('val_auc_roc', 'N/A'))
        f1 = result.get('val_f1_macro', result.get('val_f1', 'N/A'))
        status = result.get('status', 'unknown')
        
        if isinstance(acc, float):
            acc = f"{acc:.4f}"
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"
        
        log_message(f"{model:<30} {str(acc):<12} {str(auc):<12} {str(f1):<12} {status:<12}")
    
    log_message(f"\n{'='*60}")

def main():
    """Main function to train all models"""
    log_message("="*60)
    log_message("Starting training for all models")
    log_message(f"Models to train: {', '.join(MODELS)}")
    log_message(f"Results will be saved to: {RESULTS_FILE}")
    log_message("="*60)
    
    all_results = []
    
    for i, model in enumerate(MODELS, 1):
        log_message(f"\n[{i}/{len(MODELS)}] Starting {model}...")
        
        # Train model
        result = train_model(model, epochs=10)
        all_results.append(result)
        
        # Save intermediate results
        save_results(all_results)
        
        log_message(f"Progress: {i}/{len(MODELS)} models completed")
    
    # Final summary
    print_summary(all_results)
    save_results(all_results)
    
    log_message(f"\nAll models trained! Results saved to {RESULTS_FILE}")
    log_message(f"Log file: {LOG_FILE}")

if __name__ == "__main__":
    main()

