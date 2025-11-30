"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional


def visualize_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Optional[List[str]] = None,
    num_samples: int = 8,
    threshold: float = 0.5,
):
    """
    Visualize model predictions
    
    Args:
        images: Input images (N, C, H, W)
        labels: Ground truth labels (N, num_classes)
        predictions: Model predictions (N, num_classes)
        class_names: List of class names
        num_samples: Number of samples to visualize
        threshold: Threshold for binarization
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(labels.shape[1])]
    
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Apply sigmoid if needed
    if predictions.min() < 0 or predictions.max() > 1:
        predictions = 1 / (1 + np.exp(-predictions))
    
    # Denormalize images (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Show image
        axes[i, 0].imshow(img, cmap='gray' if img.shape[2] == 1 else None)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Sample {i+1}')
        
        # Show predictions
        pred_binary = (predictions[i] >= threshold).astype(int)
        true_labels = labels[i].astype(int)
        
        # Create comparison
        comparison = []
        for j, (true, pred) in enumerate(zip(true_labels, pred_binary)):
            if true == 1 or pred == 1:
                status = "[+]" if true == pred else "[-]"
                comparison.append(f"{status} {class_names[j]}: {predictions[i][j]:.2f}")
        
        axes[i, 1].axis('off')
        axes[i, 1].text(0.1, 0.5, '\n'.join(comparison), 
                        fontsize=10, verticalalignment='center',
                        family='monospace')
        axes[i, 1].set_title('Predictions')
    
    plt.tight_layout()
    return fig


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        metrics = history['val_metrics']
        
        # Extract available metrics
        metric_names = []
        metric_values = []
        
        for metric_name in ['auc_roc_macro', 'f1_macro', 'auc_roc', 'f1']:
            if metric_name in metrics[0]:
                metric_names.append(metric_name)
                metric_values.append([m[metric_name] for m in metrics])
        
        if metric_values:
            for name, values in zip(metric_names, metric_values):
                axes[1].plot(values, label=name)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Validation Metrics')
            axes[1].legend()
            axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

