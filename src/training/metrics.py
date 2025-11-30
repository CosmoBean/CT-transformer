"""
Evaluation metrics for multi-label classification and anomaly detection
"""
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
)
from typing import Dict, Tuple, Optional


def calculate_auc_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Calculate AUC-ROC score
    
    Args:
        y_true: Ground truth labels (binary or multi-label)
        y_pred: Predicted probabilities
        average: Averaging strategy ('macro', 'micro', 'weighted', None)
    
    Returns:
        AUC-ROC score
    """
    try:
        # Check if all labels are zeros (no positive samples)
        if y_true.ndim == 1:
            if np.sum(y_true) == 0:
                return 0.0  # No positive samples
            return roc_auc_score(y_true, y_pred, average=average)
        else:
            # Multi-label: check if any class has positive samples
            if np.sum(y_true) == 0:
                return 0.0  # No positive samples in any class
            return roc_auc_score(y_true, y_pred, average=average)
    except (ValueError, Exception):
        # Handle case where only one class is present or other errors
        return 0.0


def calculate_auc_pr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Calculate AUC-PR (Average Precision) score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        average: Averaging strategy
    
    Returns:
        AUC-PR score
    """
    try:
        # Check if all labels are zeros
        if y_true.ndim == 1:
            if np.sum(y_true) == 0:
                return 0.0
            return average_precision_score(y_true, y_pred, average=average)
        else:
            if np.sum(y_true) == 0:
                return 0.0
            return average_precision_score(y_true, y_pred, average=average)
    except (ValueError, Exception):
        return 0.0


def calculate_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Calculate accuracy score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Threshold for binarization
    
    Returns:
        Accuracy score
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    try:
        if y_true.ndim == 1:
            # Binary classification: simple accuracy
            return accuracy_score(y_true, y_pred_binary)
        else:
            # Multi-label: exact match accuracy (all labels must match)
            # Also compute hamming accuracy (per-label accuracy)
            exact_match = np.all(y_true == y_pred_binary, axis=1).mean()
            return exact_match
    except ValueError:
        return 0.0


def calculate_hamming_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Calculate Hamming accuracy (per-label accuracy) for multi-label classification
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Threshold for binarization
    
    Returns:
        Hamming accuracy (average per-label accuracy)
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    try:
        if y_true.ndim == 1:
            return accuracy_score(y_true, y_pred_binary)
        else:
            # Average accuracy across all labels
            return accuracy_score(y_true.flatten(), y_pred_binary.flatten())
    except ValueError:
        return 0.0


def calculate_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = "macro",
) -> float:
    """
    Calculate F1 score
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Threshold for binarization
        average: Averaging strategy
    
    Returns:
        F1 score
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    try:
        if y_true.ndim == 1:
            return f1_score(y_true, y_pred_binary, average=average, zero_division=0)
        else:
            return f1_score(y_true, y_pred_binary, average=average, zero_division=0)
    except ValueError:
        return 0.0


def calculate_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-label classification
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted logits or probabilities (N, num_classes)
        threshold: Threshold for binarization
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Apply sigmoid if needed (if logits)
    if y_pred.min() < 0 or y_pred.max() > 1:
        y_pred = 1 / (1 + np.exp(-y_pred))  # Sigmoid
    
    # Binarize predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {}
    
    # Check if all labels are zeros (no annotations)
    has_positive_labels = np.sum(y_true) > 0
    
    if not has_positive_labels:
        # No positive labels - return default metrics
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            metrics['accuracy'] = calculate_accuracy(y_true, y_pred, threshold)
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
            metrics['f1'] = 0.0
        else:
            metrics['accuracy'] = calculate_accuracy(y_true, y_pred, threshold)
            metrics['hamming_accuracy'] = calculate_hamming_accuracy(y_true, y_pred, threshold)
            metrics['auc_roc_macro'] = 0.0
            metrics['auc_roc_micro'] = 0.0
            metrics['auc_pr_macro'] = 0.0
            metrics['auc_pr_micro'] = 0.0
            metrics['f1_macro'] = 0.0
            metrics['f1_micro'] = 0.0
        return metrics
    
    # Handle binary vs multi-label
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        # Binary classification
        metrics['accuracy'] = calculate_accuracy(y_true, y_pred, threshold)
        metrics['auc_roc'] = calculate_auc_roc(y_true, y_pred, average='macro')
        metrics['auc_pr'] = calculate_auc_pr(y_true, y_pred, average='macro')
        metrics['f1'] = calculate_f1_score(y_true, y_pred, threshold, average='binary')
    else:
        # Multi-label classification
        metrics['accuracy'] = calculate_accuracy(y_true, y_pred, threshold)  # Exact match
        metrics['hamming_accuracy'] = calculate_hamming_accuracy(y_true, y_pred, threshold)  # Per-label
        metrics['auc_roc_macro'] = calculate_auc_roc(y_true, y_pred, average='macro')
        metrics['auc_roc_micro'] = calculate_auc_roc(y_true, y_pred, average='micro')
        metrics['auc_pr_macro'] = calculate_auc_pr(y_true, y_pred, average='macro')
        metrics['auc_pr_micro'] = calculate_auc_pr(y_true, y_pred, average='micro')
        metrics['f1_macro'] = calculate_f1_score(y_true, y_pred, threshold, average='macro')
        metrics['f1_micro'] = calculate_f1_score(y_true, y_pred, threshold, average='micro')
    
    # Per-class metrics for multi-label
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        per_class_auc = []
        per_class_f1 = []
        for i in range(y_true.shape[1]):
            try:
                auc = calculate_auc_roc(y_true[:, i], y_pred[:, i])
                per_class_auc.append(auc)
                f1 = calculate_f1_score(y_true[:, i], y_pred[:, i], threshold)
                per_class_f1.append(f1)
            except:
                per_class_auc.append(0.0)
                per_class_f1.append(0.0)
        
        metrics['per_class_auc'] = per_class_auc
        metrics['per_class_f1'] = per_class_f1
    
    return metrics


def calculate_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate reconstruction error for anomaly detection
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
    
    Returns:
        Reconstruction error per sample
    """
    # MSE per sample
    error = torch.mean((original - reconstructed) ** 2, dim=(1, 2, 3))
    return error

