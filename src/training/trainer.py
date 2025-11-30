"""
Training utilities and trainer class
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from typing import Dict, Optional
from pathlib import Path

from ..training.metrics import calculate_metrics


class Trainer:
    """
    Generic trainer for classification and anomaly detection models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        save_dir: str = "experiments/checkpoints",
        log_dir: str = "experiments/logs",
        use_multi_gpu: bool = False,
    ):
        # Handle multi-GPU setup
        if use_multi_gpu and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"Using DataParallel on {num_gpus} GPUs")
                self.model = nn.DataParallel(model)
                self.model = self.model.to(device)
                self.is_parallel = True
            else:
                self.model = model.to(device)
                self.is_parallel = False
        else:
            self.model = model.to(device)
            self.is_parallel = False
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }
        
        self.best_val_score = 0.0
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            if isinstance(outputs, tuple):
                # For models that return multiple outputs (e.g., Autoencoder, VAE)
                loss = self._calculate_loss(outputs, images, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            
            # For autoencoders, calculate reconstruction error as anomaly score
            if isinstance(outputs, tuple) and len(outputs[0].shape) == 4:
                # Autoencoder: outputs[0] is reconstruction (image)
                recon = outputs[0]
                # Calculate per-sample reconstruction error (MSE)
                recon_error = torch.mean((recon - images) ** 2, dim=(1, 2, 3))  # [batch_size]
                all_preds.append(recon_error.detach())
                all_labels.append(labels.detach())
            else:
                # Classification model: use outputs directly
                if isinstance(outputs, tuple):
                    preds = outputs[0]  # Use first output
                else:
                    preds = outputs
                all_preds.append(preds.detach())
                all_labels.append(labels.detach())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # For autoencoders, convert reconstruction error to anomaly predictions
            if len(all_preds.shape) == 1:  # 1D tensor = reconstruction errors
                # Use median reconstruction error as threshold
                threshold = torch.median(all_preds).item()
                # High error = anomaly (1), low error = normal (0)
                # Invert: lower error should predict normal (0), higher error predicts abnormal (1)
                # But we need to normalize and invert the scores for proper thresholding
                # For now, use a simple approach: error > threshold = anomaly
                anomaly_scores = all_preds.cpu().numpy()
                # Normalize to [0, 1] range for metric calculation
                min_err = anomaly_scores.min()
                max_err = anomaly_scores.max()
                if max_err > min_err:
                    anomaly_scores = (anomaly_scores - min_err) / (max_err - min_err)
                else:
                    anomaly_scores = np.zeros_like(anomaly_scores)
                # Convert to predictions (higher score = more anomalous)
                all_preds = torch.tensor(anomaly_scores, dtype=torch.float32)
            
            metrics = calculate_metrics(all_labels, all_preds)
        else:
            metrics = {}
        
        avg_loss = total_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'metrics': metrics,
        }
    
    def validate(self) -> Dict:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if isinstance(outputs, tuple):
                    loss = self._calculate_loss(outputs, images, labels)
                else:
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # For autoencoders, calculate reconstruction error as anomaly score
                if isinstance(outputs, tuple) and len(outputs[0].shape) == 4:
                    # Autoencoder: outputs[0] is reconstruction (image)
                    recon = outputs[0]
                    # Calculate per-sample reconstruction error (MSE)
                    recon_error = torch.mean((recon - images) ** 2, dim=(1, 2, 3))  # [batch_size]
                    all_preds.append(recon_error.detach())
                    all_labels.append(labels.detach())
                else:
                    # Classification model: use outputs directly
                    if isinstance(outputs, tuple):
                        preds = outputs[0]  # Use first output
                    else:
                        preds = outputs
                    all_preds.append(preds.detach())
                    all_labels.append(labels.detach())
                
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # For autoencoders, convert reconstruction error to anomaly predictions
            if len(all_preds.shape) == 1:  # 1D tensor = reconstruction errors
                # Normalize reconstruction errors to [0, 1] for metric calculation
                anomaly_scores = all_preds.cpu().numpy()
                min_err = anomaly_scores.min()
                max_err = anomaly_scores.max()
                if max_err > min_err:
                    anomaly_scores = (anomaly_scores - min_err) / (max_err - min_err)
                else:
                    anomaly_scores = np.zeros_like(anomaly_scores)
                # Convert to predictions (higher score = more anomalous)
                all_preds = torch.tensor(anomaly_scores, dtype=torch.float32)
            
            metrics = calculate_metrics(all_labels, all_preds)
        else:
            metrics = {}
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'metrics': metrics,
        }
    
    def _calculate_loss(self, outputs: tuple, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate loss for models with multiple outputs"""
        # For VAE: (recon, mu, logvar)
        if len(outputs) == 3:
            recon, mu, logvar = outputs
            # Reconstruction loss: compare reconstruction with input images
            recon_loss = nn.functional.mse_loss(recon, images, reduction='sum')
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return (recon_loss + kl_loss) / images.size(0)
        # For Autoencoder: (recon, z)
        elif len(outputs) == 2:
            recon, z = outputs
            # Reconstruction loss: compare reconstruction with input images
            return nn.functional.mse_loss(recon, images, reduction='mean')
        else:
            # Fallback to standard loss
            return self.criterion(outputs[0], labels)
    
    def train(self, num_epochs: int, save_best: bool = True, metric_name: str = "auc_roc_macro"):
        """Train the model"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {type(self.model).__name__}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_results = self.train_epoch()
            
            # Validate
            val_results = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_metrics'].append(train_results['metrics'])
            self.history['val_metrics'].append(val_results['metrics'])
            
            # Print results
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_results['loss']:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}")
            
            # Print key metrics
            val_metrics = val_results['metrics']
            # Always show accuracy
            if 'accuracy' in val_metrics:
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            if 'hamming_accuracy' in val_metrics:
                print(f"Val Hamming Accuracy: {val_metrics['hamming_accuracy']:.4f}")
            # Show AUC-ROC if available
            if 'auc_roc_macro' in val_metrics:
                print(f"Val AUC-ROC (macro): {val_metrics['auc_roc_macro']:.4f}")
            elif 'auc_roc' in val_metrics:
                print(f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
            # Show F1 if available
            if 'f1_macro' in val_metrics:
                print(f"Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            elif 'f1' in val_metrics:
                print(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Save checkpoint
            if save_best:
                # Get metric value - prefer accuracy, then AUC-ROC, then loss
                if metric_name in val_metrics:
                    score = val_metrics[metric_name]
                elif 'accuracy' in val_metrics:
                    score = val_metrics['accuracy']
                elif 'auc_roc_macro' in val_metrics:
                    score = val_metrics['auc_roc_macro']
                elif 'auc_roc' in val_metrics:
                    score = val_metrics['auc_roc']
                else:
                    score = -val_results['loss']  # Use negative loss as score
                
                if score > self.best_val_score:
                    self.best_val_score = score
                    self.save_checkpoint(f"best_model.pth")
                    metric_display = metric_name if metric_name in val_metrics else "loss"
                    print(f"New best model saved! ({metric_display}: {score:.4f})")
            
            # Save latest checkpoint
            self.save_checkpoint("latest_model.pth")
            
            print("-" * 60)
        
        # Save training history
        self.save_history()
        if val_results['metrics']:
            print(f"\nTraining completed! Best {metric_name}: {self.best_val_score:.4f}")
        else:
            print(f"\nTraining completed! Best loss: {-self.best_val_score:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        # Handle DataParallel: save model.module.state_dict() instead of model.state_dict()
        if self.is_parallel:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'history': self.history,
            'is_parallel': self.is_parallel,
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        
        # Handle DataParallel: load into model.module if parallel, otherwise direct
        state_dict = checkpoint['model_state_dict']
        if self.is_parallel:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_score = checkpoint['best_val_score']
        self.history = checkpoint['history']
        self.current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_history(self):
        """Save training history to JSON"""
        # Convert tensors to lists for JSON serialization
        history_json = {}
        for key, values in self.history.items():
            if key in ['train_metrics', 'val_metrics']:
                # Convert metrics dicts to JSON-serializable format
                history_json[key] = [
                    {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                     for k, v in m.items()}
                    for m in values
                ]
            else:
                history_json[key] = [float(v) for v in values]
        
        with open(self.log_dir / "training_history.json", 'w') as f:
            json.dump(history_json, f, indent=2)

