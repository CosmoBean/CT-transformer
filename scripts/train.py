#!/usr/bin/env python3
"""
Main training script for chest X-ray anomaly detection
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml

from src.data import ChestXRayDataset, AnomalyDetectionDataset
from src.models import (
    VisionTransformerClassifier,
    EfficientNetClassifier,
    ResNetClassifier,
    SwinTransformerClassifier,
    Autoencoder,
    VariationalAutoencoder,
)
from src.training import Trainer
from src.utils import load_config


def create_model(config):
    """Create model based on config"""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    pretrained = config['model'].get('pretrained', True)
    
    if model_name == "vit_base":
        model = VisionTransformerClassifier(
            num_classes=num_classes,
            img_size=config['data']['image_size'],
            pretrained=pretrained,
        )
    elif model_name.startswith("efficientnet"):
        model = EfficientNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
        )
    elif model_name.startswith("resnet"):
        model = ResNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
        )
    elif model_name.startswith("swin"):
        model = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
        )
    elif model_name == "autoencoder":
        model = Autoencoder(
            input_size=config['data']['image_size'],
            latent_dim=config['anomaly']['latent_dim'],
        )
    elif model_name == "vae":
        model = VariationalAutoencoder(
            input_size=config['data']['image_size'],
            latent_dim=config['anomaly']['latent_dim'],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train chest X-ray anomaly detection model')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Override model name from config')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs from config')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    # Set device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Using device: {device}")
    
    # Create datasets
    data_dir = Path(config['data']['data_dir'])
    train_csv = config['data'].get('train_csv')
    if train_csv:
        csv_path = data_dir / train_csv
        csv_path = str(csv_path) if csv_path.exists() else None
    else:
        # Try default train.csv
        default_csv = data_dir / 'train.csv'
        csv_path = str(default_csv) if default_csv.exists() else None
    
    train_split = config['data'].get('train_split', 0.8)
    val_split = config['data'].get('val_split', 0.2)
    seed = config.get('seed', 42)
    
    if config['data']['mode'] == 'anomaly' and config['model']['name'] in ['autoencoder', 'vae']:
        # Anomaly detection mode
        train_dataset = AnomalyDetectionDataset(
            data_dir=str(data_dir),
            csv_path=csv_path,
            image_size=config['data']['image_size'],
            split="train",
            normal_only=config['anomaly']['normal_only'],
            train_split=train_split,
            val_split=val_split,
            seed=seed,
        )
        val_dataset = AnomalyDetectionDataset(
            data_dir=str(data_dir),
            csv_path=csv_path,
            image_size=config['data']['image_size'],
            split="val",  # Use val split from training data
            normal_only=False,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
        )
    else:
        # Classification mode
        train_dataset = ChestXRayDataset(
            data_dir=str(data_dir),
            csv_path=csv_path,
            image_size=config['data']['image_size'],
            split="train",
            mode=config['data']['mode'],
            train_split=train_split,
            val_split=val_split,
            seed=seed,
        )
        val_dataset = ChestXRayDataset(
            data_dir=str(data_dir),
            csv_path=csv_path,
            image_size=config['data']['image_size'],
            split="val",  # Use val split from training data
            mode=config['data']['mode'],
            train_split=train_split,
            val_split=val_split,
            seed=seed,
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(config)
    print(f"\nModel: {config['model']['name']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    if config['training']['criterion'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif config['training']['criterion'] == 'focal':
        # Focal loss implementation would go here
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizer (ensure numeric types)
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Create scheduler (ensure numeric types)
    num_epochs = int(config['training']['num_epochs'])
    
    if config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
        )
    elif config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1,
        )
    elif config['training']['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
        )
    else:
        scheduler = None
    
    # Check for multi-GPU usage
    use_multi_gpu = config.get('use_multi_gpu', False)
    if use_multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Multi-GPU training enabled: {num_gpus} GPUs available")
        if num_gpus > 1:
            # Increase batch size proportionally for multi-GPU
            effective_batch_size = config['data']['batch_size'] * num_gpus
            print(f"Effective batch size: {effective_batch_size} (batch_size={config['data']['batch_size']} × {num_gpus} GPUs)")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=config['save_dir'],
        log_dir=config['log_dir'],
        use_multi_gpu=use_multi_gpu,
    )
    
    # Train
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_best=config['training']['save_best'],
        metric_name=config['training']['metric_name'],
    )
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

