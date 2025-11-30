#!/usr/bin/env python3
"""
Comprehensive validation script
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

def test_data_loader():
    """Test data loader with batches"""
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)
    
    from src.data import ChestXRayDataset
    
    dataset = ChestXRayDataset(
        data_dir="data",
        csv_path=None,
        image_size=224,
        split="train",
        mode="classification",
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    batch = next(iter(loader))
    
    print(f"Batch loaded successfully")
    print(f"  - Images shape: {batch['image'].shape}")
    print(f"  - Labels shape: {batch['labels'].shape}")
    print(f"  - Image IDs: {len(batch['image_id'])} samples")
    print(f"  - Image dtype: {batch['image'].dtype}")
    print(f"  - Labels dtype: {batch['labels'].dtype}")
    
    return True


def test_all_models():
    """Test all model architectures"""
    print("\n" + "=" * 60)
    print("Testing All Model Architectures")
    print("=" * 60)
    
    from src.models import (
        Autoencoder, VariationalAutoencoder,
        VisionTransformerClassifier, EfficientNetClassifier,
        ResNetClassifier, SwinTransformerClassifier
    )
    
    models = {
        "autoencoder": Autoencoder(input_size=224, latent_dim=128),
        "vae": VariationalAutoencoder(input_size=224, latent_dim=128),
        "vit": VisionTransformerClassifier(num_classes=15, pretrained=False),
        "efficientnet": EfficientNetClassifier(num_classes=15, pretrained=False),
        "resnet": ResNetClassifier(num_classes=15, pretrained=False),
        "swin": SwinTransformerClassifier(num_classes=15, pretrained=False),
    }
    
    dummy = torch.randn(2, 3, 224, 224)
    
    for name, model in models.items():
        try:
            if name in ["autoencoder", "vae"]:
                output = model(dummy)
                print(f"{name}: output type = {type(output)}")
            else:
                output = model(dummy)
                print(f"{name}: output shape = {output.shape}")
        except Exception as e:
            print(f"{name} failed: {e}")
            return False
    
    return True


def test_training_script_components():
    """Test training script components"""
    print("\n" + "=" * 60)
    print("Testing Training Script Components")
    print("=" * 60)
    
    from scripts.train import create_model
    from src.utils import load_config
    
    config = load_config("configs/default_config.yaml")
    
    test_models = ["efficientnet_b3", "resnet50", "vit_base"]
    
    for model_name in test_models:
        try:
            config["model"]["name"] = model_name
            model = create_model(config)
            print(f"{model_name} created via create_model()")
        except Exception as e:
            print(f"{model_name} failed: {e}")
            return False
    
    return True


def test_checkpoint_saving():
    """Test checkpoint saving"""
    print("\n" + "=" * 60)
    print("Testing Checkpoint System")
    print("=" * 60)
    
    from src.models import EfficientNetClassifier
    from src.training import Trainer
    from src.data import ChestXRayDataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    import os
    
    # Create minimal setup
    dataset = ChestXRayDataset(
        data_dir="data",
        csv_path=None,
        image_size=224,
        split="train",
        mode="classification",
    )
    
    # Use small subset for testing
    subset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))
    loader = DataLoader(subset, batch_size=4, shuffle=False)
    
    model = EfficientNetClassifier(num_classes=15, pretrained=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device="cpu",  # Use CPU for quick test
        save_dir="experiments/test_checkpoints",
        log_dir="experiments/test_logs",
    )
    
    # Test checkpoint saving
    trainer.save_checkpoint("test_model.pth")
    
    checkpoint_path = Path("experiments/test_checkpoints/test_model.pth")
    if checkpoint_path.exists():
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Test loading
        trainer.load_checkpoint("test_model.pth")
        print(f"Checkpoint loaded successfully")
        
        # Cleanup
        checkpoint_path.unlink()
        return True
    else:
        print(f"Checkpoint not found")
        return False


def main():
    """Run all validation tests"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VALIDATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Data Loader", test_data_loader()))
    results.append(("All Models", test_all_models()))
    results.append(("Training Script", test_training_script_components()))
    results.append(("Checkpoint System", test_checkpoint_saving()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} validations passed")
    
    if passed == total:
        print("\nAll validations passed! System is ready for training.")
        return 0
    else:
        print(f"\n{total - passed} validation(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

