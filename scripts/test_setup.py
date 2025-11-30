#!/usr/bin/env python3
"""
Test script to validate the project setup
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import traceback

def test_imports():
    """Test all imports"""
    print("=" * 60)
    print("Test 1: Importing modules")
    print("=" * 60)
    
    try:
        from src.data import ChestXRayDataset, AnomalyDetectionDataset
        print("Data modules imported")
        
        from src.models import (
            VisionTransformerClassifier,
            EfficientNetClassifier,
            ResNetClassifier,
            SwinTransformerClassifier,
            Autoencoder,
            VariationalAutoencoder,
        )
        print("Model modules imported")
        
        from src.training import Trainer, calculate_metrics
        print("Training modules imported")
        
        from src.utils import load_config, visualize_predictions, plot_training_history
        print("Utility modules imported")
        
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading"""
    print("\n" + "=" * 60)
    print("Test 2: Data loading")
    print("=" * 60)
    
    try:
        from src.data import ChestXRayDataset
        
        data_dir = project_root / "data"
        csv_path = data_dir / "train_meta.csv"
        
        if not (data_dir / "train").exists():
            print(f"Warning: {data_dir / 'train'} does not exist")
            return False
        
        # Try to create dataset
        dataset = ChestXRayDataset(
            data_dir=str(data_dir),
            csv_path=str(csv_path) if csv_path.exists() else None,
            image_size=224,
            split="train",
            mode="classification",
        )
        
        print(f"Dataset created: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("Warning: Dataset is empty")
            return False
        
        # Test loading one sample
        sample = dataset[0]
        print(f"Sample loaded")
        print(f"  - Image shape: {sample['image'].shape}")
        print(f"  - Labels shape: {sample['labels'].shape}")
        print(f"  - Image ID: {sample['image_id']}")
        
        return True
    except Exception as e:
        print(f"Data loading failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    print("\n" + "=" * 60)
    print("Test 3: Model creation")
    print("=" * 60)
    
    try:
        from src.models import EfficientNetClassifier
        
        model = EfficientNetClassifier(
            num_classes=15,
            model_name="efficientnet_b3",
            pretrained=False,  # Don't download for test
        )
        
        print(f"Model created: {type(model).__name__}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"Model creation failed: {e}")
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("Test 4: Configuration loading")
    print("=" * 60)
    
    try:
        from src.utils import load_config
        
        config_path = project_root / "configs" / "default_config.yaml"
        
        if not config_path.exists():
            print(f"Warning: {config_path} does not exist")
            return False
        
        config = load_config(str(config_path))
        
        print(f"Config loaded: {len(config)} sections")
        print(f"  - Model: {config['model']['name']}")
        print(f"  - Data dir: {config['data']['data_dir']}")
        print(f"  - Batch size: {config['data']['batch_size']}")
        
        return True
    except Exception as e:
        print(f"Config loading failed: {e}")
        traceback.print_exc()
        return False


def test_training_utilities():
    """Test training utilities"""
    print("\n" + "=" * 60)
    print("Test 5: Training utilities")
    print("=" * 60)
    
    try:
        from src.training import calculate_metrics
        import torch
        
        # Create dummy predictions and labels
        y_true = torch.randint(0, 2, (10, 15)).float()
        y_pred = torch.randn(10, 15)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        print(f"Metrics calculated")
        print(f"  - Available metrics: {list(metrics.keys())[:5]}...")
        
        return True
    except Exception as e:
        print(f"Training utilities test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PROJECT SETUP VALIDATION")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Training Utilities", test_training_utilities()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! Project setup is working correctly.")
        return 0
    else:
        print(f"\n{total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

