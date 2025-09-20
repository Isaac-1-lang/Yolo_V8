#!/usr/bin/env python3
"""
Training Script for Waste Classification with New Dataset
This script trains a YOLOv8 model on the new waste classification dataset.
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def check_gpu():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ GPU not available, using CPU")
        return False

def load_data_config():
    """Load and validate data configuration."""
    if not os.path.exists('data.yaml'):
        print("❌ data.yaml not found. Please run prepare_dataset.py first.")
        return None
    
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("📊 Dataset Configuration:")
    print(f"   Classes: {config['names']}")
    print(f"   Number of classes: {config['nc']}")
    print(f"   Dataset path: {config['path']}")
    
    return config

def train_model():
    """Train the YOLOv8 model."""
    
    print("🚀 Starting Waste Classification Training")
    print("=" * 50)
    
    # Check GPU availability
    use_gpu = check_gpu()
    
    # Load data configuration
    config = load_data_config()
    if config is None:
        return
    
    # Model selection
    model_size = 'yolov8s.pt'  # You can change to yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    print(f"\n🤖 Using model: {model_size}")
    
    # Load model
    model = YOLO(model_size)
    
    # Training parameters
    training_params = {
        'data': 'data.yaml',
        'epochs': 50,  # Increased for better training
        'imgsz': 640,
        'batch': 16 if use_gpu else 8,  # Adjust based on GPU memory
        'device': 'cuda' if use_gpu else 'cpu',
        'workers': 8 if use_gpu else 4,
        'project': 'runs/detect',
        'name': 'waste_classification_new',
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': 15,  # Early stopping patience
        'lr0': 0.01,  # Initial learning rate
        'lrf': 0.01,  # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,  # HSV augmentation
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,  # Rotation
        'translate': 0.1,  # Translation
        'scale': 0.5,  # Scaling
        'shear': 0.0,  # Shearing
        'perspective': 0.0,  # Perspective
        'flipud': 0.0,  # Flip up-down
        'fliplr': 0.5,  # Flip left-right
        'mosaic': 1.0,  # Mosaic augmentation
        'mixup': 0.0,  # Mixup augmentation
        'copy_paste': 0.0,  # Copy-paste augmentation
        'auto_augment': 'randaugment',  # Auto augmentation
        'erasing': 0.4,  # Random erasing
        'crop_fraction': 1.0,  # Image crop fraction
        'val': True,  # Validate during training
        'plots': True,  # Generate training plots
        'verbose': True,  # Verbose output
    }
    
    print("\n📋 Training Parameters:")
    for key, value in training_params.items():
        if key in ['data', 'epochs', 'imgsz', 'batch', 'device', 'lr0', 'patience']:
            print(f"   {key}: {value}")
    
    print(f"\n🏃 Starting training...")
    print("   This may take several hours depending on your hardware.")
    print("   You can monitor progress in the runs/detect/waste_classification_new directory.")
    
    try:
        # Start training
        results = model.train(**training_params)
        
        print("\n✅ Training completed successfully!")
        print(f"   Results saved to: runs/detect/waste_classification_new")
        print(f"   Best model: runs/detect/waste_classification_new/weights/best.pt")
        print(f"   Last model: runs/detect/waste_classification_new/weights/last.pt")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n📊 Final Metrics:")
            print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
            print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
            print(f"   Precision: {metrics.get('metrics/precision(B)', 'N/A'):.3f}")
            print(f"   Recall: {metrics.get('metrics/recall(B)', 'N/A'):.3f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        print("   Please check your dataset and try again.")
        return None

def validate_model(model_path='runs/detect/waste_classification_new/weights/best.pt'):
    """Validate the trained model."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"\n🔍 Validating model: {model_path}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate on validation set
    results = model.val(data='data.yaml', split='val')
    
    print("✅ Validation completed!")
    return results

def test_inference(model_path='runs/detect/waste_classification_new/weights/best.pt', 
                  test_image_path=None):
    """Test the model on a sample image."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"\n🧪 Testing inference with model: {model_path}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Test on a sample image if provided
    if test_image_path and os.path.exists(test_image_path):
        results = model.predict(test_image_path, conf=0.25, save=True)
        print(f"✅ Inference test completed! Results saved.")
    else:
        print("ℹ️  No test image provided. You can test inference manually using:")
        print(f"   yolo task=detect mode=predict model={model_path} source=path/to/image.jpg conf=0.25")

if __name__ == "__main__":
    print("🗑️  Waste Classification Training Script")
    print("=" * 50)
    
    # Check if dataset is prepared
    if not os.path.exists('data.yaml'):
        print("❌ Dataset not prepared. Please run prepare_dataset.py first.")
        exit(1)
    
    # Start training
    results = train_model()
    
    if results is not None:
        # Validate the model
        validate_model()
        
        # Test inference
        test_inference()
        
        print("\n🎉 Training pipeline completed!")
        print("\nNext steps:")
        print("1. Check the training results in runs/detect/waste_classification_new/")
        print("2. Use the best.pt model for inference")
        print("3. Update your Streamlit app to use the new model")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

