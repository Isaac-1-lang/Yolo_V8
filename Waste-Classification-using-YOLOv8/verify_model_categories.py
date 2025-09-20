#!/usr/bin/env python3
"""
Script to verify the categories in your trained YOLOv8 model
"""

import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def verify_model_categories():
    """Verify the categories in the trained model"""
    
    print("🔍 Verifying YOLOv8 Model Categories")
    print("=" * 50)
    
    # Load the data configuration
    with open('data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("📊 Expected Categories from data.yaml:")
    for i, category in enumerate(data_config['names']):
        print(f"   {i}: {category}")
    print(f"   Total categories: {data_config['nc']}")
    
    # Check if the trained model exists
    model_path = Path("streamlit-detection-tracking - app/weights/best.pt")
    if not model_path.exists():
        print(f"\n❌ Trained model not found at {model_path}")
        print("   Please train a model first using train_new_model.py")
        return
    
    print(f"\n🤖 Loading trained model: {model_path}")
    
    try:
        # Load the trained model
        model = YOLO(str(model_path))
        
        # Get model info
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model.model).__name__}")
        
        # Get the class names from the model
        if hasattr(model.model, 'names'):
            model_classes = model.model.names
            print(f"\n📋 Categories in trained model:")
            for class_id, class_name in model_classes.items():
                print(f"   {class_id}: {class_name}")
            
            # Compare with expected categories
            expected_categories = data_config['names']
            model_categories = list(model_classes.values())
            
            print(f"\n🔍 Category Comparison:")
            if expected_categories == model_categories:
                print("   ✅ Model categories match expected categories!")
            else:
                print("   ⚠️  Model categories don't match expected categories:")
                print(f"   Expected: {expected_categories}")
                print(f"   Model has: {model_categories}")
                
        else:
            print("   ⚠️  Could not retrieve class names from model")
            
        # Test inference on a sample image
        print(f"\n🧪 Testing model inference...")
        test_image_path = "dataset/train/images/0.jpg"
        if Path(test_image_path).exists():
            results = model.predict(test_image_path, conf=0.25, verbose=False)
            print(f"   ✅ Inference test successful!")
            
            # Show detection results
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"   📊 Detected {len(result.boxes)} objects:")
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.model.names[class_id]
                        print(f"      - {class_name}: {confidence:.2f}")
                else:
                    print(f"   ℹ️  No objects detected in test image")
        else:
            print(f"   ⚠️  Test image not found at {test_image_path}")
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    print(f"\n✅ Model verification completed!")

if __name__ == "__main__":
    verify_model_categories()
