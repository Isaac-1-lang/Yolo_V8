#!/usr/bin/env python3
"""
Category Management Utility for Waste Classification YOLOv8 Model
This script helps you manage and verify the categories in your waste classification model.
"""

import yaml
import json
from pathlib import Path
from ultralytics import YOLO
import argparse

class WasteCategoryManager:
    """Manages waste classification categories for YOLOv8 model"""
    
    def __init__(self, data_yaml_path="data.yaml", model_path=None):
        self.data_yaml_path = data_yaml_path
        self.model_path = model_path
        self.categories = self.load_categories()
        
    def load_categories(self):
        """Load categories from data.yaml"""
        try:
            with open(self.data_yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            return {
                'names': config['names'],
                'nc': config['nc'],
                'path': config.get('path', '')
            }
        except Exception as e:
            print(f"❌ Error loading categories from {self.data_yaml_path}: {e}")
            return None
    
    def display_categories(self):
        """Display current categories"""
        if not self.categories:
            print("❌ No categories loaded")
            return
            
        print("🗑️  Current Waste Categories:")
        print("=" * 40)
        for i, category in enumerate(self.categories['names']):
            print(f"   {i}: {category}")
        print(f"\n   Total categories: {self.categories['nc']}")
        print(f"   Dataset path: {self.categories['path']}")
    
    def update_categories(self, new_categories):
        """Update categories in data.yaml"""
        if not isinstance(new_categories, list):
            print("❌ Categories must be a list")
            return False
            
        if len(new_categories) != self.categories['nc']:
            print(f"❌ Number of categories must be {self.categories['nc']}")
            return False
            
        try:
            # Update the configuration
            config = {
                'names': new_categories,
                'nc': len(new_categories),
                'path': self.categories['path'],
                'test': 'test/images',
                'train': 'train/images',
                'val': 'val/images'
            }
            
            # Save to data.yaml
            with open(self.data_yaml_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Reload categories
            self.categories = self.load_categories()
            print("✅ Categories updated successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error updating categories: {e}")
            return False
    
    def verify_model_categories(self):
        """Verify that the trained model uses the correct categories"""
        if not self.model_path or not Path(self.model_path).exists():
            print(f"❌ Model not found at {self.model_path}")
            return False
            
        try:
            model = YOLO(self.model_path)
            
            if hasattr(model.model, 'names'):
                model_classes = model.model.names
                model_categories = list(model_classes.values())
                expected_categories = self.categories['names']
                
                print(f"\n🔍 Model Category Verification:")
                print(f"   Expected: {expected_categories}")
                print(f"   Model has: {model_categories}")
                
                if expected_categories == model_categories:
                    print("   ✅ Model categories match expected categories!")
                    return True
                else:
                    print("   ⚠️  Model categories don't match expected categories!")
                    return False
            else:
                print("   ⚠️  Could not retrieve class names from model")
                return False
                
        except Exception as e:
            print(f"❌ Error verifying model: {e}")
            return False
    
    def test_inference(self, image_path, confidence=0.25):
        """Test model inference on an image"""
        if not self.model_path or not Path(self.model_path).exists():
            print(f"❌ Model not found at {self.model_path}")
            return None
            
        if not Path(image_path).exists():
            print(f"❌ Image not found at {image_path}")
            return None
            
        try:
            model = YOLO(self.model_path)
            results = model.predict(image_path, conf=confidence, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    print(f"\n📊 Detection Results:")
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.model.names[class_id]
                        print(f"   - {class_name}: {confidence:.2f}")
                    return results
                else:
                    print(f"   ℹ️  No objects detected")
                    return results
            else:
                print(f"   ℹ️  No results returned")
                return None
                
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            return None
    
    def export_categories(self, output_path="categories.json"):
        """Export categories to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.categories, f, indent=2)
            print(f"✅ Categories exported to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error exporting categories: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Waste Category Manager")
    parser.add_argument("--data-yaml", default="data.yaml", help="Path to data.yaml file")
    parser.add_argument("--model", help="Path to trained model file")
    parser.add_argument("--test-image", help="Path to test image for inference")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold for inference")
    parser.add_argument("--export", help="Export categories to JSON file")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = WasteCategoryManager(args.data_yaml, args.model)
    
    if not manager.categories:
        return
    
    # Display current categories
    manager.display_categories()
    
    # Verify model if provided
    if args.model:
        manager.verify_model_categories()
    
    # Test inference if image provided
    if args.test_image:
        manager.test_inference(args.test_image, args.confidence)
    
    # Export categories if requested
    if args.export:
        manager.export_categories(args.export)

if __name__ == "__main__":
    main()
