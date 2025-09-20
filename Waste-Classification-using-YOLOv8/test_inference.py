#!/usr/bin/env python3
"""
Simple inference script for testing waste classification model
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

def test_waste_classification(model_path, image_path, confidence=0.25, save_result=True):
    """
    Test waste classification on an image
    
    Args:
        model_path (str): Path to the trained YOLOv8 model
        image_path (str): Path to the test image
        confidence (float): Confidence threshold for detection
        save_result (bool): Whether to save the result image
    """
    
    print("🗑️  Waste Classification Test")
    print("=" * 40)
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Display model categories
    if hasattr(model.model, 'names'):
        print(f"\n📋 Model Categories:")
        for class_id, class_name in model.model.names.items():
            print(f"   {class_id}: {class_name}")
    
    # Load and process image
    print(f"\n🖼️  Processing image: {image_path}")
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Run inference
    print(f"\n🔍 Running inference (confidence threshold: {confidence})...")
    try:
        results = model.predict(image, conf=confidence, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Display results
            if result.boxes is not None and len(result.boxes) > 0:
                print(f"✅ Detected {len(result.boxes)} waste item(s):")
                
                # Count by category
                category_counts = {}
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.model.names[class_id]
                    
                    if class_name not in category_counts:
                        category_counts[class_name] = []
                    category_counts[class_name].append(confidence)
                
                # Display summary
                print(f"\n📊 Detection Summary:")
                for category, confidences in category_counts.items():
                    avg_conf = sum(confidences) / len(confidences)
                    print(f"   • {category}: {len(confidences)} item(s) (avg confidence: {avg_conf:.3f})")
                
                # Display detailed results
                print(f"\n📋 Detailed Results:")
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.model.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    print(f"   {i+1}. {class_name}: {confidence:.3f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
                
                # Save result if requested
                if save_result:
                    result_image = result.plot()
                    output_path = f"result_{Path(image_path).stem}.jpg"
                    cv2.imwrite(output_path, result_image)
                    print(f"\n💾 Result saved to: {output_path}")
                
            else:
                print("ℹ️  No waste items detected in the image")
                
        else:
            print("ℹ️  No results returned from model")
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test waste classification model")
    parser.add_argument("--model", required=True, help="Path to trained model file")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save result image")
    
    args = parser.parse_args()
    
    test_waste_classification(
        model_path=args.model,
        image_path=args.image,
        confidence=args.confidence,
        save_result=not args.no_save
    )

if __name__ == "__main__":
    main()
