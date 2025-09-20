#!/usr/bin/env python3
"""
Dataset Preparation Script for Waste Classification
This script prepares the dataset for YOLO training by:
1. Creating train/val/test splits
2. Generating YOLO format annotations (full image bounding boxes)
3. Creating the data.yaml configuration file
"""

import os
import shutil
import random
from pathlib import Path
import yaml

def create_yolo_annotations(image_path, class_id, output_path):
    """
    Create YOLO format annotation file for an image.
    Since we don't have bounding box annotations, we'll create full-image annotations.
    """
    # YOLO format: class_id center_x center_y width height (all normalized 0-1)
    # For full image annotation: center_x=0.5, center_y=0.5, width=1.0, height=1.0
    annotation = f"{class_id} 0.5 0.5 1.0 1.0\n"
    
    with open(output_path, 'w') as f:
        f.write(annotation)

def prepare_dataset():
    """Prepare the dataset for YOLO training."""
    
    # Define paths
    data_root = Path("../Data")
    output_root = Path("dataset")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (output_root / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Define class mapping
    classes = ['Glass', 'Organic', 'Others', 'Packaged', 'Plastic']
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    print("Preparing dataset...")
    print(f"Classes: {classes}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Process each class
    for class_name in classes:
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping...")
            continue
            
        class_id = class_to_id[class_name]
        print(f"\nProcessing {class_name} (ID: {class_id})...")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(class_dir.glob(ext))
        
        print(f"Found {len(image_files)} images")
        
        # Shuffle and split
        random.shuffle(image_files)
        
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Copy files and create annotations
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            print(f"   Processing {split} split...")
            for i, img_path in enumerate(files):
                if i % 100 == 0:  # Progress update every 100 files
                    print(f"     Progress: {i}/{len(files)} files")
                
                # Copy image
                img_dest = output_root / split / 'images' / img_path.name
                try:
                    shutil.copy2(img_path, img_dest)
                except Exception as e:
                    print(f"     Warning: Failed to copy {img_path.name}: {e}")
                    continue
                
                # Create annotation
                label_dest = output_root / split / 'labels' / (img_path.stem + '.txt')
                create_yolo_annotations(img_path, class_id, label_dest)
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_root.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\nDataset preparation complete!")
    print(f"Dataset saved to: {output_root.absolute()}")
    print(f"Configuration saved to: data.yaml")
    
    # Print summary
    print("\nDataset Summary:")
    for split in ['train', 'val', 'test']:
        img_count = len(list((output_root / split / 'images').glob('*')))
        print(f"{split.capitalize()}: {img_count} images")

if __name__ == "__main__":
    # Set random seed for reproducible splits
    random.seed(42)
    prepare_dataset()
