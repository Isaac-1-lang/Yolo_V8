# Waste Classification using YOLOv8 - Training & Fine-tuning Guide

## Overview
This guide covers how to train and fine-tune the YOLOv8 model for waste classification. The project includes a Jupyter notebook (`waste_YOLOv8.ipynb`) with the complete training pipeline.

## Dataset Information
- **Source**: Roboflow (https://roboflow.com/)
- **Classes**: 6 waste categories (BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC)
- **Size**: 6000+ images
- **Split**: Training, Validation, Testing sets
- **Format**: YOLO format with bounding box annotations

## Training Options

### Option 1: Use the Jupyter Notebook (Recommended)
1. Open `waste_YOLOv8.ipynb` in Jupyter Lab/Notebook
2. Follow the cells step by step
3. Modify parameters as needed

### Option 2: Command Line Training
```bash
# Basic training command
yolo task=detect mode=train model=yolov8s.pt data=path/to/data.yaml epochs=30 imgsz=640

# Advanced training with custom parameters
yolo task=detect mode=train model=yolov8s.pt data=path/to/data.yaml epochs=50 imgsz=800 batch=16 lr0=0.01
```

### Option 3: Python Script Training
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # or yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(
    data='path/to/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    lr0=0.01,
    weight_decay=0.0005,
    momentum=0.937
)
```

## Step-by-Step Training Process

### 1. Prepare Your Dataset

#### Using Roboflow (Recommended):
```python
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("your_workspace").project("waste-classification-new")
dataset = project.version(1).download("yolov8")
```

#### Manual Dataset Preparation:
1. Organize images in folders: `train/`, `val/`, `test/`
2. Create corresponding label folders with `.txt` files
3. Create `data.yaml` file:

```yaml
# data.yaml
path: /path/to/dataset  # dataset root dir
train: train/images     # train images (relative to 'path')
val: val/images         # val images (relative to 'path')
test: test/images       # test images (relative to 'path')

# Classes
nc: 6  # number of classes
names: ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
```

### 2. Choose Model Size
- **yolov8n.pt**: Nano (fastest, least accurate)
- **yolov8s.pt**: Small (good balance)
- **yolov8m.pt**: Medium (better accuracy)
- **yolov8l.pt**: Large (high accuracy)
- **yolov8x.pt**: Extra Large (best accuracy, slowest)

### 3. Training Parameters

#### Essential Parameters:
```python
# Basic training
model.train(
    data='data.yaml',           # Path to data file
    epochs=30,                  # Number of training epochs
    imgsz=640,                  # Input image size
    batch=16,                   # Batch size
    device='cuda',              # Device (cuda, cpu, or specific GPU)
    workers=8,                  # Number of worker threads
    project='runs/detect',      # Project directory
    name='waste_classification' # Experiment name
)
```

#### Advanced Parameters:
```python
# Advanced training with hyperparameter tuning
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=800,
    batch=16,
    lr0=0.01,                   # Initial learning rate
    lrf=0.01,                   # Final learning rate
    momentum=0.937,             # SGD momentum
    weight_decay=0.0005,        # Optimizer weight decay
    warmup_epochs=3.0,          # Warmup epochs
    warmup_momentum=0.8,        # Warmup momentum
    warmup_bias_lr=0.1,         # Warmup bias learning rate
    box=7.5,                    # Box loss gain
    cls=0.5,                    # Class loss gain
    dfl=1.5,                    # DFL loss gain
    pose=12.0,                  # Pose loss gain
    kobj=1.0,                   # Keypoint object loss gain
    label_smoothing=0.0,        # Label smoothing
    nbs=64,                     # Nominal batch size
    hsv_h=0.015,                # Image HSV-Hue augmentation
    hsv_s=0.7,                  # Image HSV-Saturation augmentation
    hsv_v=0.4,                  # Image HSV-Value augmentation
    degrees=0.0,                # Image rotation degrees
    translate=0.1,              # Image translation
    scale=0.5,                  # Image scale
    shear=0.0,                  # Image shear
    perspective=0.0,            # Image perspective
    flipud=0.0,                 # Image flip up-down
    fliplr=0.5,                 # Image flip left-right
    mosaic=1.0,                 # Image mosaic
    mixup=0.0,                  # Image mixup
    copy_paste=0.0,             # Segment copy-paste
    auto_augment='randaugment', # Auto augmentation policy
    erasing=0.4,                # Random erasing probability
    crop_fraction=1.0           # Image crop fraction
)
```

### 4. Monitor Training

#### Training Metrics:
- **Loss**: Should decrease over time
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision at IoU 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

#### Visualization:
Training generates several plots in `runs/detect/train/`:
- `results.png`: Training curves
- `confusion_matrix.png`: Confusion matrix
- `F1_curve.png`: F1 score curve
- `PR_curve.png`: Precision-Recall curve
- `P_curve.png`: Precision curve
- `R_curve.png`: Recall curve

### 5. Validation and Testing

#### Validate the Model:
```bash
# Validate on validation set
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

# Validate on test set
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml split=test
```

#### Test on New Images:
```bash
# Test on single image
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg conf=0.25

# Test on folder of images
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=path/to/images/ conf=0.25
```

### 6. Fine-tuning Strategies

#### Transfer Learning:
```python
# Load pre-trained model
model = YOLO('yolov8s.pt')

# Fine-tune with lower learning rate
model.train(
    data='your_data.yaml',
    epochs=20,
    lr0=0.001,  # Lower learning rate
    freeze=10   # Freeze first 10 layers
)
```

#### Resume Training:
```python
# Resume from last checkpoint
model.train(resume=True)
```

#### Custom Dataset Augmentation:
```python
# Increase data augmentation for small datasets
model.train(
    data='data.yaml',
    epochs=100,
    hsv_h=0.1,      # More color variation
    hsv_s=0.9,      # More saturation variation
    hsv_v=0.5,      # More brightness variation
    degrees=15.0,   # More rotation
    translate=0.2,  # More translation
    scale=0.8,      # More scaling
    shear=2.0,      # More shearing
    perspective=0.0001,  # More perspective
    mosaic=1.0,     # Enable mosaic
    mixup=0.1,      # Enable mixup
    copy_paste=0.1  # Enable copy-paste
)
```

### 7. Model Optimization

#### Export for Production:
```python
# Export to different formats
model.export(format='onnx')      # ONNX format
model.export(format='torchscript')  # TorchScript format
model.export(format='tflite')    # TensorFlow Lite format
model.export(format='coreml')    # CoreML format
```

#### Quantization:
```python
# Quantize model for faster inference
model.export(format='onnx', int8=True)
```

### 8. Hyperparameter Tuning

#### Using Optuna (Advanced):
```python
import optuna
from ultralytics import YOLO

def objective(trial):
    model = YOLO('yolov8s.pt')
    results = model.train(
        data='data.yaml',
        epochs=10,
        lr0=trial.suggest_float('lr0', 0.001, 0.1),
        weight_decay=trial.suggest_float('weight_decay', 0.0001, 0.01),
        momentum=trial.suggest_float('momentum', 0.8, 0.95),
        batch=trial.suggest_categorical('batch', [8, 16, 32])
    )
    return results.results_dict['metrics/mAP50(B)']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### 9. Troubleshooting Training Issues

#### Common Problems:

1. **Out of Memory Error**:
   - Reduce batch size: `batch=8` or `batch=4`
   - Reduce image size: `imgsz=416` instead of `imgsz=640`
   - Use gradient accumulation

2. **Poor Performance**:
   - Increase training epochs
   - Adjust learning rate
   - Check data quality and annotations
   - Try different model sizes

3. **Overfitting**:
   - Increase data augmentation
   - Add regularization (weight_decay, dropout)
   - Reduce model complexity
   - Get more training data

4. **Slow Training**:
   - Use GPU acceleration
   - Increase batch size (if memory allows)
   - Reduce image size
   - Use mixed precision training

### 10. Best Practices

1. **Data Quality**:
   - Ensure high-quality annotations
   - Balance class distribution
   - Include diverse lighting conditions
   - Add challenging examples

2. **Training Strategy**:
   - Start with pre-trained weights
   - Use appropriate learning rate
   - Monitor validation metrics
   - Save best model checkpoints

3. **Evaluation**:
   - Test on unseen data
   - Use multiple metrics
   - Compare with baseline models
   - Consider inference speed

## Next Steps
- See `RUNNING_GUIDE.md` for using the trained model
- Check `SETUP_GUIDE.md` for installation instructions
