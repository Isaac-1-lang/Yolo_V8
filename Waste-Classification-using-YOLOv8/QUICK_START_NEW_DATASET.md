# Quick Start Guide - New Dataset Training

This guide will help you train a new YOLOv8 model on your updated waste classification dataset with 5 categories: Glass, Organic, Others, Packaged, and Plastic.

## 🚀 Quick Start (3 Easy Steps)

### Option 1: Automated Setup (Recommended)
```bash
# Double-click this file or run in command prompt:
run_training.bat
```

### Option 2: Manual Setup
```bash
# Step 1: Prepare dataset
python prepare_dataset.py

# Step 2: Train model
python train_new_model.py
```

### Option 3: Complete Pipeline
```bash
# Run everything at once
python setup_and_train.py
```

## 📊 Your Dataset

- **Glass**: 501 images
- **Organic**: 12,565 images (largest category)
- **Others**: 578 images
- **Packaged**: 105 images (smallest category)
- **Plastic**: 1,240 images

**Total**: ~15,000 images

## 🔧 What the Scripts Do

### 1. `prepare_dataset.py`
- Creates train/val/test splits (70%/20%/10%)
- Generates YOLO format annotations
- Creates `data.yaml` configuration file
- Organizes data in proper YOLO structure

### 2. `train_new_model.py`
- Trains YOLOv8s model on your dataset
- Uses GPU if available, falls back to CPU
- Saves best model as `best.pt`
- Generates training plots and metrics

### 3. `setup_and_train.py`
- Runs the complete pipeline
- Checks requirements
- Prepares dataset
- Trains model
- Updates Streamlit app

## 📁 Output Structure

After training, you'll have:

```
Waste-Classification-using-YOLOv8/
├── dataset/                          # Prepared dataset
│   ├── train/images/ & train/labels/
│   ├── val/images/ & val/labels/
│   └── test/images/ & test/labels/
├── runs/detect/waste_classification_new/  # Training results
│   ├── weights/
│   │   ├── best.pt                   # Best model
│   │   └── last.pt                   # Last checkpoint
│   ├── results.png                   # Training curves
│   ├── confusion_matrix.png          # Confusion matrix
│   └── F1_curve.png                  # F1 score curve
├── data.yaml                         # Dataset configuration
└── streamlit-detection-tracking - app/
    └── weights/
        └── best_new.pt               # Copy of trained model
```

## ⚙️ Training Parameters

- **Model**: YOLOv8s (good balance of speed/accuracy)
- **Epochs**: 50 (with early stopping)
- **Image Size**: 640x640
- **Batch Size**: 16 (GPU) / 8 (CPU)
- **Learning Rate**: 0.01
- **Augmentation**: Enabled (rotation, flip, HSV, etc.)

## 🎯 Expected Results

Based on your dataset size and categories:
- **Training Time**: 2-6 hours (depending on hardware)
- **Expected mAP50**: 0.6-0.8 (60-80% accuracy)
- **Best Classes**: Likely Organic (most data) and Glass
- **Challenging Classes**: Packaged (least data)

## 🔍 Monitoring Training

During training, you can monitor progress by checking:
- **Console output**: Real-time metrics
- **TensorBoard**: `tensorboard --logdir runs/detect/waste_classification_new`
- **Results plots**: `runs/detect/waste_classification_new/results.png`

## 🧪 Testing Your Model

After training, test your model:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/waste_classification_new/weights/best.pt')

# Test on an image
results = model.predict('path/to/test/image.jpg', conf=0.25, save=True)
```

## 🚨 Troubleshooting

### Common Issues:

1. **Out of Memory Error**:
   - Reduce batch size in `train_new_model.py`
   - Change `batch=16` to `batch=8` or `batch=4`

2. **Slow Training**:
   - Ensure GPU is being used
   - Check CUDA installation

3. **Poor Performance**:
   - Increase training epochs
   - Check data quality
   - Try different model size (yolov8m.pt, yolov8l.pt)

4. **Dataset Issues**:
   - Ensure all images are valid
   - Check file permissions
   - Verify class names match

## 📱 Using with Streamlit App

After training, update your Streamlit app:

1. The trained model is automatically copied to the app directory
2. Update `settings.py` to use the new model
3. Update class names in the app code
4. Run: `streamlit run app.py`

## 🎉 Success!

Once training completes, you'll have:
- ✅ A trained YOLOv8 model for 5 waste categories
- ✅ Training metrics and visualizations
- ✅ Model ready for deployment
- ✅ Updated Streamlit app

## 📞 Need Help?

If you encounter issues:
1. Check the console output for error messages
2. Verify your dataset structure
3. Ensure all dependencies are installed
4. Check GPU availability and memory

Happy training! 🚀

