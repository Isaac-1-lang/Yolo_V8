# 🗑️ Waste Classification Category Management Guide

This guide explains how to manage and use your YOLOv8 waste classification model with the correct categories: **Glass**, **Organic**, **Others**, **Packaged**, and **Plastic**.

## 📋 Current Configuration

Your model is configured with the following waste categories:
- **0**: Glass
- **1**: Organic  
- **2**: Others
- **3**: Packaged
- **4**: Plastic

## 🛠️ Available Tools

### 1. Category Manager (`category_manager.py`)
A comprehensive utility for managing your waste categories.

**Basic Usage:**
```bash
python category_manager.py --data-yaml data.yaml --model "streamlit-detection-tracking - app/weights/best.pt"
```

**Features:**
- View current categories
- Verify model categories match expected categories
- Test inference on images
- Export categories to JSON

**Examples:**
```bash
# View current categories
python category_manager.py

# Test inference on an image
python category_manager.py --model "streamlit-detection-tracking - app/weights/best.pt" --test-image "dataset/train/images/0.jpg"

# Export categories to JSON
python category_manager.py --export categories.json
```

### 2. Model Verification (`verify_model_categories.py`)
Quick script to verify your model is using the correct categories.

```bash
python verify_model_categories.py
```

### 3. Test Inference (`test_inference.py`)
Simple script to test your model on any image.

```bash
python test_inference.py --model "streamlit-detection-tracking - app/weights/best.pt" --image "path/to/your/image.jpg"
```

## 🚀 Using the Streamlit App

### Starting the App
```bash
cd "streamlit-detection-tracking - app"
streamlit run app.py
```

### Features
- **Image Upload**: Upload images to classify waste
- **Webcam**: Real-time waste classification using your webcam
- **Confidence Control**: Adjust detection sensitivity
- **Detailed Results**: See category counts and confidence scores

### App Interface
1. **Main Page**: Shows the waste categories your model can detect
2. **Sidebar**: Configure model settings and select input source
3. **Results**: Detailed breakdown of detected waste items by category

## 🔧 Managing Categories

### Viewing Current Categories
```bash
python category_manager.py
```

### Updating Categories (if needed)
Edit `data.yaml`:
```yaml
names:
- Glass
- Organic
- Others
- Packaged
- Plastic
nc: 5
path: C:\Users\user\Documents\YEAR 2\Group projects\TERM I\WasteNet\Train\Waste-Classification-using-YOLOv8\dataset
test: test/images
train: train/images
val: val/images
```

### Retraining with New Categories
If you need to change categories:
1. Update `data.yaml` with new categories
2. Retrain the model:
   ```bash
   python train_new_model.py
   ```

## 📊 Understanding Results

### Detection Output
When you run inference, you'll see:
- **Category Summary**: Count of items per waste type
- **Confidence Scores**: How certain the model is about each detection
- **Bounding Boxes**: Location of detected waste items

### Example Output
```
📊 Detection Summary:
   • Plastic: 2 item(s) (avg confidence: 0.85)
   • Glass: 1 item(s) (avg confidence: 0.92)

📋 Detailed Results:
   1. Plastic: 0.856 at (100, 150, 200, 250)
   2. Glass: 0.923 at (300, 400, 350, 450)
   3. Plastic: 0.789 at (500, 100, 600, 200)
```

## 🎯 Best Practices

### For Better Results
1. **Good Lighting**: Ensure images are well-lit
2. **Clear View**: Avoid blurry or partially obscured waste items
3. **Single Items**: Try to have one waste item per image for best results
4. **Confidence Threshold**: Start with 0.25, adjust based on results

### Troubleshooting
- **No Detections**: Lower confidence threshold or check image quality
- **Wrong Categories**: Verify model was trained with correct data
- **Poor Performance**: Consider retraining with more data

## 📁 File Structure
```
Waste-Classification-using-YOLOv8/
├── data.yaml                          # Category configuration
├── category_manager.py                # Category management utility
├── verify_model_categories.py         # Model verification script
├── test_inference.py                  # Simple inference testing
├── train_new_model.py                 # Model training script
├── dataset/                           # Training data
│   ├── train/images/                  # Training images
│   ├── train/labels/                  # Training labels
│   ├── val/images/                    # Validation images
│   └── val/labels/                    # Validation labels
└── streamlit-detection-tracking - app/
    ├── app.py                         # Streamlit application
    ├── helper.py                      # Helper functions
    ├── settings.py                    # App configuration
    └── weights/
        └── best.pt                    # Trained model
```

## 🔄 Quick Start Checklist

1. ✅ **Verify Categories**: Run `python verify_model_categories.py`
2. ✅ **Test Model**: Run `python test_inference.py --model "streamlit-detection-tracking - app/weights/best.pt" --image "dataset/train/images/0.jpg"`
3. ✅ **Launch App**: Run `python quick_start.py` or manually start Streamlit
4. ✅ **Upload Image**: Test with your own waste images

## 🆘 Getting Help

If you encounter issues:
1. Check that your model file exists and is accessible
2. Verify your image paths are correct
3. Ensure all dependencies are installed (`pip install -r requirements.txt`)
4. Check the console output for specific error messages

## 📈 Next Steps

- **Improve Accuracy**: Add more training data for better performance
- **Custom Categories**: Modify categories if needed for your specific use case
- **Batch Processing**: Process multiple images at once
- **Integration**: Integrate the model into other applications

---

**Your waste classification model is ready to use with the correct categories: Glass, Organic, Others, Packaged, and Plastic!** 🎉
