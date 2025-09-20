# 🗑️ Waste Classification Model - Training & Deployment

A comprehensive YOLOv8-based waste classification system that automatically identifies and categorizes waste materials into 5 main categories: **Glass**, **Organic**, **Others**, **Packaged**, and **Plastic**.

## 📋 Table of Contents

- [Overview](#overview)
- [Model Categories](#model-categories)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Web Application](#web-application)
- [Category Management](#category-management)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a state-of-the-art waste classification system using YOLOv8 (You Only Look Once version 8) for real-time object detection and classification. The system is designed to help improve waste management practices by automatically sorting waste materials.

### Key Features

- ✅ **Real-time Detection**: Fast and accurate waste classification
- ✅ **5 Waste Categories**: Glass, Organic, Others, Packaged, Plastic
- ✅ **Web Interface**: User-friendly Streamlit application
- ✅ **Category Mapping**: Intelligent mapping from model categories to desired categories
- ✅ **Multiple Input Sources**: Images, webcam, and video files
- ✅ **Confidence Control**: Adjustable detection sensitivity
- ✅ **Detailed Results**: Category counts and confidence scores

## 🏷️ Model Categories

The model classifies waste into the following categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **Glass** | Glass containers and bottles | Wine bottles, jars, glass containers |
| **Organic** | Biodegradable waste | Food scraps, leaves, organic matter |
| **Others** | Miscellaneous waste | Metal, ceramics, non-recyclable items |
| **Packaged** | Packaging materials | Cardboard, paper, packaging |
| **Plastic** | Plastic containers and items | Plastic bottles, bags, containers |

### Category Mapping

The model uses intelligent category mapping to convert from the original training categories to your desired categories:

```
BIODEGRADABLE → Organic
CARDBOARD     → Packaged
GLASS         → Glass
METAL         → Others
PAPER         → Packaged
PLASTIC       → Plastic
```

## 📁 Project Structure

```
Train/
├── Waste-Classification-using-YOLOv8/
│   ├── dataset/                          # Training and validation data
│   │   ├── train/images/                 # Training images
│   │   ├── train/labels/                 # Training labels
│   │   ├── val/images/                   # Validation images
│   │   └── val/labels/                   # Validation labels
│   ├── streamlit-detection-tracking - app/  # Web application
│   │   ├── app.py                        # Main Streamlit app
│   │   ├── helper.py                     # Helper functions
│   │   ├── settings.py                   # App configuration
│   │   ├── category_mapper.py            # Category mapping utility
│   │   └── weights/
│   │       └── best.pt                   # Trained model weights
│   ├── data.yaml                         # Dataset configuration
│   ├── category_mapper.py                # Category mapping utility
│   ├── verify_model_categories.py        # Model verification script
│   ├── test_inference.py                 # Inference testing script
│   ├── train_new_model.py                # Model training script
│   ├── quick_start.py                    # Quick start script
│   └── requirements.txt                  # Python dependencies
└── README.md                             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd WasteNet/Train
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Waste-Classification-using-YOLOv8
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the model**:
   ```bash
   python verify_model_categories.py
   ```

5. **Launch the web application**:
   ```bash
   python quick_start.py
   ```

### Quick Test

Test the model on a sample image:
```bash
python test_inference.py --model "streamlit-detection-tracking - app/weights/best.pt" --image "dataset/train/images/0.jpg"
```

## 🏋️ Training

### Dataset Preparation

The dataset should be organized in YOLO format:
- Images in `dataset/train/images/` and `dataset/val/images/`
- Labels in `dataset/train/labels/` and `dataset/val/labels/`
- Each label file should contain bounding box annotations in YOLO format

### Training the Model

1. **Prepare your dataset** (if using custom data):
   ```bash
   python prepare_dataset.py
   ```

2. **Start training**:
   ```bash
   python train_new_model.py
   ```

3. **Monitor training progress**:
   - Check the `runs/detect/` directory for training logs
   - View training plots and metrics
   - The best model will be saved as `runs/detect/waste_classification_new/weights/best.pt`

### Training Parameters

Key training parameters (configurable in `train_new_model.py`):
- **Epochs**: 50 (default)
- **Batch Size**: 16 (GPU) / 8 (CPU)
- **Image Size**: 640x640
- **Learning Rate**: 0.01
- **Device**: CUDA (if available) / CPU

## 🔍 Inference

### Command Line Inference

```bash
# Basic inference
python test_inference.py --model "path/to/model.pt" --image "path/to/image.jpg"

# With custom confidence threshold
python test_inference.py --model "path/to/model.pt" --image "path/to/image.jpg" --confidence 0.3

# Without saving result image
python test_inference.py --model "path/to/model.pt" --image "path/to/image.jpg" --no-save
```

### Python API

```python
from ultralytics import YOLO
from category_mapper import WasteCategoryMapper

# Load model
model = YOLO("path/to/model.pt")

# Load category mapper
mapper = WasteCategoryMapper()

# Run inference
results = model.predict("image.jpg", conf=0.25)

# Map to desired categories
mapped_results = mapper.map_prediction(model, results[0])

# Process results
for item in mapped_results:
    print(f"Category: {item['mapped_class']}, Confidence: {item['confidence']:.3f}")
```

## 🌐 Web Application

### Launching the App

```bash
cd "streamlit-detection-tracking - app"
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Features

- **Image Upload**: Upload images for waste classification
- **Webcam Support**: Real-time classification using your webcam
- **Confidence Control**: Adjust detection sensitivity
- **Category Mapping**: See how model categories map to your desired categories
- **Detailed Results**: View category counts and confidence scores

### App Interface

1. **Main Page**: Shows supported waste categories
2. **Sidebar**: Model configuration and input source selection
3. **Results Panel**: Detailed detection results with category mapping
4. **Mapping Info**: Explanation of category conversions

## 🔧 Category Management

### Viewing Categories

```bash
python category_manager.py
```

### Testing Category Mapping

```bash
python category_mapper.py
```

### Updating Categories

If you need to modify categories:

1. **Edit `data.yaml`**:
   ```yaml
   names:
   - Glass
   - Organic
   - Others
   - Packaged
   - Plastic
   nc: 5
   ```

2. **Retrain the model** (if needed):
   ```bash
   python train_new_model.py
   ```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**:
   - Check that `best.pt` exists in the weights directory
   - Verify the file path in `settings.py`

2. **CUDA out of memory**:
   - Reduce batch size in training parameters
   - Use CPU instead of GPU

3. **Category mismatch**:
   - Run `python verify_model_categories.py` to check categories
   - Use the category mapping solution

4. **Poor detection accuracy**:
   - Adjust confidence threshold
   - Ensure good image quality
   - Consider retraining with more data

### Getting Help

- Check the console output for error messages
- Verify all dependencies are installed
- Ensure image paths are correct
- Review the category mapping if results seem incorrect

## 📊 Performance

### Model Performance

- **mAP50**: ~0.85 (on validation set)
- **Inference Speed**: ~50ms per image (GPU)
- **Model Size**: ~50MB
- **Input Size**: 640x640 pixels

### System Requirements

- **Minimum**: 4GB RAM, CPU-only
- **Recommended**: 8GB RAM, CUDA-compatible GPU
- **Optimal**: 16GB RAM, RTX 3060 or better

## 🔄 Future Improvements

- [ ] Support for more waste categories
- [ ] Mobile app development
- [ ] Real-time video processing
- [ ] Integration with IoT devices
- [ ] Multi-language support
- [ ] Batch processing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for image processing
- The waste classification research community

---

**Ready to classify waste? Start with the Quick Start guide above!** 🚀
