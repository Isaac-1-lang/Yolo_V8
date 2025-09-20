# Waste Classification using YOLOv8 - Project Summary

## 🎯 Project Overview
This project implements a waste classification system using YOLOv8 deep learning model with a Streamlit web interface. It can classify waste into 6 categories: BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, and PLASTIC.

## 📁 Project Structure
```
Waste-Classification-using-YOLOv8/
├── README.md                           # Original project description
├── requirements.txt                    # Python dependencies
├── quick_start.py                     # Quick start script
├── SETUP_GUIDE.md                     # Installation guide
├── RUNNING_GUIDE.md                   # How to run the application
├── TRAINING_GUIDE.md                  # Model training and fine-tuning
├── PROJECT_SUMMARY.md                 # This file
├── waste_YOLOv8.ipynb                 # Training notebook
├── streamlit-detection-tracking - app/
│   ├── app.py                         # Main Streamlit application
│   ├── helper.py                      # Helper functions
│   ├── settings.py                    # Configuration settings
│   ├── images/                        # Sample images
│   └── weights/                       # Model weights
│       ├── best.pt                    # Trained model
│       ├── yolov8n.pt                 # Pre-trained YOLOv8 nano
│       └── yolov8 (1).pkl             # Pickled model (legacy)
└── results/                           # Training results and visualizations
```

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
# Option A: Use the quick start script
python quick_start.py

# Option B: Manual start
cd "streamlit-detection-tracking - app"
streamlit run app.py
```

### 3. Open Your Browser
Navigate to `http://localhost:8501` and start classifying waste!

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Installation Steps
1. **Clone/Download** the project
2. **Create virtual environment** (recommended)
3. **Install packages**: `pip install -r requirements.txt`
4. **Verify installation**: Check GPU availability and model files

See `SETUP_GUIDE.md` for detailed instructions.

## 🎮 How to Use

### Web Interface Features
- **Image Upload**: Upload waste images for classification
- **Real-time Webcam**: Live detection using your camera
- **Confidence Adjustment**: Tune detection sensitivity (25-100%)
- **Multiple Formats**: Supports JPG, PNG, BMP, WEBP

### Usage Steps
1. Select "Detection" mode
2. Adjust confidence threshold (start with 40-50%)
3. Choose input source (Image or Webcam)
4. Upload image or start webcam
5. Click "Detect Objects" or "Detect Trash"
6. View results with bounding boxes and labels

See `RUNNING_GUIDE.md` for detailed usage instructions.

## 🎓 Training & Fine-tuning

### Dataset
- **Source**: Roboflow (6000+ images)
- **Classes**: 6 waste categories
- **Format**: YOLO format with bounding box annotations
- **Split**: Training, Validation, Testing sets

### Training Options
1. **Jupyter Notebook**: Use `waste_YOLOv8.ipynb`
2. **Command Line**: `yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=30`
3. **Python Script**: Custom training scripts

### Model Sizes
- **yolov8n.pt**: Nano (fastest, least accurate)
- **yolov8s.pt**: Small (good balance) ⭐ Recommended
- **yolov8m.pt**: Medium (better accuracy)
- **yolov8l.pt**: Large (high accuracy)
- **yolov8x.pt**: Extra Large (best accuracy, slowest)

See `TRAINING_GUIDE.md` for comprehensive training instructions.

## 📊 Model Performance
Based on the training notebook results:
- **mAP50**: 0.503 (50.3% accuracy at IoU 0.5)
- **mAP50-95**: 0.341 (34.1% accuracy at IoU 0.5-0.95)
- **Best performing classes**: GLASS (76.7% mAP50), BIODEGRADABLE (61.3% mAP50)
- **Challenging classes**: PAPER (3.65% mAP50), PLASTIC (40.5% mAP50)

## 🛠️ Technical Details

### Technologies Used
- **YOLOv8**: Object detection model
- **Streamlit**: Web interface
- **OpenCV**: Image processing
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO implementation

### Key Features
- Real-time detection
- GPU acceleration support
- Multiple input sources
- Confidence threshold adjustment
- Batch processing capability
- Model export options

## 🔍 Troubleshooting

### Common Issues
1. **Model loading errors**: Check if `best.pt` exists in weights folder
2. **Webcam not working**: Try different camera indices (0, 1, 2)
3. **Slow performance**: Use GPU, reduce image size, increase confidence
4. **No detections**: Lower confidence threshold, check image quality

### Performance Tips
- Use GPU for faster inference
- Start with 40-50% confidence threshold
- Ensure good lighting for webcam detection
- Use appropriate image sizes

## 📈 Future Improvements

### Potential Enhancements
1. **Data Augmentation**: Improve model robustness
2. **Class Balancing**: Address imbalanced dataset
3. **Model Optimization**: Quantization for mobile deployment
4. **Additional Classes**: Expand waste categories
5. **Real-time Tracking**: Object tracking capabilities
6. **Batch Processing**: Process multiple images/videos
7. **API Integration**: REST API for external applications

### Advanced Features
- Multi-class detection
- Confidence calibration
- Model ensemble
- Active learning
- Edge deployment

## 📚 Documentation Files

- `SETUP_GUIDE.md`: Complete installation instructions
- `RUNNING_GUIDE.md`: Detailed usage guide
- `TRAINING_GUIDE.md`: Model training and fine-tuning
- `requirements.txt`: Python dependencies
- `quick_start.py`: Automated setup script

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the original repository for license details.

## 🙏 Acknowledgments

- **Ultralytics**: For the YOLOv8 implementation
- **Roboflow**: For the waste classification dataset
- **Streamlit**: For the web interface framework
- **OpenCV**: For computer vision capabilities

---

**Ready to start?** Run `python quick_start.py` and begin classifying waste!
