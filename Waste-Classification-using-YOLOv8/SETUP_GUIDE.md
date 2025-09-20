# Waste Classification using YOLOv8 - Setup Guide

## Project Overview
This project uses YOLOv8 for waste classification with 6 categories: BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, and PLASTIC. It includes both training capabilities and a Streamlit web interface for real-time detection.

## Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- 10GB free disk space

## Installation Steps

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd Waste-Classification-using-YOLOv8

# Or download and extract the project files
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv waste_classification_env

# Activate virtual environment
# On Windows:
waste_classification_env\Scripts\activate
# On macOS/Linux:
source waste_classification_env/bin/activate
```

### 3. Install Required Packages
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually:
pip install ultralytics streamlit opencv-python pillow numpy pandas matplotlib seaborn
```

### 4. Verify Installation
```bash
# Check if YOLOv8 is properly installed
python -c "from ultralytics import YOLO; print('YOLOv8 installed successfully')"

# Check if Streamlit is working
streamlit --version
```

## Project Structure
```
Waste-Classification-using-YOLOv8/
├── README.md
├── requirements.txt
├── SETUP_GUIDE.md
├── waste_YOLOv8.ipynb          # Training notebook
├── streamlit-detection-tracking - app/
│   ├── app.py                   # Main Streamlit application
│   ├── helper.py                # Helper functions
│   ├── settings.py              # Configuration settings
│   ├── images/                  # Sample images
│   └── weights/                 # Model weights
│       ├── best.pt              # Trained model
│       ├── yolov8n.pt           # Pre-trained YOLOv8 nano
│       └── yolov8 (1).pkl       # Pickled model (legacy)
└── results/                     # Training results and visualizations
```

## Quick Start

### Option 1: Use Pre-trained Model
1. The project includes a pre-trained model (`best.pt`) in the weights folder
2. Run the Streamlit app directly:
```bash
cd "streamlit-detection-tracking - app"
streamlit run app.py
```

### Option 2: Train Your Own Model
1. Follow the training guide in `TRAINING_GUIDE.md`
2. Place your trained model in the `weights/` folder
3. Update the model path in `settings.py` if needed

## Troubleshooting

### Common Issues:

1. **CUDA/GPU Issues:**
   - Install CUDA toolkit if you have an NVIDIA GPU
   - Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Path Issues:**
   - Make sure you're running commands from the correct directory
   - Check that all file paths in `settings.py` are correct

3. **Import Errors:**
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Model Loading Issues:**
   - Verify the model file exists in the weights folder
   - Check file permissions

## Next Steps
- See `RUNNING_GUIDE.md` for how to use the application
- See `TRAINING_GUIDE.md` for model fine-tuning instructions
