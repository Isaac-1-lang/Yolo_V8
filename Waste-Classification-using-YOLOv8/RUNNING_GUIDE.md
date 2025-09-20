# Waste Classification using YOLOv8 - Running Guide

## How to Run the Application

### 1. Start the Streamlit Web Application

```bash
# Navigate to the app directory
cd "streamlit-detection-tracking - app"

# Run the Streamlit app
streamlit run app.py
```

The application will start and automatically open in your web browser at `http://localhost:8501`

### 2. Using the Web Interface

#### Main Features:
- **Image Upload**: Upload waste images for classification
- **Webcam Detection**: Real-time detection using your camera
- **Confidence Adjustment**: Adjust detection sensitivity (25-100%)

#### Step-by-Step Usage:

1. **Select Detection Mode**:
   - Choose "Detection" from the sidebar

2. **Adjust Model Confidence**:
   - Use the slider to set confidence threshold (25-100%)
   - Higher values = more confident detections, fewer false positives
   - Lower values = more detections, may include false positives

3. **Choose Input Source**:
   - **Image**: Upload a waste image file
   - **Webcam**: Use your camera for real-time detection

#### For Image Detection:
1. Select "Image" from the source options
2. Click "Choose an image..." and select a waste image
3. Click "Detect Objects" button
4. View the results with bounding boxes and labels

#### For Webcam Detection:
1. Select "Webcam" from the source options
2. Click "Detect Trash" button
3. Allow camera access when prompted
4. View real-time waste detection

### 3. Understanding the Results

The application will display:
- **Bounding boxes** around detected waste objects
- **Class labels** (BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC)
- **Confidence scores** for each detection
- **Detection details** in the expandable results section

### 4. Supported File Formats

**Images**: JPG, JPEG, PNG, BMP, WEBP

### 5. Performance Tips

- **GPU Usage**: The app automatically uses GPU if available
- **Confidence Threshold**: Start with 40-50% for balanced results
- **Image Size**: Larger images may take longer to process
- **Webcam**: Ensure good lighting for better detection

### 6. Troubleshooting

#### Common Issues:

1. **"Unable to load model" Error**:
   - Check if `best.pt` exists in the `weights/` folder
   - Verify the model path in `settings.py`

2. **Webcam Not Working**:
   - Ensure camera permissions are granted
   - Try changing `WEBCAM_PATH` in `settings.py` (try 0, 1, or 2)

3. **Slow Performance**:
   - Reduce image size before upload
   - Increase confidence threshold
   - Check if GPU is being used

4. **No Detections**:
   - Lower the confidence threshold
   - Ensure the image contains recognizable waste objects
   - Check if the model is properly loaded

### 7. Advanced Configuration

#### Modify Settings in `settings.py`:
```python
# Change default confidence
confidence = 0.4  # 40% confidence threshold

# Change webcam source
WEBCAM_PATH = 0  # Try 0, 1, or 2 for different cameras

# Add more image formats
# In app.py, modify the file_uploader type parameter
```

#### Custom Model:
1. Place your trained model in the `weights/` folder
2. Update `DETECTION_MODEL` path in `settings.py`
3. Restart the application

### 8. Command Line Usage

You can also use the model directly from command line:

```bash
# Detect objects in an image
yolo task=detect mode=predict model=weights/best.pt source=path/to/image.jpg conf=0.4

# Detect objects in a video
yolo task=detect mode=predict model=weights/best.pt source=path/to/video.mp4 conf=0.4

# Detect objects using webcam
yolo task=detect mode=predict model=weights/best.pt source=0 conf=0.4
```

### 9. Batch Processing

For processing multiple images:

```python
from ultralytics import YOLO
import os

# Load model
model = YOLO('weights/best.pt')

# Process all images in a folder
input_folder = 'path/to/images'
output_folder = 'path/to/results'

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        results = model.predict(image_path, conf=0.4, save=True, project=output_folder)
```

## Next Steps
- See `TRAINING_GUIDE.md` for fine-tuning the model
- Check `SETUP_GUIDE.md` for installation instructions
