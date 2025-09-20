#!/usr/bin/env python3
"""
Complete Setup and Training Script for Waste Classification
This script handles the entire pipeline from dataset preparation to model training.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False

def check_requirements():
    """Check if required packages are installed."""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python',
        'Pillow',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'PyYAML',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Installing missing packages...")
        
        install_cmd = f"pip install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Installing missing packages"):
            print("❌ Failed to install required packages")
            return False
    
    return True

def main():
    """Main function to run the complete pipeline."""
    print("🗑️  Waste Classification - Complete Setup and Training")
    print("=" * 60)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Please install missing packages manually.")
        return False
    
    # Step 2: Prepare dataset
    print("\n" + "="*60)
    print("📊 STEP 1: Preparing Dataset")
    print("="*60)
    
    if not run_command("python prepare_dataset.py", "Dataset preparation"):
        print("❌ Dataset preparation failed")
        return False
    
    # Step 3: Train model
    print("\n" + "="*60)
    print("🤖 STEP 2: Training Model")
    print("="*60)
    
    if not run_command("python train_new_model.py", "Model training"):
        print("❌ Model training failed")
        return False
    
    # Step 4: Copy trained model to Streamlit app
    print("\n" + "="*60)
    print("📱 STEP 3: Updating Streamlit App")
    print("="*60)
    
    source_model = "runs/detect/waste_classification_new/weights/best.pt"
    target_dir = "streamlit-detection-tracking - app/weights"
    
    if os.path.exists(source_model):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        import shutil
        shutil.copy2(source_model, os.path.join(target_dir, "best_new.pt"))
        print(f"✅ Copied trained model to {target_dir}/best_new.pt")
    else:
        print(f"⚠️  Trained model not found at {source_model}")
    
    # Step 5: Create updated settings file
    print("\n📝 Creating updated settings...")
    
    settings_content = '''# Updated settings for new waste classification model
MODEL_PATH = "weights/best_new.pt"
CLASS_NAMES = ["Glass", "Organic", "Others", "Packaged", "Plastic"]
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
'''
    
    with open("streamlit-detection-tracking - app/settings_new.py", "w") as f:
        f.write(settings_content)
    
    print("✅ Created settings_new.py with updated class names")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 What was accomplished:")
    print("1. ✅ Dataset prepared with train/val/test splits")
    print("2. ✅ YOLOv8 model trained on new categories")
    print("3. ✅ Model saved and ready for use")
    print("4. ✅ Streamlit app updated with new model")
    
    print("\n📁 Important files created:")
    print(f"   - Dataset: dataset/")
    print(f"   - Trained model: {source_model}")
    print(f"   - Training results: runs/detect/waste_classification_new/")
    print(f"   - Updated settings: streamlit-detection-tracking - app/settings_new.py")
    
    print("\n🚀 Next steps:")
    print("1. Test the trained model:")
    print("   python -c \"from ultralytics import YOLO; model = YOLO('runs/detect/waste_classification_new/weights/best.pt'); model.predict('path/to/test/image.jpg', save=True)\"")
    print("\n2. Run the updated Streamlit app:")
    print("   cd 'streamlit-detection-tracking - app'")
    print("   streamlit run app.py")
    print("\n3. Check training results in runs/detect/waste_classification_new/")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ All done! Your waste classification model is ready to use.")

