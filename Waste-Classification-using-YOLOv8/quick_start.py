#!/usr/bin/env python3
"""
Quick Start Script for Waste Classification using YOLOv8
This script helps you get started quickly with the project.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_requirements():
    """Check if requirements.txt exists."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    print("✅ requirements.txt found")
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_model_files():
    """Check if model files exist."""
    weights_dir = Path("streamlit-detection-tracking - app/weights")
    if not weights_dir.exists():
        print("❌ Weights directory not found!")
        return False
    
    model_files = list(weights_dir.glob("*.pt"))
    if not model_files:
        print("❌ No .pt model files found in weights directory!")
        print("   You may need to train a model first or download pre-trained weights.")
        return False
    
    print(f"✅ Found model files: {[f.name for f in model_files]}")
    return True

def run_streamlit_app():
    """Run the Streamlit application."""
    app_dir = Path("streamlit-detection-tracking - app")
    if not app_dir.exists():
        print("❌ Streamlit app directory not found!")
        return False
    
    app_file = app_dir / "app.py"
    if not app_file.exists():
        print("❌ app.py not found!")
        return False
    
    print("\n🚀 Starting Streamlit application...")
    print("   The app will open in your web browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    
    try:
        os.chdir(app_dir)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        return True
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        return False

def main():
    """Main function to run the quick start process."""
    print("🤖 Waste Classification using YOLOv8 - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check requirements file
    if not check_requirements():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        print("\n⚠️  Model files not found. You have a few options:")
        print("   1. Train a new model using the Jupyter notebook")
        print("   2. Download pre-trained weights")
        print("   3. Continue anyway (app may not work properly)")
        
        choice = input("\nDo you want to continue anyway? (y/n): ").lower()
        if choice != 'y':
            sys.exit(1)
    
    # Run the application
    print("\n" + "=" * 50)
    run_streamlit_app()

if __name__ == "__main__":
    main()
