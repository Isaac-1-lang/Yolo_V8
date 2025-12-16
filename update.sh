#!/bin/bash

# Waste Classification Update Script
# This script updates the deployed waste classification model

echo "🔄 Updating Waste Classification Application..."
echo "=============================================="

# Configuration
PROJECT_DIR="/opt/waste-classification"
SERVICE_NAME="waste-classification"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root. Run as a regular user with sudo privileges."
    exit 1
fi

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory $PROJECT_DIR not found!"
    print_warning "Please ensure your project is located at $PROJECT_DIR"
    exit 1
fi

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1

print_status "Navigated to project directory: $PROJECT_DIR"

# Create backup
print_status "Creating backup..."
BACKUP_DIR="/opt/waste-classification-backup-$(date +%Y%m%d-%H%M%S)"
sudo cp -r "$PROJECT_DIR" "$BACKUP_DIR"
print_status "Backup created at: $BACKUP_DIR"

# If using Git, pull latest changes
if [ -d ".git" ]; then
    print_status "Pulling latest changes from Git..."
    git pull origin main
else
    print_warning "Not a Git repository. Please manually update your files."
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source waste_classification_env/bin/activate

# Update pip
print_status "Updating pip..."
pip install --upgrade pip

# Update dependencies
print_status "Updating dependencies..."
pip install --upgrade -r requirements.txt

# Test the application
print_status "Testing application..."
if python verify_model_categories.py; then
    print_status "Model verification successful!"
else
    print_warning "Model verification failed, but continuing with update..."
fi

# Restart service
print_status "Restarting service..."
sudo systemctl restart $SERVICE_NAME

# Wait a moment for service to start
sleep 5

# Check service status
print_status "Checking service status..."
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    print_status "Service is running!"
else
    print_error "Service failed to start!"
    print_warning "Rolling back to previous version..."
    sudo rm -rf "$PROJECT_DIR"
    sudo mv "$BACKUP_DIR" "$PROJECT_DIR"
    sudo systemctl restart $SERVICE_NAME
    print_error "Update failed! Rolled back to previous version."
    exit 1
fi

# Check if application is accessible
print_status "Testing application accessibility..."
if curl -s http://localhost:8501 > /dev/null; then
    print_status "Application is accessible!"
else
    print_warning "Application might not be accessible yet. Check logs for details."
fi

echo ""
echo "🎉 Update completed successfully!"
echo "================================"
echo "📊 Service status:"
sudo systemctl status $SERVICE_NAME --no-pager -l
echo ""
echo "🔧 Useful commands:"
echo "   View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "   Restart service: sudo systemctl restart $SERVICE_NAME"
echo "   Check Nginx status: sudo systemctl status nginx"
echo ""
print_status "Update completed! 🚀"
