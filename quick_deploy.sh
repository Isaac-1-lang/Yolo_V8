#!/bin/bash

# Quick Deployment Script for Waste Classification
# This script provides a simple way to deploy the application

echo "🚀 Quick Deploy - Waste Classification Model"
echo "==========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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

# Get user input
echo "Please provide the following information:"
echo ""

read -p "Enter your domain name or IP address: " DOMAIN
read -p "Enter the full path to your project directory (e.g., /opt/waste-classification): " PROJECT_DIR

# Validate inputs
if [ -z "$DOMAIN" ]; then
    print_error "Domain name is required!"
    exit 1
fi

if [ -z "$PROJECT_DIR" ]; then
    print_error "Project directory path is required!"
    exit 1
fi

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    print_error "Project directory $PROJECT_DIR not found!"
    print_warning "Please ensure your project is located at $PROJECT_DIR"
    exit 1
fi

print_status "Domain: $DOMAIN"
print_status "Project Directory: $PROJECT_DIR"

# Update the deployment script with user inputs
print_status "Configuring deployment script..."
sed -i "s|your-domain.com|$DOMAIN|g" deploy.sh
sed -i "s|/opt/waste-classification|$PROJECT_DIR|g" deploy.sh

# Update the service file
print_status "Configuring systemd service..."
sed -i "s|/opt/waste-classification|$PROJECT_DIR|g" waste-classification.service

# Update the nginx configuration
print_status "Configuring Nginx..."
sed -i "s|your-domain.com|$DOMAIN|g" nginx.conf

# Copy configuration files to system locations
print_status "Installing configuration files..."

# Copy nginx configuration
sudo cp nginx.conf /etc/nginx/sites-available/waste-classification

# Copy systemd service
sudo cp waste-classification.service /etc/systemd/system/waste-classification.service

# Copy streamlit configuration
sudo mkdir -p /opt/waste-classification/.streamlit
sudo cp streamlit_config.toml /opt/waste-classification/.streamlit/config.toml

print_status "Configuration files installed!"

# Make deployment scripts executable
chmod +x deploy.sh
chmod +x update.sh

print_status "Deployment scripts are ready!"

echo ""
echo "🎯 Next Steps:"
echo "=============="
echo "1. Run the deployment script:"
echo "   ./deploy.sh"
echo ""
echo "2. If deployment is successful, your app will be available at:"
echo "   http://$DOMAIN"
echo ""
echo "3. To set up SSL (recommended):"
echo "   sudo certbot --nginx -d $DOMAIN"
echo ""
echo "4. To update the application later:"
echo "   ./update.sh"
echo ""

print_warning "Make sure to:"
echo "  - Update DNS records to point to this server"
echo "  - Configure firewall to allow HTTP/HTTPS traffic"
echo "  - Test the application after deployment"

print_status "Quick setup completed! Run ./deploy.sh to start deployment."
