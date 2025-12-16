#!/bin/bash

# Waste Classification Deployment Script
# This script deploys the waste classification model to a server

echo "🚀 Starting Waste Classification Deployment..."
echo "=============================================="

# Configuration
PROJECT_DIR="/opt/waste-classification"
SERVICE_NAME="waste-classification"
NGINX_SITE="waste-classification"
DOMAIN="your-domain.com"  # Change this to your actual domain

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

# Create virtual environment if it doesn't exist
if [ ! -d "waste_classification_env" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv waste_classification_env
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source waste_classification_env/bin/activate

# Update pip
print_status "Updating pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Test the application
print_status "Testing application..."
if python verify_model_categories.py; then
    print_status "Model verification successful!"
else
    print_warning "Model verification failed, but continuing with deployment..."
fi

# Create systemd service file
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Waste Classification Streamlit App
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/waste_classification_env/bin
ExecStart=$PROJECT_DIR/waste_classification_env/bin/streamlit run streamlit-detection-tracking\ -\ app/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create Nginx configuration
print_status "Creating Nginx configuration..."
sudo tee /etc/nginx/sites-available/$NGINX_SITE > /dev/null <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
EOF

# Enable Nginx site
print_status "Enabling Nginx site..."
sudo ln -sf /etc/nginx/sites-available/$NGINX_SITE /etc/nginx/sites-enabled/

# Test Nginx configuration
print_status "Testing Nginx configuration..."
if sudo nginx -t; then
    print_status "Nginx configuration is valid!"
else
    print_error "Nginx configuration test failed!"
    exit 1
fi

# Set proper permissions
print_status "Setting permissions..."
sudo chown -R www-data:www-data "$PROJECT_DIR"
sudo chmod -R 755 "$PROJECT_DIR"

# Reload systemd
print_status "Reloading systemd..."
sudo systemctl daemon-reload

# Enable and start service
print_status "Enabling service..."
sudo systemctl enable $SERVICE_NAME

print_status "Starting service..."
sudo systemctl start $SERVICE_NAME

# Restart Nginx
print_status "Restarting Nginx..."
sudo systemctl restart nginx

# Wait a moment for service to start
sleep 5

# Check service status
print_status "Checking service status..."
if sudo systemctl is-active --quiet $SERVICE_NAME; then
    print_status "Service is running!"
else
    print_error "Service failed to start!"
    print_warning "Check logs with: sudo journalctl -u $SERVICE_NAME -f"
    exit 1
fi

# Check if application is accessible
print_status "Testing application accessibility..."
if curl -s http://localhost:8501 > /dev/null; then
    print_status "Application is accessible on localhost:8501"
else
    print_warning "Application might not be accessible yet. Check logs for details."
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo "=================================="
echo "🌐 Your application should be available at:"
echo "   http://$DOMAIN"
echo "   http://$(curl -s ifconfig.me)"  # External IP
echo ""
echo "📊 Useful commands:"
echo "   Check service status: sudo systemctl status $SERVICE_NAME"
echo "   View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "   Restart service: sudo systemctl restart $SERVICE_NAME"
echo "   Check Nginx status: sudo systemctl status nginx"
echo ""
echo "🔧 Next steps:"
echo "   1. Update DNS records to point to this server"
echo "   2. Configure SSL certificate with: sudo certbot --nginx -d $DOMAIN"
echo "   3. Test the application in your browser"
echo ""
print_status "Deployment completed! 🚀"
