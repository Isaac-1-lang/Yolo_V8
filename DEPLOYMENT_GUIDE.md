# 🚀 Server Deployment Guide - Waste Classification Model

This guide will help you deploy your waste classification model on a server so others can access it via web browser.

## 📋 Prerequisites

- **Server**: Ubuntu 20.04+ or CentOS 7+ (recommended)
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 10GB free space
- **GPU**: Optional but recommended for better performance

## 🛠️ Server Setup

### 1. Update System Packages

```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y

# CentOS/RHEL
sudo yum update -y
```

### 2. Install Required System Packages

```bash
# Ubuntu/Debian
sudo apt install -y python3 python3-pip python3-venv nginx git curl

# CentOS/RHEL
sudo yum install -y python3 python3-pip nginx git curl
```

### 3. Install CUDA (Optional - for GPU acceleration)

```bash
# Download and install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-12-0-local/7fa2af80.pub
sudo apt update
sudo apt install -y cuda
```

## 📁 Project Deployment

### 1. Clone/Upload Project

```bash
# If using Git
git clone <your-repository-url>
cd WasteNet/Train/Waste-Classification-using-YOLOv8

# Or upload your project files to the server
```

### 2. Create Virtual Environment

```bash
cd WasteNet/Train/Waste-Classification-using-YOLOv8
python3 -m venv waste_classification_env
source waste_classification_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Test the Application

```bash
# Test if everything works
python verify_model_categories.py
python test_inference.py --model "streamlit-detection-tracking - app/weights/best.pt" --image "dataset/train/images/0.jpg"
```

## 🌐 Web Server Configuration

### 1. Configure Nginx

Create Nginx configuration:

```bash
sudo nano /etc/nginx/sites-available/waste-classification
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or IP

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

### 2. Enable the Site

```bash
# Create symbolic link
sudo ln -s /etc/nginx/sites-available/waste-classification /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## 🔧 Systemd Service Setup

### 1. Create Systemd Service

```bash
sudo nano /etc/systemd/system/waste-classification.service
```

Add the following content:

```ini
[Unit]
Description=Waste Classification Streamlit App
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/path/to/WasteNet/Train/Waste-Classification-using-YOLOv8
Environment=PATH=/path/to/WasteNet/Train/Waste-Classification-using-YOLOv8/waste_classification_env/bin
ExecStart=/path/to/WasteNet/Train/Waste-Classification-using-YOLOv8/waste_classification_env/bin/streamlit run streamlit-detection-tracking\ -\ app/app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Important**: Replace `/path/to/` with your actual project path!

### 2. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable waste-classification

# Start service
sudo systemctl start waste-classification

# Check status
sudo systemctl status waste-classification
```

## 🔒 SSL/HTTPS Setup (Optional but Recommended)

### 1. Install Certbot

```bash
# Ubuntu/Debian
sudo apt install -y certbot python3-certbot-nginx

# CentOS/RHEL
sudo yum install -y certbot python3-certbot-nginx
```

### 2. Get SSL Certificate

```bash
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoring and Logs

### 1. View Application Logs

```bash
# View service logs
sudo journalctl -u waste-classification -f

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 2. Monitor System Resources

```bash
# Check system resources
htop
nvidia-smi  # If using GPU
```

## 🚀 Production Configuration

### 1. Optimize Streamlit for Production

Create a production config file:

```bash
mkdir -p ~/.streamlit
nano ~/.streamlit/config.toml
```

Add the following configuration:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### 2. Firewall Configuration

```bash
# Ubuntu/Debian
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## 🔄 Deployment Scripts

### 1. Create Deployment Script

```bash
nano deploy.sh
```

Add the following content:

```bash
#!/bin/bash

# Waste Classification Deployment Script
echo "🚀 Starting Waste Classification Deployment..."

# Set variables
PROJECT_DIR="/path/to/WasteNet/Train/Waste-Classification-using-YOLOv8"
SERVICE_NAME="waste-classification"

# Navigate to project directory
cd $PROJECT_DIR

# Activate virtual environment
source waste_classification_env/bin/activate

# Update dependencies
echo "📦 Updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Test the application
echo "🧪 Testing application..."
python verify_model_categories.py

# Restart service
echo "🔄 Restarting service..."
sudo systemctl restart $SERVICE_NAME

# Check service status
echo "📊 Checking service status..."
sudo systemctl status $SERVICE_NAME

echo "✅ Deployment completed!"
echo "🌐 Application should be available at: http://your-domain.com"
```

Make it executable:

```bash
chmod +x deploy.sh
```

### 2. Create Update Script

```bash
nano update.sh
```

Add the following content:

```bash
#!/bin/bash

# Waste Classification Update Script
echo "🔄 Updating Waste Classification..."

# Set variables
PROJECT_DIR="/path/to/WasteNet/Train/Waste-Classification-using-YOLOv8"
SERVICE_NAME="waste-classification"

# Navigate to project directory
cd $PROJECT_DIR

# Pull latest changes (if using Git)
# git pull origin main

# Activate virtual environment
source waste_classification_env/bin/activate

# Update dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Restart service
sudo systemctl restart $SERVICE_NAME

echo "✅ Update completed!"
```

Make it executable:

```bash
chmod +x update.sh
```

## 🎯 Access Your Application

Once deployed, your waste classification application will be available at:

- **HTTP**: `http://your-server-ip` or `http://your-domain.com`
- **HTTPS**: `https://your-domain.com` (if SSL is configured)

## 🔧 Troubleshooting

### Common Issues

1. **Service won't start**:
   ```bash
   sudo journalctl -u waste-classification -f
   ```

2. **Permission issues**:
   ```bash
   sudo chown -R www-data:www-data /path/to/your/project
   ```

3. **Port already in use**:
   ```bash
   sudo netstat -tulpn | grep :8501
   sudo kill -9 <PID>
   ```

4. **Nginx configuration error**:
   ```bash
   sudo nginx -t
   sudo systemctl restart nginx
   ```

### Performance Optimization

1. **Enable GPU acceleration** (if available):
   - Install CUDA toolkit
   - Ensure PyTorch with CUDA support is installed

2. **Optimize Nginx**:
   - Enable gzip compression
   - Set appropriate cache headers

3. **Monitor resources**:
   - Use `htop` to monitor CPU/RAM usage
   - Use `nvidia-smi` to monitor GPU usage

## 📈 Scaling Considerations

- **Load Balancing**: Use multiple instances behind a load balancer
- **Caching**: Implement Redis for session management
- **Database**: Add database for user data and results storage
- **CDN**: Use CloudFlare or similar for static assets

---

**Your waste classification model is now deployed and accessible to users worldwide!** 🌍
