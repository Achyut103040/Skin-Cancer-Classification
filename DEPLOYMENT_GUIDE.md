# 🚀 Deployment Guide - Skin Cancer Detection System

## Overview
This guide covers multiple deployment options for your skin cancer detection application (both Flask and Streamlit versions).

---

## 📊 Quick Comparison

| Platform | Type | Cost | Difficulty | Best For |
|----------|------|------|-----------|----------|
| **Streamlit Cloud** | Streamlit | FREE | ⭐ Easy | Quick sharing, demos |
| **Hugging Face Spaces** | Both | FREE | ⭐⭐ Easy | ML models, portfolio |
| **Render** | Flask/Streamlit | FREE tier | ⭐⭐ Medium | Production apps |
| **Railway** | Flask/Streamlit | FREE tier | ⭐⭐ Medium | Full-stack apps |
| **PythonAnywhere** | Flask | FREE tier | ⭐⭐ Medium | Flask-specific |
| **Heroku** | Flask/Streamlit | PAID | ⭐⭐⭐ Medium | Enterprise |
| **AWS/Azure/GCP** | Both | Variable | ⭐⭐⭐⭐ Hard | Scalable production |

---

## 🎯 RECOMMENDED: Streamlit Cloud (FREE & Easiest)

### Why Streamlit Cloud?
- ✅ **100% FREE** for public repos
- ✅ **No credit card required**
- ✅ **Auto-deploys** from GitHub
- ✅ **Perfect for ML/AI** applications
- ✅ **Handles large models** (up to 1GB)

### Step-by-Step Deployment:

#### 1. Prepare Your Repository

Create these files in your repo root:

**`.streamlit/config.toml`** (Already configured):
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false
```

**`requirements.txt`** (For Streamlit deployment):
```txt
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
Pillow==10.0.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
opencv-python-headless==4.8.0.76
scipy==1.11.1
```

**`.gitignore`** (Important!):
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
my_env/
venv/
.env
.venv
*.pth
!best_skin_cancer_model_balanced.pth
HAM10000_images_part_1/
HAM10000_images_part_2/
*.ipynb_checkpoints
.DS_Store
```

#### 2. Push to GitHub

```bash
cd "D:\Skin Cancer"
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

#### 3. Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with GitHub
3. **Click**: "New app"
4. **Configure**:
   - Repository: `Achyut103040/Skin-Cancer-Classification`
   - Branch: `main`
   - Main file path: `web_interface/streamlit_web_app.py`
5. **Click**: "Deploy!"

⏱️ Deployment takes 5-10 minutes

#### 4. Your App is Live! 🎉
- URL: `https://achyut103040-skin-cancer-detection.streamlit.app`
- Auto-updates on each git push
- Built-in analytics dashboard

---

## 🤗 Option 2: Hugging Face Spaces (FREE)

### Why Hugging Face?
- ✅ **FREE for all** (no limits on private repos)
- ✅ **ML-focused** platform
- ✅ **Great for portfolio**
- ✅ **Supports Gradio, Streamlit, and Flask**

### Deployment Steps:

#### 1. Create Space

1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - Space name: `skin-cancer-detection`
   - SDK: **Streamlit**
   - Visibility: Public/Private

#### 2. Upload Files

Create `app.py` (rename from `streamlit_web_app.py`):
```bash
cp web_interface/streamlit_web_app.py app.py
```

Create `requirements.txt`:
```txt
streamlit
torch
torchvision
Pillow
numpy
pandas
matplotlib
seaborn
opencv-python-headless
scipy
```

#### 3. Push to Hugging Face

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/skin-cancer-detection
cd skin-cancer-detection
cp "D:\Skin Cancer\app.py" .
cp "D:\Skin Cancer\requirements.txt" .
cp "D:\Skin Cancer\best_skin_cancer_model_balanced.pth" .
cp -r "D:\Skin Cancer\benign_cascade_results" .
git add .
git commit -m "Initial deployment"
git push
```

🎉 Live at: `https://huggingface.co/spaces/YOUR_USERNAME/skin-cancer-detection`

---

## 🌐 Option 3: Render (FREE Tier)

### Why Render?
- ✅ **FREE tier** with 750 hours/month
- ✅ **Supports Flask AND Streamlit**
- ✅ **Auto-deploy** from GitHub
- ✅ **Custom domains** available

### Flask Deployment:

#### 1. Create `render.yaml`

```yaml
services:
  - type: web
    name: skin-cancer-detection
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn enhanced_app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

#### 2. Create `requirements.txt` (Flask version)

```txt
Flask==3.0.0
gunicorn==21.2.0
torch==2.0.1
torchvision==0.15.2
Pillow==10.0.0
numpy==1.24.3
opencv-python-headless==4.8.0.76
Werkzeug==3.0.0
```

#### 3. Deploy

1. Go to: https://render.com
2. Sign in with GitHub
3. Click "New +" → "Web Service"
4. Connect your repository
5. Configure:
   - Name: `skin-cancer-detection`
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn enhanced_app:app`
6. Click "Create Web Service"

⏱️ First deployment: ~10 minutes

### Streamlit on Render:

Change start command to:
```bash
streamlit run web_interface/streamlit_web_app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🚂 Option 4: Railway (FREE $5 Credit Monthly)

### Why Railway?
- ✅ **$5 FREE credit** every month
- ✅ **Easy deployment**
- ✅ **Great for hobby projects**

### Steps:

1. Go to: https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Python
6. Add environment variables if needed
7. Deploy! 🚀

---

## 🐍 Option 5: PythonAnywhere (FREE - Flask Only)

### Why PythonAnywhere?
- ✅ **FREE tier** available
- ✅ **Flask-optimized**
- ✅ **Easy setup**

### Steps:

1. Sign up: https://www.pythonanywhere.com
2. Go to "Web" tab
3. Click "Add a new web app"
4. Choose Flask
5. Upload your files via "Files" tab
6. Configure WSGI file
7. Reload web app

⚠️ **Limitations**: 
- Free tier: limited CPU/bandwidth
- No GPU support
- Max file size: 512MB

---

## 📦 Model Size Optimization (Important!)

Your model files are large. Here's how to handle them:

### Option A: Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS"
git push
```

### Option B: External Storage (Recommended)

Store models on cloud storage:

**Using Google Drive:**
```python
import gdown

# In your app initialization
model_url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
gdown.download(model_url, "model.pth", quiet=False)
```

**Using Hugging Face Hub:**
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/skin-cancer-models",
    filename="best_skin_cancer_model_balanced.pth"
)
```

### Option C: Model Quantization

Reduce model size by 75%:

```python
import torch

# Load model
model = torch.load("model.pth")

# Quantize
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save
torch.save(model_quantized, "model_quantized.pth")
```

---

## 🔒 Environment Variables (Security)

Never commit sensitive data! Use environment variables:

**Create `.env` file (local only):**
```env
SECRET_KEY=your-secret-key-here
MODEL_PATH=/path/to/models
DEBUG=False
```

**Access in code:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
MODEL_PATH = os.getenv('MODEL_PATH', 'default/path')
```

**Set on deployment platform:**
- Streamlit Cloud: Settings → Secrets
- Render: Environment → Environment Variables
- Railway: Variables tab
- Heroku: Settings → Config Vars

---

## 🎯 Deployment Checklist

Before deploying, ensure:

- [ ] `requirements.txt` is complete and tested
- [ ] Large files are in Git LFS or external storage
- [ ] Sensitive data is in environment variables
- [ ] `.gitignore` excludes unnecessary files
- [ ] App runs locally without errors
- [ ] Model files are accessible
- [ ] CORS is configured (for API access)
- [ ] Error handling is implemented
- [ ] Loading states are shown to users
- [ ] Documentation is up to date

---

## 🚀 Quick Start Command Summary

### For Streamlit Cloud:
```bash
# 1. Commit and push
git add .
git commit -m "Deploy to Streamlit Cloud"
git push

# 2. Go to share.streamlit.io and deploy
```

### For Hugging Face:
```bash
# 1. Clone HF space
git clone https://huggingface.co/spaces/USERNAME/SPACE_NAME
cd SPACE_NAME

# 2. Copy files
cp /path/to/app.py .
cp /path/to/requirements.txt .

# 3. Push
git add .
git commit -m "Deploy"
git push
```

### For Render/Railway:
```bash
# Just push to GitHub - they auto-deploy!
git push origin main
```

---

## 📊 Performance Optimization Tips

### 1. Model Caching
```python
@st.cache_resource
def load_model():
    return torch.load("model.pth")
```

### 2. Image Preprocessing
```python
# Resize images before processing
max_size = 1024
if img.width > max_size or img.height > max_size:
    img.thumbnail((max_size, max_size))
```

### 3. Lazy Loading
```python
# Load models only when needed
if prediction_button_clicked:
    model = load_model()
```

---

## 🆘 Common Issues & Solutions

### Issue 1: "Module not found"
**Solution**: Add missing package to `requirements.txt`

### Issue 2: "Memory limit exceeded"
**Solution**: 
- Use model quantization
- Reduce batch size
- Use external model storage

### Issue 3: "Port already in use"
**Solution**: 
```bash
# Kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### Issue 4: "CUDA not available"
**Solution**: Use CPU version:
```txt
# In requirements.txt
torch==2.0.1+cpu
torchvision==0.15.2+cpu
-f https://download.pytorch.org/whl/torch_stable.html
```

---

## 📈 Monitoring & Analytics

### Streamlit Cloud:
- Built-in analytics dashboard
- View usage, errors, and performance

### Custom Analytics:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction: {result}, Confidence: {confidence}")
```

### Error Tracking:
Use Sentry for production:
```python
import sentry_sdk

sentry_sdk.init(dsn="YOUR_SENTRY_DSN")
```

---

## 🎓 Next Steps

1. **Deploy to Streamlit Cloud** (Easiest - Start Here!)
2. **Test thoroughly** with various images
3. **Share with colleagues** for feedback
4. **Monitor performance** and errors
5. **Iterate and improve** based on usage
6. **Consider paid tier** if you need more resources

---

## 📞 Support Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Hugging Face Docs**: https://huggingface.co/docs
- **Render Docs**: https://render.com/docs
- **Railway Docs**: https://docs.railway.app/

---

**Ready to Deploy?** Start with **Streamlit Cloud** - it's the easiest! 🚀

**Estimated Time to Deploy**: 15-20 minutes (first time)

Good luck! 🎉
