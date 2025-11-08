# 🚀 Quick Deployment Guide

## 🎯 Fastest Way to Deploy (Streamlit Cloud - FREE)

### Prerequisites
- GitHub account
- Your code pushed to GitHub repository

### 3-Minute Deployment

1. **Prepare requirements file**
   ```bash
   copy requirements_streamlit_cloud.txt requirements.txt
   ```

2. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

3. **Deploy**
   - Go to: https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Fill in:
     - Repository: `Achyut103040/Skin-Cancer-Classification`
     - Branch: `main`
     - Main file: `web_interface/streamlit_web_app.py`
   - Click "Deploy!"

4. **Done!** ✅
   Your app will be live at: `https://achyut103040-skin-cancer-classification.streamlit.app`

---

## 🔥 Alternative: Hugging Face Spaces (FREE)

### Why Hugging Face?
- No GitHub push required
- Direct file upload
- Perfect for ML models
- Great for portfolio

### Steps

1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - Name: `skin-cancer-detection`
   - SDK: **Streamlit**
   - Hardware: **CPU (free)**
4. Upload files:
   - Rename `streamlit_web_app.py` → `app.py`
   - Upload `app.py`
   - Upload `requirements_streamlit_cloud.txt` → `requirements.txt`
   - Upload model files (`.pth`)
   - Upload `benign_cascade_results` folder
5. Wait 2-3 minutes for build
6. **Done!** Your app is live!

---

## 🖥️ Local Testing

### Flask Version (Current)
```bash
cd "D:\Skin Cancer\web_interface"
python enhanced_app.py
```
Access at: http://localhost:5000

### Streamlit Version (Recommended for Deployment)
```bash
cd "D:\Skin Cancer\web_interface"
streamlit run streamlit_web_app.py --server.port 8502
```
Access at: http://localhost:8502

---

## 📦 Model Files Setup

⚠️ **Important**: Model files are large (~100MB each)

### Option 1: Git LFS (Recommended)
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track models with Git LFS"
```

### Option 2: External Hosting
Upload models to Google Drive/Dropbox and download in app:

```python
import gdown

# In your app
model_url = "YOUR_GOOGLE_DRIVE_LINK"
gdown.download(model_url, "model.pth", quiet=False)
```

---

## ⚡ Quick Deployment Script

Just double-click `deploy.bat` and follow the menu!

```
1. Streamlit Cloud (Recommended - FREE)
2. Hugging Face Spaces (FREE)
3. Render (FREE tier)
4. Railway (FREE $5/month credit)
5. Local Flask Server
6. Local Streamlit Server
```

---

## 🆘 Common Issues

### "Module not found"
**Solution**: Install missing package
```bash
pip install <package-name>
```

### "Port already in use"
**Solution**: Kill the process
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### "Model file not found"
**Solution**: Check model path in code
```python
MODEL_PATH = r'd:\Skin Cancer\best_skin_cancer_model_balanced.pth'
```

---

## 📊 Deployment Status

| Platform | Setup Time | Monthly Cost | Status |
|----------|------------|--------------|--------|
| Streamlit Cloud | 3 mins | FREE | ⭐ Recommended |
| Hugging Face | 5 mins | FREE | ✅ Easy |
| Render | 10 mins | FREE tier | ✅ Good |
| Railway | 10 mins | $5 credit | ✅ Good |
| Local | 1 min | FREE | ✅ Testing |

---

## 🎓 Next Steps After Deployment

1. **Test thoroughly** with various images
2. **Share the link** with team/users
3. **Monitor usage** (analytics dashboard)
4. **Gather feedback** and iterate
5. **Update models** as needed

---

## 📚 Full Documentation

For detailed deployment guides, see:
- `DEPLOYMENT_GUIDE.md` - Complete deployment options
- `FREE_DEPLOYMENT_GUIDE.md` - Free deployment focus
- `README.md` - Project overview

---

## 🎉 You're Ready to Deploy!

**Recommended Path**: Start with Streamlit Cloud (easiest)

**Time Required**: 15-20 minutes (first time)

**Good Luck!** 🚀

---

*Last Updated: November 9, 2025*
