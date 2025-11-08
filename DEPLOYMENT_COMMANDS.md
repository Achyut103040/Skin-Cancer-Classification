# 🚀 Complete Deployment Guide - Step by Step Commands
# Skin Cancer Detection System with Google Drive Models

## ✅ Configuration Complete!
Your Google Drive file IDs have been integrated into the system.

---

## 📋 STEP-BY-STEP DEPLOYMENT COMMANDS

### ⚙️ **Step 1: Prepare for GitHub Upload**

Open Command Prompt (cmd) and run these commands:

```cmd
cd /d "D:\Skin Cancer"

REM Create .gitignore to exclude large files
echo __pycache__/ > .gitignore
echo *.pyc >> .gitignore
echo my_env/ >> .gitignore
echo venv/ >> .gitignore
echo .env >> .gitignore
echo *.log >> .gitignore
echo HAM10000_images_part_1/ >> .gitignore
echo HAM10000_images_part_2/ >> .gitignore
echo combined_dataset/ >> .gitignore
echo kfold_combined_dataset/ >> .gitignore

REM IMPORTANT: Exclude local model files (we'll use Google Drive)
echo best_skin_cancer_model_balanced.pth >> .gitignore
echo benign_cascade_results/models/*.pth >> .gitignore

REM Copy deployment requirements
copy requirements_deploy.txt requirements.txt
```

---

### 📤 **Step 2: Initialize Git and Push to GitHub**

```cmd
REM Initialize git (if not already done)
git init

REM Add all files (excluding those in .gitignore)
git add .

REM Commit changes
git commit -m "Deploy Skin Cancer Detection with Google Drive models - Lesion Detection v2.0"

REM Add your GitHub repository as remote (if not already added)
REM Replace with your actual repo URL if different
git remote add origin https://github.com/Achyut103040/Skin-Cancer-Classification.git

REM Push to GitHub
git branch -M main
git push -u origin main
```

**⚠️ If you get "remote already exists" error:**
```cmd
git remote remove origin
git remote add origin https://github.com/Achyut103040/Skin-Cancer-Classification.git
git push -u origin main
```

---

### 🌐 **Step 3: Deploy to Streamlit Cloud**

#### **Option A: Using Web Interface (Recommended)**

1. **Open Browser** and go to: https://share.streamlit.io/

2. **Sign in** with GitHub account

3. **Click "New app"** button

4. **Fill in the form**:
   - **Repository**: `Achyut103040/Skin-Cancer-Classification`
   - **Branch**: `main`
   - **Main file path**: `web_interface/streamlit_web_app.py`
   - **App URL** (optional): Choose custom URL or use auto-generated

5. **Advanced settings** (click to expand):
   - **Python version**: `3.12`
   - **Requirements file**: `requirements.txt` (should auto-detect)

6. **Click "Deploy!"**

7. **Wait 5-10 minutes**:
   - First time: Downloads models from Google Drive (~3 mins)
   - Installs dependencies
   - Starts app

8. **Your app is LIVE!** 🎉
   - URL will be: `https://achyut103040-skin-cancer-classification.streamlit.app`
   - Or your custom URL

---

#### **Option B: Using Streamlit CLI (Advanced)**

```cmd
REM Install streamlit CLI
pip install streamlit

REM Login to Streamlit Cloud
streamlit cloud login

REM Deploy
streamlit cloud deploy web_interface/streamlit_web_app.py
```

---

### 🧪 **Step 4: Test Locally First (Optional but Recommended)**

Before deploying, test that Google Drive download works:

```cmd
cd /d "D:\Skin Cancer\web_interface"

REM Install gdown if not already installed
pip install gdown

REM Run the app locally
streamlit run streamlit_web_app.py --server.port 8502
```

**Expected behavior**:
- App starts on http://localhost:8502
- On first load, it will download models from Google Drive
- You'll see: "🔄 Downloading models from Google Drive..."
- Takes ~2-3 minutes for all models
- Subsequent loads are instant!

**Test with an image** to ensure everything works before deploying!

---

### 📊 **Step 5: Verify Deployment**

After Streamlit Cloud deployment completes:

1. **Open your app URL**
2. **First visit takes ~3-5 minutes** (downloading models from Google Drive)
3. **You'll see progress messages**:
   ```
   🔄 Downloading models from Google Drive (first time only)...
   Downloading binary model...
   ✅ binary model downloaded!
   Downloading nv model...
   ✅ nv model downloaded!
   ...
   ✅ All models loaded successfully!
   ```
4. **Upload a test image**
5. **Click "🔍 Analyze Lesion"**
6. **Check if lesion detection works** ✅

---

### 🔄 **Step 6: Update/Redeploy (Future Updates)**

When you make changes:

```cmd
cd /d "D:\Skin Cancer"

REM Stage changes
git add .

REM Commit with descriptive message
git commit -m "Update: Description of changes"

REM Push to GitHub
git push origin main
```

**Streamlit Cloud will automatically redeploy!** 🚀

---

## 📁 FILES CHECKLIST

Ensure these files exist before deploying:

- [x] `web_interface/streamlit_web_app.py` (Main app - **UPDATED** with your Google Drive IDs)
- [x] `requirements.txt` or `requirements_deploy.txt` (Dependencies with gdown)
- [x] `.gitignore` (Excludes large files)
- [x] `README.md` (Project documentation)
- [x] `.streamlit/config.toml` (Streamlit configuration)

**NOT needed in GitHub** (thanks to Google Drive):
- ❌ `best_skin_cancer_model_balanced.pth` (~100MB)
- ❌ `benign_cascade_results/models/*.pth` (~500MB total)
- ❌ Image datasets

---

## 🆘 TROUBLESHOOTING

### Issue 1: "Git push rejected"
```cmd
git pull origin main --rebase
git push origin main
```

### Issue 2: "Port 8502 already in use"
```cmd
netstat -ano | findstr :8502
taskkill /PID <PID_NUMBER> /F
```

### Issue 3: "Module 'gdown' not found"
```cmd
pip install gdown
```

### Issue 4: "Failed to download from Google Drive"
**Solution**: Make sure all your Google Drive files are:
1. Set to "Anyone with the link can view"
2. Not restricted by organization policies
3. Not deleted or moved

**Test a download manually**:
```cmd
pip install gdown
gdown "https://drive.google.com/uc?id=1LJefcrYSiUOPID-McuxRScoMCGiAVnIF" -O test_model.pth
```

### Issue 5: "Streamlit Cloud build failed"
Check logs for missing dependencies and add to `requirements.txt`

---

## 🎯 QUICK COMMAND SUMMARY

**Complete deployment in 5 commands:**

```cmd
cd /d "D:\Skin Cancer"
copy requirements_deploy.txt requirements.txt
git add .
git commit -m "Deploy with Google Drive models"
git push origin main
```

Then go to: **https://share.streamlit.io/** and deploy!

---

## ✨ FEATURES ENABLED

Your deployed app will have:

✅ **Lesion Detection v2.0** - Automatic ROI detection  
✅ **Multi-Lesion Analysis** - Analyze multiple lesions per image  
✅ **6 AI Models** - Binary + 5 cascade classifiers  
✅ **GradCAM Visualization** - AI attention maps  
✅ **Adjustable Sensitivity** - Low/Medium/High detection  
✅ **Google Drive Models** - No large files in repo  
✅ **Fast Loading** - Models cached after first download  
✅ **Professional UI** - Gradient design, responsive layout  

---

## 📞 SUPPORT

**Streamlit Cloud Dashboard**: https://share.streamlit.io/  
**App Logs**: Available in Streamlit Cloud dashboard  
**GitHub Repo**: https://github.com/Achyut103040/Skin-Cancer-Classification  

---

## 🎉 YOU'RE READY!

**Estimated Total Time**: 15-20 minutes  
**First App Load**: 3-5 minutes (downloading models)  
**Subsequent Loads**: Instant! ⚡

**Start with Step 1 above and follow the commands!** 🚀

---

*Last Updated: November 9, 2025*
*Configuration: Google Drive Models Integrated*
