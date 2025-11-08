# 📦 Google Drive Model Setup Guide

## ✅ Models Already Uploaded to Google Drive!

Great! Now let's configure your app to download them automatically when deployed.

---

## 🔑 Step 1: Get Your Google Drive File IDs

For each model file in your Google Drive:

1. **Right-click the file** → **Share** → **Copy link**
2. Your link looks like:
   ```
   https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=sharing
                                  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                                  This is the FILE ID
   ```

### Files You Need IDs For:

- [ ] `best_skin_cancer_model_balanced.pth` (Binary classifier)
- [ ] `nv_model.pth` (Melanocytic Nevus)
- [ ] `bkl_model_cascade_fixed.pth` (Benign Keratosis)
- [ ] `bcc_model.pth` (Basal Cell Carcinoma)
- [ ] `akiec_model.pth` (Actinic Keratosis)
- [ ] `vasc_model.pth` (Vascular Lesion)

---

## 🔧 Step 2: Update Model URLs in Code

Open `d:\Skin Cancer\web_interface\streamlit_web_app.py`

Find this section (around line 40-50):

```python
# Google Drive Model URLs - Replace with your actual file IDs
GDRIVE_MODEL_URLS = {
    'binary': 'https://drive.google.com/uc?id=YOUR_BINARY_MODEL_ID',
    'nv': 'https://drive.google.com/uc?id=YOUR_NV_MODEL_ID',
    'bkl': 'https://drive.google.com/uc?id=YOUR_BKL_MODEL_ID',
    'bcc': 'https://drive.google.com/uc?id=YOUR_BCC_MODEL_ID',
    'akiec': 'https://drive.google.com/uc?id=YOUR_AKIEC_MODEL_ID',
    'vasc': 'https://drive.google.com/uc?id=YOUR_VASC_MODEL_ID',
}
```

**Replace** `YOUR_BINARY_MODEL_ID` etc. with actual file IDs from Step 1.

### Example:
```python
GDRIVE_MODEL_URLS = {
    'binary': 'https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j',
    'nv': 'https://drive.google.com/uc?id=2b3c4d5e6f7g8h9i0j1k',
    'bkl': 'https://drive.google.com/uc?id=3c4d5e6f7g8h9i0j1k2l',
    'bcc': 'https://drive.google.com/uc?id=4d5e6f7g8h9i0j1k2l3m',
    'akiec': 'https://drive.google.com/uc?id=5e6f7g8h9i0j1k2l3m4n',
    'vasc': 'https://drive.google.com/uc?id=6f7g8h9i0j1k2l3m4n5o',
}
```

---

## 🚀 Step 3: Enable Google Drive Download

Find this line (around line 60):

```python
USE_GDRIVE_MODELS = False  # Set to True when you want to use Google Drive models
```

**Change to**:

```python
USE_GDRIVE_MODELS = True  # Models will be downloaded from Google Drive
```

---

## 📋 Step 4: Update requirements.txt

Add `gdown` to your `requirements.txt`:

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
gdown==4.7.1
```

---

## ⚙️ How It Works

### When Deployed:

1. **First Run**: 
   - App checks if models exist locally
   - If not, downloads from Google Drive
   - Caches them for future use
   - Takes ~2-3 minutes (one-time only)

2. **Subsequent Runs**:
   - Uses cached models
   - No re-download needed
   - Instant loading

3. **User Experience**:
   ```
   🔄 Downloading models from Google Drive (first time only)...
   Downloading binary model...
   ✅ binary model downloaded!
   Downloading nv model...
   ✅ nv model downloaded!
   ...
   ```

---

## 🧪 Testing Locally

1. **Enable Google Drive**:
   ```python
   USE_GDRIVE_MODELS = True
   ```

2. **Temporarily rename local models** (to test download):
   ```bash
   cd "D:\Skin Cancer"
   ren best_skin_cancer_model_balanced.pth best_skin_cancer_model_balanced.pth.backup
   ```

3. **Run Streamlit**:
   ```bash
   streamlit run web_interface\streamlit_web_app.py
   ```

4. **Watch the download happen**!

5. **Restore local models after testing**:
   ```bash
   ren best_skin_cancer_model_balanced.pth.backup best_skin_cancer_model_balanced.pth
   ```

---

## 🔒 Important: Google Drive File Permissions

### Make Files Public:

For each model file:

1. Right-click file → **Share**
2. Click "**Change to anyone with the link**"
3. Set to: **Anyone with the link can view**
4. Copy the shareable link

**⚠️ Security Note**: Models are AI weights (not personal data), so public sharing is safe.

---

## 🌐 Deployment Options Now

### Option A: Streamlit Cloud (with Google Drive)

**Pros**:
- ✅ No model upload to GitHub needed
- ✅ Automatically downloads from Google Drive
- ✅ FREE
- ✅ Easy deployment

**Steps**:
1. Update file IDs in code
2. Set `USE_GDRIVE_MODELS = True`
3. Add `gdown` to `requirements.txt`
4. Push to GitHub
5. Deploy on Streamlit Cloud
6. First deployment takes ~5 mins (downloads models)
7. Subsequent loads are instant!

### Option B: Hugging Face Spaces

**Better alternative** for ML models:

1. Upload models directly to Hugging Face
2. No Google Drive needed
3. Faster loading
4. Better for ML projects

```python
# Use this instead:
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/skin-cancer-models",
    filename="best_skin_cancer_model_balanced.pth"
)
```

---

## 📊 Comparison

| Method | GitHub Size | First Load | Subsequent | Maintenance |
|--------|-------------|------------|------------|-------------|
| **Local Files** | Large (~500MB) | Fast | Fast | Must upload models |
| **Google Drive** | Small (~50MB) | ~3 mins | Fast | Just update IDs |
| **Hugging Face** | Small (~50MB) | ~2 mins | Fast | Best for ML |

---

## 🆘 Troubleshooting

### Error: "gdown not installed"
```bash
pip install gdown
```

### Error: "Failed to download"
- Check Google Drive file permissions (must be public/anyone with link)
- Check file ID is correct
- Try downloading manually to test the link

### Error: "Module cv2 not found"
```bash
pip install opencv-python-headless
```

### Download is slow
- Normal for first time (models are ~100MB each)
- Streamlit Cloud has good bandwidth
- Consider Hugging Face for faster downloads

---

## 🎯 Recommended Approach

### For Quick Testing/Demo:
✅ **Use Google Drive** (what we just set up)

### For Production/Portfolio:
✅ **Upload to Hugging Face Hub**
- Create account: https://huggingface.co/
- Create model repository
- Upload .pth files
- Update code to use `hf_hub_download`

---

## 📞 Need Help?

If you're stuck, I can:
1. Help you extract file IDs from your Google Drive links
2. Set up Hugging Face alternative
3. Configure Git LFS for GitHub
4. Test the deployment

**Just share your Google Drive folder link or file IDs!** 🚀

---

*Last Updated: November 9, 2025*
