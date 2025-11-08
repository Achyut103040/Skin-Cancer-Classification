"""
Model Downloader - Automatically downloads models from Google Drive
Add this to the beginning of your streamlit_web_app.py
"""

import os
import gdown
import streamlit as st

# Model URLs from Google Drive (Public sharing links)
MODEL_URLS = {
    'binary': 'https://drive.google.com/uc?id=YOUR_BINARY_MODEL_ID',
    'nv': 'https://drive.google.com/uc?id=YOUR_NV_MODEL_ID',
    'bkl': 'https://drive.google.com/uc?id=YOUR_BKL_MODEL_ID',
    'bcc': 'https://drive.google.com/uc?id=YOUR_BCC_MODEL_ID',
    'akiec': 'https://drive.google.com/uc?id=YOUR_AKIEC_MODEL_ID',
    'vasc': 'https://drive.google.com/uc?id=YOUR_VASC_MODEL_ID',
}

@st.cache_resource(show_spinner=False)
def download_models():
    """Download models from Google Drive if not present locally."""
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_files = {
        'binary': os.path.join(models_dir, 'best_skin_cancer_model_balanced.pth'),
        'nv': os.path.join(models_dir, 'nv_model.pth'),
        'bkl': os.path.join(models_dir, 'bkl_model.pth'),
        'bcc': os.path.join(models_dir, 'bcc_model.pth'),
        'akiec': os.path.join(models_dir, 'akiec_model.pth'),
        'vasc': os.path.join(models_dir, 'vasc_model.pth'),
    }
    
    with st.spinner("🔄 Downloading models (first time only)..."):
        for model_name, file_path in model_files.items():
            if not os.path.exists(file_path):
                try:
                    st.info(f"Downloading {model_name} model...")
                    gdown.download(MODEL_URLS[model_name], file_path, quiet=False)
                    st.success(f"✅ {model_name} model downloaded!")
                except Exception as e:
                    st.error(f"❌ Failed to download {model_name}: {e}")
                    raise
    
    return model_files

# Call this before loading models
model_paths = download_models()
