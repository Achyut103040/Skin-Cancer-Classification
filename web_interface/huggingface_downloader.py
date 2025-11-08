"""
Hugging Face Hub Model Downloader
Upload models to Hugging Face Hub and download automatically
"""

import os
from huggingface_hub import hf_hub_download
import streamlit as st

# Your Hugging Face repository
HF_REPO = "YOUR_USERNAME/skin-cancer-models"

@st.cache_resource(show_spinner=False)
def download_from_huggingface():
    """Download models from Hugging Face Hub."""
    
    model_files = {
        'binary': 'best_skin_cancer_model_balanced.pth',
        'nv': 'nv_model.pth',
        'bkl': 'bkl_model.pth',
        'bcc': 'bcc_model.pth',
        'akiec': 'akiec_model.pth',
        'vasc': 'vasc_model.pth',
    }
    
    downloaded_paths = {}
    
    with st.spinner("🔄 Loading models from Hugging Face..."):
        for model_name, filename in model_files.items():
            try:
                # Downloads to cache, returns path
                model_path = hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filename,
                    cache_dir="./hf_cache"
                )
                downloaded_paths[model_name] = model_path
                st.success(f"✅ Loaded {model_name}")
            except Exception as e:
                st.error(f"❌ Failed to load {model_name}: {e}")
                raise
    
    return downloaded_paths

# Usage in your app
model_paths = download_from_huggingface()
