---
title: MsBiCNet - Skin Cancer Detection AI
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# MsBiCNet - Multi-stage Binary Cascade Network

AI-powered skin cancer detection system with **96.1% accuracy** on HAM10000 dataset. Featuring 6 specialized deep learning models for comprehensive skin lesion analysis.

## 🚀 New: Mobile App Available!

📱 **Cross-platform mobile app** for Android, iOS, and Desktop is now available in the `mobile_app/` folder!

- ✅ **6 AI Models**: Binary + 5 Cascade classifiers
- ✅ **Material Design UI**: Modern, responsive interface
- ✅ **Offline Ready**: Models cached after first download
- ✅ **Multi-Page App**: Home, Analysis, History, About, Publications

👉 **[Get Started with Mobile App](mobile_app/README.md)**  
📖 **[Complete Mobile Deployment Guide](MOBILE_APP_GUIDE.md)**

---

## 📱 Platform Options

Choose your deployment platform:

| Platform | Best For | Status |
|----------|----------|--------|
| **[Mobile App](mobile_app/)** | Android, iOS, Desktop | ✅ Ready |
| **[Gradio Web App](app.py)** | Hugging Face Spaces | ✅ Live |
| **[Streamlit App](streamlit_enhanced_app.py)** | Web deployment | ✅ Ready |

---

## Featured Publication
- **MsBiCNet**: Multi-stage Binary Cascade Network for Skin Cancer Detection
- Published in **ICTEAH 2025** (International Conference on Technology Enhanced Academic Health)

## Project Structure

```
d:\Skin Cancer\
├── mobile_app/                    # 📱 NEW: Mobile App (Android/iOS/Desktop)
│   ├── main.py                    # Main app entry
│   ├── buildozer.spec             # Android build config
│   ├── models/
│   │   └── model_manager.py       # 6 AI models handler
│   └── screens/                   # Multi-page UI
│       ├── home_screen.py
│       ├── analysis_screen.py
│       ├── history_screen.py
│       ├── about_screen.py
│       └── publications_screen.py
├── app.py                         # Gradio web app
├── streamlit_enhanced_app.py      # Streamlit web app
├── model_explanations/
│   ├── shap_explanations.py
│   ├── lime_explanations.py
│   ├── gradcam_explanations.py
│   ├── compare_explanations.py
│   ├── requirements.txt
│   ├── shap_results/
│   ├── lime_results/
│   └── gradcam_results/
├── web_interface/
│   ├── enhanced_app.py            # Flask web interface
│   ├── templates/
│   │   ├── index.html
│   │   ├── layout.html
│   │   ├── about.html
│   │   ├── gallery.html
│   │   ├── documentation.html
│   │   ├── publications.html
│   │   └── contact.html
│   ├── static/
│   │   ├── styles.css
│   │   ├── script.js
│   │   └── images/
│   └── uploads/
├── best_skin_cancer_model_balanced.pth
├── training_results/
│   ├── classification_report_pytorch.json
│   └── comprehensive_analysis.png
└── README.md
```

## Key Features

- **High Accuracy Classification**: 99.66% accuracy in binary classification (benign/malignant)
- **Multiple Explanation Methods**: GradCAM, SHAP, and LIME visualization techniques
- **User-Friendly Web Interface**: Upload images and receive instant classifications with explanations
- **Comprehensive Documentation**: Detailed information about model architecture and usage
- **Research Publications**: Access to associated research papers and findings

## Technology Stack
- **Deep Learning**: PyTorch, ResNet50 backbone
- **Explainability**: GradCAM, SHAP, LIME
- **Backend**: Flask, Python 3.8+
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Data Processing**: NumPy, Pandas, OpenCV, PIL
- **Deployment**: Waitress WSGI server

## Key Features

- **High Accuracy Classification**: 99.66% accuracy in binary classification (benign/malignant)
- **Multiple Explanation Methods**: GradCAM, SHAP, and LIME visualization techniques
- **Professional Web Interface**: Upload images and receive instant classifications with explanations
- **Dataset**: Built using the HAM10000 dataset of dermatoscopic images
- **Researcher Profiles**: Accounts for tracking analysis history and saving results
- **Comprehensive Documentation**: Detailed information about model architecture and usage

## Installation

1. Install the required dependencies:

```
pip install -r requirements.txt
```

## Model Explanations

To generate model explanations:

1. Run GradCAM explanations (recommended, works with PyTorch):

```
python model_explanations/gradcam_explanations.py
```

2. Run SHAP explanations (requires additional setup):

```
python model_explanations/shap_explanations.py
```

3. Run LIME explanations (requires additional setup):

```
python model_explanations/lime_explanations.py
```

4. Compare explanation methods:

```
python model_explanations/compare_explanations.py
```

## Web Interface

To start the web application:

```
python web_interface/app.py
```

Then open a browser and navigate to http://localhost:8080

### Web Interface Sections

- **Home**: Upload and classify skin lesion images
- **Gallery**: View example classifications with explanations
- **Documentation**: Technical details and usage instructions
- **Publications**: Research papers and findings
- **About**: Project information and team details
- **Contact**: Get in touch with the research team

## Technical Details

### Model Architecture

- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Custom Classifier**: Multi-layer neural network with dropout for regularization
- **Training Dataset**: HAM10000 (Human Against Machine with 10,000 training images)
- **Data Processing**: Class balancing, data augmentation, and normalization

### Model Performance

- Test Accuracy: 99.66%
- Precision (Benign): 99.78%
- Recall (Benign): 99.55%
- Precision (Malignant): 99.55%
- Recall (Malignant): 99.78%
- F1-Score: 99.66%

### Explainability Methods

1. **GradCAM**: Gradient-weighted Class Activation Mapping to highlight regions influencing predictions
2. **SHAP**: SHapley Additive exPlanations for pixel-level feature importance
3. **LIME**: Local Interpretable Model-agnostic Explanations using superpixels

## Medical Disclaimer

**Important**: This tool is for educational and research purposes only and is not intended to provide medical advice. The predictions made by this application should not be used for diagnosis or treatment decisions. Always consult with a qualified healthcare professional for any medical concerns.

## Research Team

- **Prof. Pramod Kachare**: Project Guide
- **Research Team**: AI & Healthcare Research Group

## Notes

- GradCAM is recommended as it works natively with PyTorch and requires no additional dependencies
- The model was trained on the HAM10000 dataset with extensive data augmentation to balance classes
- This project represents state-of-the-art techniques in explainable AI for medical imaging

## License

This project is licensed under the MIT License - see the LICENSE file for details.

© 2025 DermAI Detect. All rights reserved.
