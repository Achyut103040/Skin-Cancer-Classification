"""
Gradio Interface for Skin Cancer Detection AI
Alternative to Streamlit for Hugging Face Spaces
"""

import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your model classes
from Binary_PyTorch_Fixed_Complete import SkinCancerModel
from Benign_Cascade_Classifier import CascadeClassifier

# Global variables for models
binary_model = None
cascade_classifier = None
device = torch.device('cpu')  # Use CPU for Hugging Face Spaces

def load_models():
    """Load all models on startup"""
    global binary_model, cascade_classifier
    
    try:
        # Load binary classification model
        binary_model = SkinCancerModel(architecture='resnet50')
        binary_model.load_state_dict(torch.load('best_skin_cancer_model_balanced.pth', map_location=device))
        binary_model.eval()
        
        # Load cascade classifier
        cascade_classifier = CascadeClassifier()
        
        return "✅ All models loaded successfully!"
    
    except Exception as e:
        return f"❌ Error loading models: {str(e)}"

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def analyze_skin_lesion(image):
    """Main analysis function"""
    if binary_model is None or cascade_classifier is None:
        return "❌ Models not loaded. Please refresh the page.", None, None
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Stage 1: Binary Classification
        with torch.no_grad():
            binary_output = binary_model(input_tensor)
            binary_prob = torch.nn.functional.softmax(binary_output, dim=1)
            binary_confidence = float(binary_prob.max())
            is_malignant = binary_prob[0][1] > 0.5
        
        # Stage 2: Cascade Classification (if malignant)
        cascade_results = "Not applicable (Benign)"
        cascade_confidence = "N/A"
        
        if is_malignant:
            try:
                cascade_pred = cascade_classifier.predict_single_image(image)
                cascade_results = cascade_pred['predicted_class']
                cascade_confidence = f"{cascade_pred['confidence']:.1%}"
            except:
                cascade_results = "Malignant (Cascade analysis unavailable)"
                cascade_confidence = "N/A"
        
        # Format results
        binary_result = "🔴 MALIGNANT" if is_malignant else "🟢 BENIGN"
        
        # Create detailed report
        report = f"""
## 🔬 **AI Analysis Results**

### **Stage 1: Binary Classification**
- **Result**: {binary_result}
- **Confidence**: {binary_confidence:.1%}

### **Stage 2: Specific Type Analysis**
- **Predicted Type**: {cascade_results}
- **Confidence**: {cascade_confidence}

### **⚠️ Medical Disclaimer**
This tool is for educational purposes only. Always consult healthcare professionals for medical advice.

### **📊 Model Information**
- **Binary Model**: ResNet50 (96.1% accuracy)
- **Cascade Model**: EfficientNet-B0 for BCC (94.0% accuracy)
- **Dataset**: HAM10000 (10,015 dermatoscopic images)
"""
        
        return report, binary_result, f"{binary_confidence:.1%}"
    
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}", "Error", "N/A"

# Load models on startup
load_status = load_models()

# Create Gradio interface
with gr.Blocks(
    title="🔬 Skin Cancer Detection AI",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    """
) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🔬 Skin Cancer Detection AI</h1>
        <p>Professional-grade AI system for skin lesion analysis</p>
        <p><strong>Binary Classification (96.1% accuracy) + Cascade Analysis (94.0% accuracy)</strong></p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("## 📤 **Upload Skin Lesion Image**")
            image_input = gr.Image(
                type="pil",
                label="Drop your image here or click to browse",
                height=300
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary",
                size="lg"
            )
            
            # Model status
            gr.Markdown(f"**Model Status**: {load_status}")
            
        with gr.Column(scale=1):
            # Results section
            gr.Markdown("## 📊 **Analysis Results**")
            
            result_text = gr.Markdown(
                "Upload an image to see AI analysis results here.",
                label="Detailed Report"
            )
            
            with gr.Row():
                binary_result = gr.Textbox(
                    label="Binary Classification",
                    placeholder="Waiting for analysis..."
                )
                confidence_result = gr.Textbox(
                    label="Confidence Score",
                    placeholder="Waiting for analysis..."
                )
    
    # Sample images section
    gr.Markdown("## 🖼️ **Try Sample Images**")
    
    # Load sample images if available
    sample_dir = Path("HAM10000_images_part_1")
    if sample_dir.exists():
        sample_images = list(sample_dir.glob("*.jpg"))[:6]  # First 6 images
        if sample_images:
            with gr.Row():
                for img_path in sample_images:
                    gr.Image(
                        str(img_path),
                        label=f"Sample: {img_path.name}",
                        height=100,
                        show_download_button=False,
                        interactive=True
                    )
    
    # Information section
    with gr.Accordion("📚 About This AI System", open=False):
        gr.Markdown("""
        ### **🎯 How It Works**
        1. **Stage 1**: Binary classification (Malignant vs Benign) using ResNet50
        2. **Stage 2**: If malignant, specific type classification using cascade of specialized models
        3. **Stage 3**: Confidence scoring and detailed analysis
        
        ### **🔬 Model Performance**
        - **Binary Model**: 96.1% accuracy on HAM10000 dataset
        - **BCC Detection**: 94.0% accuracy using EfficientNet-B0
        - **Training Data**: 10,015 dermatoscopic images from HAM10000
        
        ### **⚠️ Important Notes**
        - This is an educational tool for learning about AI in medical imaging
        - Results should never replace professional medical consultation
        - Always consult qualified dermatologists for medical advice
        
        ### **🚀 Technical Details**
        - **Framework**: PyTorch
        - **Architecture**: ResNet50 + EfficientNet-B0 cascade
        - **Deployment**: Hugging Face Spaces
        - **Processing**: CPU-optimized for free hosting
        """)
    
    # Connect the button to the analysis function
    analyze_btn.click(
        fn=analyze_skin_lesion,
        inputs=[image_input],
        outputs=[result_text, binary_result, confidence_result]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )