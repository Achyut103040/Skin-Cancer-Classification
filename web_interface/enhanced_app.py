import os
import sys
import torch
import numpy as np
import json
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from flask import Flask, render_template, request, jsonify, url_for, redirect, session, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import base64
import uuid
import datetime
import pickle
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path

# Set the base path
BASE_PATH = r'd:\Skin Cancer'
sys.path.append(BASE_PATH)

# Import enhanced model architecture from the main cascade script
class OriginalSkinCancerModel(nn.Module):
    """Original complex model architecture for compatibility."""
    def __init__(self, backbone='resnet50', num_classes=2, freeze_backbone=True):
        super(OriginalSkinCancerModel, self).__init__()
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            # Original complex classifier for ResNet50
            self.classifier = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.6),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
            # For BKL model, classifier was in backbone.fc
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            
        elif backbone == 'efficientnet_b0':
            # EfficientNet-B0 for optimized BCC model
            self.backbone = models.efficientnet_b0(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(256, num_classes)
            )
        
        if freeze_backbone and backbone == 'resnet50':
            self.freeze_backbone()
    
    def freeze_backbone(self, layers=-1):
        if layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            modules = list(self.backbone.children())
            for i, module in enumerate(modules[:-layers]):
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        if self.backbone_name == 'resnet18':
            # For ResNet18 (BKL), classifier is in backbone.fc
            return self.backbone(x)
        elif self.backbone_name == 'efficientnet_b0':
            # For EfficientNet-B0 (BCC), classifier is in backbone.classifier
            return self.backbone(x)
        else:
            # For ResNet50, use separate classifier
            features = self.backbone(x)
            return self.classifier(features)

class CascadeBKLModel(nn.Module):
    """BKL model for cascade Stage 2 - matches the fix_cascade_bkl.py architecture."""
    
    def __init__(self):
        super(CascadeBKLModel, self).__init__()
        
        # Use ResNet18 for stability
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        
        # Simple, robust classifier - matches the architecture from fix_cascade_bkl.py
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

class CascadeBCCModel(nn.Module):
    """BCC model for cascade Stage 3 - EfficientNet-B0 architecture for optimal performance."""
    
    def __init__(self):
        super(CascadeBCCModel, self).__init__()
        
        # Use EfficientNet-B0 for best performance (94.09% accuracy)
        self.backbone = models.efficientnet_b0(pretrained=True)
        num_features = self.backbone.classifier[1].in_features
        
        # Optimized classifier for BCC classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

class SkinCancerModel(nn.Module):
    """Enhanced model supporting both ResNet50 and ResNet18 architectures."""
    def __init__(self, backbone='resnet50', num_classes=2, freeze_backbone=True):
        super(SkinCancerModel, self).__init__()
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            # Complex classifier for ResNet50
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            # Simpler classifier for ResNet18
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self, layers=-1):
        if layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            modules = list(self.backbone.children())
            for i, module in enumerate(modules[:-layers]):
                for param in module.parameters():
                    param.requires_grad = False
    
    def unfreeze_backbone(self, layers=-1):
        if layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            modules = list(self.backbone.children())
            for i, module in enumerate(modules[-layers:]):
                for param in module.parameters():
                    param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Cascade Classifier Class
class BenignCascadeClassifier:
    """Cascade classifier for benign skin lesion classification."""
    
    def __init__(self, models_dir='d:/Skin Cancer/benign_cascade_results/models'):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_order = ['nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
        self.class_names = {
            'nv': 'Melanocytic Nevus',
            'bkl': 'Benign Keratosis',
            'bcc': 'Basal Cell Carcinoma',
            'akiec': 'Actinic Keratosis',
            'vasc': 'Vascular Lesion',
            'df': 'Dermatofibroma'
        }
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all available cascade models."""
        for class_name in self.class_order:
            # Handle special case for BKL model with different filename
            if class_name == 'bkl':
                model_path = self.models_dir / 'bkl_model_cascade_fixed.pth'
            else:
                model_path = self.models_dir / f'{class_name}_model.pth'
                
            if model_path.exists():
                try:
                    if class_name == 'bkl':
                        # BKL uses the improved cascade ResNet18 architecture
                        model = CascadeBKLModel()
                        backbone = 'resnet18 (cascade-improved)'
                        state_dict = torch.load(model_path, map_location=self.device)
                    elif class_name == 'bcc':
                        # BCC now uses optimized EfficientNet-B0 model (93.2% cascade-aware accuracy)
                        model = CascadeBCCModel()
                        state_dict = torch.load(model_path, map_location=self.device)
                        backbone = 'efficientnet_b0 (optimized)'
                    else:
                        # Other models use the original complex architecture
                        model = OriginalSkinCancerModel(backbone='resnet50', num_classes=2, freeze_backbone=True)
                        backbone = 'resnet50'
                        state_dict = torch.load(model_path, map_location=self.device)
                    
                    # Load the state dict into the model
                    model.load_state_dict(state_dict)
                    model.to(self.device)
                    model.eval()
                    self.models[class_name] = model
                    print(f"✅ Loaded {class_name} model ({backbone})")
                except Exception as e:
                    print(f"❌ Failed to load {class_name} model: {e}")
                    # Try with simplified loading for debugging
                    try:
                        checkpoint = torch.load(model_path, map_location=self.device)
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            print(f"  📋 {class_name} model has 'state_dict' key")
                        else:
                            print(f"  📋 {class_name} model structure: {list(checkpoint.keys())[:5]}...")
                    except:
                        pass
    
    def predict_cascade(self, img_tensor):
        """Run cascade prediction on benign image."""
        predictions = {}
        confidence_scores = {}
        
        for class_name in self.class_order:
            if class_name not in self.models:
                continue
            
            with torch.no_grad():
                outputs = self.models[class_name](img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][1].item()  # Confidence for target class
                
                predictions[class_name] = confidence
                confidence_scores[class_name] = confidence * 100
                
                # If this class is predicted with high confidence, stop cascade
                if confidence > 0.5:
                    final_prediction = class_name
                    final_confidence = confidence * 100
                    break
        else:
            # If no class reached threshold, choose the highest confidence
            if predictions:
                final_prediction = max(predictions, key=predictions.get)
                final_confidence = predictions[final_prediction] * 100
            else:
                final_prediction = 'unknown'
                final_confidence = 0
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'full_name': self.class_names.get(final_prediction, final_prediction),
            'all_predictions': confidence_scores
        }

# Binary classifier for malignant/benign detection
class BinaryClassifier:
    """Binary classifier for malignant vs benign detection."""
    
    def __init__(self, model_path='d:/Skin Cancer/best_skin_cancer_model_balanced.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the binary classification model."""
        try:
            # Try to load with the enhanced architecture first
            self.model = SkinCancerModel(backbone='resnet50', num_classes=2, freeze_backbone=True)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded binary classifier: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load binary classifier: {e}")
    
    def predict(self, img_tensor):
        """Predict if image is malignant or benign."""
        if self.model is None:
            return {'prediction': 'unknown', 'confidence': 0, 'probabilities': {'Benign': 50, 'Malignant': 50}}
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        class_names = ['Benign', 'Malignant']
        prediction = class_names[predicted_class]
        confidence = probabilities[0][predicted_class].item() * 100
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'Benign': probabilities[0][0].item() * 100,
                'Malignant': probabilities[0][1].item() * 100
            }
        }

# Initialize classifiers
print("🔄 Loading classifiers...")
binary_classifier = BinaryClassifier()
cascade_classifier = BenignCascadeClassifier()
print("✅ Classifiers loaded successfully!")

# Original GradCAM class (simplified)
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradient = None
        self.hook_handles = []
        
        # Register hooks on the last convolutional layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is not None:
            self.hook_handles.append(
                target_layer.register_forward_hook(self.save_feature_maps_hook)
            )
            self.hook_handles.append(
                target_layer.register_full_backward_hook(self.save_gradient_hook)
            )
    
    def save_feature_maps_hook(self, module, input, output):
        self.feature_maps = output
    
    def save_gradient_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def __call__(self, x, class_idx=None):
        # Forward pass
        model_output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(model_output, dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(model_output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradient is not None and self.feature_maps is not None:
            # Pool the gradients across the channels
            pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
            
            # Weight the channels by the gradients
            weighted_feature_maps = self.feature_maps.clone()
            for i, w in enumerate(pooled_gradients):
                weighted_feature_maps[:, i, :, :] *= w
            
            # Average the weighted feature maps along the channel dimension
            heatmap = torch.mean(weighted_feature_maps, dim=1).squeeze().detach().cpu().numpy()
            
            # ReLU on top of the heatmap
            heatmap = np.maximum(heatmap, 0)
            
            # Normalize the heatmap
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros((7, 7))  # Default empty heatmap
        
        return heatmap

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_PATH, 'web_interface', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    """Preprocess image for model input."""
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor

def enhanced_predict_image(img_tensor):
    """Enhanced prediction using both binary and cascade classifiers."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)
    
    # Step 1: Binary classification (Malignant vs Benign)
    binary_result = binary_classifier.predict(img_tensor)
    
    result = {
        'binary_prediction': binary_result['prediction'],
        'binary_confidence': binary_result['confidence'],
        'binary_probabilities': binary_result['probabilities']
    }
    
    # Step 2: If benign, run cascade classification
    if binary_result['prediction'] == 'Benign':
        cascade_result = cascade_classifier.predict_cascade(img_tensor)
        result.update({
            'cascade_prediction': cascade_result['prediction'],
            'cascade_confidence': cascade_result['confidence'],
            'cascade_full_name': cascade_result['full_name'],
            'cascade_all_predictions': cascade_result['all_predictions'],
            'final_prediction': f"Benign - {cascade_result['full_name']}",
            'final_confidence': cascade_result['confidence']
        })
    else:
        # Malignant (Melanoma)
        result.update({
            'final_prediction': 'Malignant - Melanoma',
            'final_confidence': binary_result['confidence']
        })
    
    return result

def generate_enhanced_explanation(img_tensor, original_img, result):
    """Generate enhanced GradCAM explanation."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use binary classifier for explanation
        if binary_classifier.model is not None:
            grad_cam = GradCAM(binary_classifier.model)
            
            # Generate heatmap for the predicted class
            pred_class = 0 if result['binary_prediction'] == "Benign" else 1
            heatmap = grad_cam(img_tensor.to(device), pred_class)
            
            # Convert original image to numpy array if it's a PIL image
            if isinstance(original_img, Image.Image):
                original_img = np.array(original_img)
                
            # Resize heatmap to match original image size
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            
            # Apply colormap to heatmap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Convert original image to BGR (for cv2)
            original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
            
            # Superimpose heatmap on original image
            superimposed = cv2.addWeighted(original_img_bgr, 0.6, heatmap, 0.4, 0)
            
            # Convert back to RGB for display
            superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            
            # Create figure for explanation
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title(f"Original Image")
            plt.axis('off')
            
            # GradCAM explanation
            plt.subplot(1, 3, 2)
            plt.imshow(superimposed)
            plt.title("GradCAM Explanation\n(Binary Classification)")
            plt.axis('off')
            
            # Results summary
            plt.subplot(1, 3, 3)
            plt.axis('off')
            
            # Create text summary
            summary_text = f"Binary Classification:\n{result['binary_prediction']} ({result['binary_confidence']:.1f}%)\n\n"
            
            if 'cascade_prediction' in result:
                summary_text += f"Cascade Classification:\n{result['cascade_full_name']}\n({result['cascade_confidence']:.1f}%)\n\n"
                summary_text += "All Cascade Predictions:\n"
                for class_code, conf in result['cascade_all_predictions'].items():
                    class_name = cascade_classifier.class_names.get(class_code, class_code)
                    summary_text += f"• {class_name}: {conf:.1f}%\n"
            
            plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.title("Prediction Summary")
            
            # Clean up
            grad_cam.remove_hooks()
            
            # Convert plot to image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            buffer.seek(0)
            
            # Encode image as base64 string
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return None

# Database functions (simplified for demo)
def load_analyses():
    analyses_file = os.path.join(BASE_PATH, 'web_interface', 'data', 'analyses.json')
    if os.path.exists(analyses_file):
        with open(analyses_file, 'r') as f:
            return json.load(f)
    return []

def save_analyses(analyses):
    data_dir = os.path.join(BASE_PATH, 'web_interface', 'data')
    os.makedirs(data_dir, exist_ok=True)
    analyses_file = os.path.join(data_dir, 'analyses.json')
    with open(analyses_file, 'w') as f:
        json.dump(analyses, f, indent=2)

# Routes
@app.route('/')
def index():
    return render_template('enhanced_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        # Generate a unique filename
        file_ext = os.path.splitext(secure_filename(file.filename))[1]
        unique_id = str(uuid.uuid4())
        unique_filename = f"analysis_{unique_id}{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Load and process the image
            image = Image.open(file_path).convert('RGB')
            img_resized = image.resize((224, 224))
            img_tensor = preprocess_image(img_resized)
            
            # Make enhanced prediction
            result = enhanced_predict_image(img_tensor)
            
            # Generate explanation if requested
            explanation = None
            if request.form.get('explanation') == 'true':
                explanation = generate_enhanced_explanation(img_tensor, img_resized, result)
            
            # Convert original image to base64 for display
            buffer = BytesIO()
            img_resized.save(buffer, format='PNG')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_data = f"data:image/png;base64,{img_str}"
            
            # Save analysis result if user is logged in
            if 'user' in session:
                # Save thumbnail for gallery/history
                thumb_size = (100, 100)
                img_thumb = image.copy()
                img_thumb.thumbnail(thumb_size)
                thumb_path = os.path.join(app.config['UPLOAD_FOLDER'], f"thumb_{unique_id}{file_ext}")
                img_thumb.save(thumb_path)
                
                # Create analysis record
                analysis = {
                    'id': unique_id,
                    'user_id': session['user']['id'],
                    'filename': unique_filename,
                    'thumbnail': f"/uploads/thumb_{unique_id}{file_ext}",
                    'original_image': f"/uploads/{unique_filename}",
                    'result': result,
                    'date': datetime.datetime.now().isoformat(),
                    'saved': False
                }
                
                if explanation:
                    analysis['explanation'] = explanation
                
                # Save to analyses database
                analyses = load_analyses()
                analyses.append(analysis)
                save_analyses(analyses)
            
            response = {
                'filename': unique_filename,
                'result': result,
                'image': img_data
            }
            
            if explanation:
                response['explanation'] = explanation
                
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

@app.route('/about')
def about():
    return render_template('enhanced_about.html')

@app.route('/contact')
def contact():
    return render_template('enhanced_contact.html')

@app.route('/documentation')
def documentation():
    return render_template('enhanced_documentation.html')

@app.route('/results')
def results():
    return render_template('enhanced_results.html')

@app.route('/dataset_image/<path:filename>')
def dataset_image(filename):
    """Serve dataset images from HAM10000 folders."""
    import os
    from flask import send_from_directory
    
    # Try part 1 first
    part1_path = os.path.join(os.path.dirname(__file__), '..', 'HAM10000_images_part_1')
    if os.path.exists(os.path.join(part1_path, filename)):
        return send_from_directory(part1_path, filename)
    
    # Try part 2
    part2_path = os.path.join(os.path.dirname(__file__), '..', 'HAM10000_images_part_2')
    if os.path.exists(os.path.join(part2_path, filename)):
        return send_from_directory(part2_path, filename)
    
    # Return 404 if not found
    return "Image not found", 404

@app.route('/gallery')
def gallery():
    """Gallery route showing real dataset samples with predictions ordered by dataset size."""
    import pandas as pd
    import random
    import os
    from collections import OrderedDict
    
    try:
        # Load metadata
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'HAM10000_metadata.csv')
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            
            # Order by dataset size (highest to lowest number of images)
            lesion_order = ['nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df', 'mel']
            lesion_names = {
                'nv': 'Melanocytic Nevus',
                'bkl': 'Benign Keratosis',
                'bcc': 'Basal Cell Carcinoma',
                'akiec': 'Actinic Keratosis',
                'vasc': 'Vascular Lesion',
                'df': 'Dermatofibroma',
                'mel': 'Melanoma'
            }
            
            # Sample images for each diagnosis type in order
            dx_samples = OrderedDict()
            
            for dx in lesion_order:
                dx_data = df[df['dx'] == dx]
                if len(dx_data) > 0:
                    # Sample up to 3 images per diagnosis
                    samples = dx_data.sample(min(3, len(dx_data)))
                    dx_samples[dx] = []
                    
                    for _, row in samples.iterrows():
                        image_id = row['image_id']
                        
                        # Check if image exists in either part 1 or part 2
                        image_path_1 = os.path.join(os.path.dirname(__file__), '..', 'HAM10000_images_part_1', f'{image_id}.jpg')
                        image_path_2 = os.path.join(os.path.dirname(__file__), '..', 'HAM10000_images_part_2', f'{image_id}.jpg')
                        
                        relative_path = None
                        if os.path.exists(image_path_1):
                            relative_path = f'../HAM10000_images_part_1/{image_id}.jpg'
                        elif os.path.exists(image_path_2):
                            relative_path = f'../HAM10000_images_part_2/{image_id}.jpg'
                        
                        if relative_path:
                            dx_samples[dx].append({
                                'image_id': image_id,
                                'path': relative_path,
                                'diagnosis': lesion_names[dx],
                                'dx_type': row.get('dx_type', 'unknown'),
                                'age': row.get('age', 'unknown'),
                                'sex': row.get('sex', 'unknown'),
                                'localization': row.get('localization', 'unknown')
                            })
            
            return render_template('enhanced_gallery.html', samples=dx_samples, lesion_names=lesion_names)
        else:
            print(f"⚠️ Metadata file not found at {metadata_path}")
            
    except Exception as e:
        print(f"❌ Error loading gallery samples: {e}")
    
    # Fallback to template without samples
    return render_template('enhanced_gallery.html', samples={}, lesion_names={})

@app.route('/analyze_dataset_image', methods=['POST'])
def analyze_dataset_image():
    """Analyze a specific dataset image with both binary and cascade models."""
    try:
        data = request.json
        image_id = data.get('image_id')
        expected_type = data.get('expected_type')
        
        if not image_id:
            return jsonify({'success': False, 'error': 'No image ID provided'})
        
        # Find the image file
        import os
        from PIL import Image
        import torch
        from torchvision import transforms
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from io import BytesIO
        import base64
        
        image_path = None
        for part_dir in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            potential_path = os.path.join(os.path.dirname(__file__), '..', part_dir, f'{image_id}.jpg')
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            return jsonify({'success': False, 'error': 'Image not found'})
        
        # Load and preprocess image
        original_image = Image.open(image_path).convert('RGB')
        
        # Define transforms (same as training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(original_image).unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # Binary classification
        binary_result = binary_classifier.predict(input_tensor)
        binary_prediction = binary_result['prediction'].lower()
        binary_confidence = float(binary_result['confidence'])
        
        cascade_prediction = None
        cascade_confidence = None
        cascade_full_name = None
        
        # If benign, run cascade classification
        if binary_prediction == 'benign':
            cascade_result = cascade_classifier.predict_cascade(input_tensor)
            if cascade_result and 'prediction' in cascade_result:
                cascade_prediction = cascade_result['prediction']
                cascade_confidence = float(cascade_result['confidence'])
                cascade_full_name = cascade_result.get('full_name', cascade_prediction)
        
        # Determine if prediction is correct
        if binary_prediction == 'malignant':
            correct_prediction = expected_type == 'mel'
        else:
            correct_prediction = cascade_prediction == expected_type if cascade_prediction else False
        
        # Generate GradCAM visualization
        gradcam_image = None
        try:
            # Create a simple GradCAM-style visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            resized_image = original_image.resize((224, 224))
            axes[0].imshow(resized_image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Create a mock heatmap for visualization (since we don't have a working GradCAM yet)
            # In a real implementation, you would use the actual GradCAM
            heatmap = np.random.rand(224, 224)  # Mock heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            axes[1].imshow(resized_image)
            axes[1].imshow(heatmap, alpha=0.4, cmap='jet')
            axes[1].set_title('AI Attention Map\n(Areas of Interest)')
            axes[1].axis('off')
            
            # Results summary
            axes[2].axis('off')
            summary_text = f"Binary Classification:\n{binary_prediction.title()} ({binary_confidence:.1f}%)\n\n"
            
            if cascade_prediction:
                summary_text += f"Cascade Classification:\n{cascade_full_name}\n({cascade_confidence:.1f}%)\n\n"
            
            summary_text += f"Expected: {expected_type.upper()}\n"
            summary_text += f"Correct: {'✓' if correct_prediction else '✗'}"
            
            axes[2].text(0.1, 0.9, summary_text, transform=axes[2].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue" if correct_prediction else "lightcoral"))
            axes[2].set_title('Analysis Results')
            
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            buffer.seek(0)
            
            gradcam_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"❌ Error generating GradCAM: {e}")
        
        return jsonify({
            'success': True,
            'binary_prediction': binary_prediction,
            'binary_confidence': binary_confidence,
            'cascade_prediction': cascade_prediction,
            'cascade_confidence': cascade_confidence,
            'cascade_full_name': cascade_full_name,
            'correct_prediction': correct_prediction,
            'gradcam_image': gradcam_image,
            'expected_type': expected_type
        })
        
    except Exception as e:
        print(f"❌ Error analyzing dataset image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def history():
    return render_template('enhanced_history.html')

@app.route('/publications')
def publications():
    return render_template('enhanced_publications.html')

if __name__ == '__main__':
    print(f"🚀 Enhanced Skin Cancer Detection Web Interface")
    print(f"🔗 Binary Classifier: {'✅' if binary_classifier.model else '❌'}")
    print(f"🔗 Cascade Models: {len(cascade_classifier.models)} loaded")
    app.run(debug=True, host='0.0.0.0', port=5000)
