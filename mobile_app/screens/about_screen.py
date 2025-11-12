"""
About Screen - Information about MsBiCNet system
"""

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

Builder.load_string("""
<AboutScreen>:
    name: 'about'
    
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_light
        
        MDTopAppBar:
            title: "About MsBiCNet"
            elevation: 2
            md_bg_color: app.theme_cls.primary_color
            left_action_items: [["arrow-left", lambda x: app.navigate_to('home')]]
        
        ScrollView:
            MDBoxLayout:
                orientation: 'vertical'
                padding: dp(20)
                spacing: dp(15)
                adaptive_height: True
                
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    spacing: dp(15)
                    size_hint_y: None
                    height: dp(1500)
                    elevation: 2
                    radius: [15,]
                    
                    MDLabel:
                        text: "MsBiCNet System"
                        font_style: 'H5'
                        theme_text_color: "Primary"
                        size_hint_y: None
                        height: dp(40)
                    
                    MDSeparator:
                    
                    MDLabel:
                        text: "Multi-stage Binary Cascade Network"
                        font_style: 'H6'
                        theme_text_color: "Primary"
                        size_hint_y: None
                        height: dp(35)
                    
                    MDLabel:
                        text: "MsBiCNet uses 6 specialized deep learning models for comprehensive skin lesion analysis:\\n\\n🔬 Stage 1: Binary Classification\\n• Model: ResNet50 with Deep Classifier\\n• Purpose: Primary classification (Benign vs Malignant)\\n• Accuracy: 96.1% on test set\\n\\n🔍 Stage 2: Cascade Classification (5 Models)\\n\\n1. NV Model (Melanocytic Nevi)\\n   • Architecture: ResNet50\\n   • Detects: Common moles\\n\\n2. BKL Model (Benign Keratosis)\\n   • Architecture: ResNet18\\n   • Detects: Benign keratosis-like lesions\\n\\n3. BCC Model (Basal Cell Carcinoma)\\n   • Architecture: EfficientNet-B0\\n   • Detects: Most common skin cancer\\n\\n4. AKIEC Model (Actinic Keratoses)\\n   • Architecture: ResNet50\\n   • Detects: Actinic keratoses\\n\\n5. VASC Model (Vascular Lesions)\\n   • Architecture: ResNet50\\n   • Detects: Vascular lesions\\n\\n📊 Training Details\\n• Dataset: HAM10000 (10,015 images)\\n• Validation: 5-fold cross-validation\\n• Optimization: Adam optimizer\\n• Augmentation: Rotations, flips, color jittering\\n\\n⚠️ Medical Disclaimer\\nThis tool is for educational and research purposes only. Always consult qualified medical professionals for diagnosis and treatment."
                        font_style: 'Body1'
                        theme_text_color: "Secondary"
                        markup: True
                        size_hint_y: None
                        height: dp(1300)
                    
                    MDLabel:
                        text: "Developed by: AI Research Team\\nFramework: PyTorch + Kivy\\nVersion: 1.0.0"
                        font_style: 'Caption'
                        halign: 'center'
                        theme_text_color: "Hint"
                        size_hint_y: None
                        height: dp(80)
""")

class AboutScreen(Screen):
    pass
