"""
Analysis Screen - Upload and analyze skin lesion images
"""

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.properties import StringProperty, BooleanProperty
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast
from kivy.clock import Clock
import os

Builder.load_string("""
<AnalysisScreen>:
    name: 'analysis'
    image_path: ''
    analyzing: False
    
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_light
        
        # Top App Bar
        MDTopAppBar:
            title: "Analyze Skin Lesion"
            elevation: 2
            md_bg_color: app.theme_cls.primary_color
            left_action_items: [["arrow-left", lambda x: app.navigate_to('home')]]
            
        # Main Content
        ScrollView:
            MDBoxLayout:
                orientation: 'vertical'
                padding: dp(20)
                spacing: dp(15)
                adaptive_height: True
                
                # Image Upload Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    spacing: dp(15)
                    size_hint_y: None
                    height: dp(400)
                    elevation: 3
                    radius: [15,]
                    
                    MDLabel:
                        text: "Step 1: Upload Image"
                        font_style: 'H6'
                        theme_text_color: "Primary"
                        size_hint_y: None
                        height: dp(40)
                    
                    # Image Preview
                    AsyncImage:
                        id: image_preview
                        source: root.image_path if root.image_path else ''
                        size_hint: 1, None
                        height: dp(250)
                        allow_stretch: True
                        keep_ratio: True
                        
                    MDBoxLayout:
                        orientation: 'horizontal'
                        spacing: dp(10)
                        size_hint_y: None
                        height: dp(56)
                        
                        MDRaisedButton:
                            text: "📷 Select Image"
                            size_hint_x: 0.5
                            md_bg_color: app.theme_cls.primary_color
                            on_release: root.open_file_manager()
                        
                        MDRaisedButton:
                            text: "📸 Take Photo"
                            size_hint_x: 0.5
                            md_bg_color: (0.3, 0.7, 0.5, 1)
                            on_release: root.take_photo()
                
                # Options Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    spacing: dp(10)
                    size_hint_y: None
                    height: dp(150)
                    elevation: 2
                    radius: [15,]
                    
                    MDLabel:
                        text: "Step 2: Analysis Options"
                        font_style: 'H6'
                        theme_text_color: "Primary"
                        size_hint_y: None
                        height: dp(40)
                    
                    MDBoxLayout:
                        orientation: 'horizontal'
                        spacing: dp(10)
                        
                        MDCheckbox:
                            id: cascade_checkbox
                            active: True
                            size_hint: None, None
                            size: dp(48), dp(48)
                        
                        MDLabel:
                            text: "Enable Cascade Classification\\n(Detailed sub-type analysis for benign lesions)"
                            font_style: 'Body2'
                            theme_text_color: "Secondary"
                
                # Analyze Button
                MDRaisedButton:
                    text: "🔬 Analyze Image" if not root.analyzing else "⏳ Analyzing..."
                    pos_hint: {'center_x': 0.5}
                    size_hint_x: 0.9
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: app.theme_cls.primary_color
                    disabled: not root.image_path or root.analyzing
                    on_release: root.start_analysis()
                
                # Results Card (shown after analysis)
                MDCard:
                    id: results_card
                    orientation: 'vertical'
                    padding: dp(20)
                    spacing: dp(10)
                    size_hint_y: None
                    height: 0
                    elevation: 3
                    radius: [15,]
                    opacity: 0
                    
                    MDLabel:
                        id: result_title
                        text: "Analysis Results"
                        font_style: 'H6'
                        theme_text_color: "Primary"
                    
                    MDSeparator:
                    
                    MDLabel:
                        id: result_text
                        text: ""
                        font_style: 'Body1'
                        theme_text_color: "Secondary"
                        markup: True
                
                # Guidelines
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    spacing: dp(10)
                    size_hint_y: None
                    height: dp(220)
                    elevation: 2
                    radius: [15,]
                    
                    MDLabel:
                        text: "📋 Image Guidelines"
                        font_style: 'Subtitle1'
                        bold: True
                        theme_text_color: "Primary"
                    
                    MDLabel:
                        text: "✓ Clear, well-lit photograph\\n✓ Lesion should be in focus\\n✓ Recommended: 600x600 pixels or larger\\n✓ Supported: JPG, PNG formats"
                        font_style: 'Body2'
                        theme_text_color: "Secondary"
                        markup: True
""")

class AnalysisScreen(Screen):
    """Screen for uploading and analyzing skin lesion images."""
    
    image_path = StringProperty('')
    analyzing = BooleanProperty(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = None
    
    def open_file_manager(self):
        """Open file manager to select image."""
        if not self.file_manager:
            self.file_manager = MDFileManager(
                exit_manager=self.exit_file_manager,
                select_path=self.select_image_path,
            )
        
        # Open in pictures directory
        if os.path.exists('/sdcard/Pictures'):
            self.file_manager.show('/sdcard/Pictures')
        elif os.path.exists(os.path.expanduser('~/Pictures')):
            self.file_manager.show(os.path.expanduser('~/Pictures'))
        else:
            self.file_manager.show('/')
    
    def select_image_path(self, path):
        """Handle selected image path."""
        self.image_path = path
        self.exit_file_manager()
        toast(f"Selected: {os.path.basename(path)}")
    
    def exit_file_manager(self, *args):
        """Close file manager."""
        if self.file_manager:
            self.file_manager.close()
    
    def take_photo(self):
        """Take photo using camera (mobile only)."""
        toast("Camera feature coming soon!")
        # TODO: Implement camera capture for Android/iOS
    
    def start_analysis(self):
        """Start image analysis."""
        if not self.image_path:
            toast("Please select an image first")
            return
        
        self.analyzing = True
        Clock.schedule_once(self.perform_analysis, 0.5)
    
    def perform_analysis(self, dt):
        """Perform the actual analysis."""
        try:
            app = self.manager.parent.parent  # Get app instance
            use_cascade = self.ids.cascade_checkbox.active
            
            # Analyze image
            result = app.analyze_image(self.image_path, use_cascade)
            
            if result and 'error' not in result:
                self.show_results(result)
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Analysis failed'
                toast(f"Error: {error_msg}")
            
        except Exception as e:
            toast(f"Error: {str(e)}")
        finally:
            self.analyzing = False
    
    def show_results(self, result):
        """Display analysis results."""
        # Show results card
        self.ids.results_card.height = dp(300) if not result['cascade_results'] else dp(500)
        self.ids.results_card.opacity = 1
        
        # Format results
        prediction = result['prediction']
        confidence = result['confidence']
        
        result_text = f"[b]Prediction:[/b] {prediction}\\n"
        result_text += f"[b]Confidence:[/b] {confidence:.2f}%\\n\\n"
        result_text += f"[b]Benign:[/b] {result['benign_prob']:.2f}%\\n"
        result_text += f"[b]Malignant:[/b] {result['malignant_prob']:.2f}%\\n"
        
        if result['cascade_results']:
            result_text += "\\n[b]Cascade Analysis:[/b]\\n"
            for cascade in result['cascade_results']:
                status = "✅" if cascade['positive'] else "❌"
                result_text += f"{status} {cascade['type']}: {cascade['confidence']:.2f}%\\n"
        
        self.ids.result_text.text = result_text
        
        # Show appropriate message
        if result['is_malignant']:
            self.ids.result_title.text = "⚠️ URGENT - Immediate Action Required"
            self.ids.result_title.theme_text_color = "Error"
        else:
            self.ids.result_title.text = "✅ Analysis Complete"
            self.ids.result_title.theme_text_color = "Custom"
        
        toast("Analysis complete!")
