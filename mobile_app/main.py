"""
🔬 MsBiCNet Mobile App - Skin Cancer Detection
Multi-platform mobile application (Android, iOS, Desktop)
Built with Kivy + KivyMD
"""

import os
os.environ['KIVY_NO_CONSOLELOG'] = '1'

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.utils import platform
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.list import OneLineAvatarIconListItem, IconLeftWidget
from kivymd.uix.filemanager import MDFileManager
from kivymd.toast import toast

# Import screens
from screens.home_screen import HomeScreen
from screens.analysis_screen import AnalysisScreen
from screens.history_screen import HistoryScreen
from screens.about_screen import AboutScreen
from screens.publications_screen import PublicationsScreen

# Import AI models
from models.model_manager import ModelManager

# Set window size for desktop testing
if platform not in ('android', 'ios'):
    Window.size = (400, 700)
    Window.top = 50
    Window.left = 100

class SkinCancerApp(MDApp):
    """Main application class for MsBiCNet mobile app."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "MsBiCNet - Skin Cancer Detection"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.primary_hue = "700"
        self.theme_cls.theme_style = "Light"
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # File manager for image selection
        self.file_manager = None
        
        # Current analysis results
        self.current_result = None
        
    def build(self):
        """Build the application UI."""
        # Create screen manager
        self.sm = ScreenManager(transition=FadeTransition())
        
        # Add all screens
        self.sm.add_widget(HomeScreen(name='home'))
        self.sm.add_widget(AnalysisScreen(name='analysis'))
        self.sm.add_widget(HistoryScreen(name='history'))
        self.sm.add_widget(AboutScreen(name='about'))
        self.sm.add_widget(PublicationsScreen(name='publications'))
        
        # Show loading screen
        Clock.schedule_once(self.load_models, 1)
        
        return self.sm
    
    def load_models(self, dt):
        """Load AI models in background."""
        try:
            toast("Loading AI models... Please wait")
            # Load models (this will download from Google Drive if needed)
            success = self.model_manager.load_all_models()
            
            if success:
                toast("✅ All 6 models loaded successfully!")
            else:
                self.show_error_dialog("Failed to load models. Please check your internet connection.")
        except Exception as e:
            self.show_error_dialog(f"Error loading models: {str(e)}")
    
    def show_error_dialog(self, message):
        """Show error dialog."""
        dialog = MDDialog(
            title="Error",
            text=message,
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()
    
    def show_info_dialog(self, title, message):
        """Show information dialog."""
        dialog = MDDialog(
            title=title,
            text=message,
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()
    
    def navigate_to(self, screen_name):
        """Navigate to specified screen."""
        self.sm.current = screen_name
    
    def analyze_image(self, image_path, use_cascade=True):
        """Analyze skin lesion image."""
        try:
            if not self.model_manager.models_loaded:
                self.show_error_dialog("Models not loaded yet. Please wait...")
                return None
            
            # Perform analysis
            result = self.model_manager.analyze_image(image_path, use_cascade)
            self.current_result = result
            
            return result
            
        except Exception as e:
            self.show_error_dialog(f"Analysis error: {str(e)}")
            return None
    
    def on_pause(self):
        """Handle app pause (mobile)."""
        return True
    
    def on_resume(self):
        """Handle app resume (mobile)."""
        pass

if __name__ == '__main__':
    SkinCancerApp().run()
