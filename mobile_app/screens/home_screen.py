"""
Home Screen - Main navigation hub
"""

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivymd.uix.card import MDCard

Builder.load_string("""
<HomeScreen>:
    name: 'home'
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_light
        
        # Top App Bar
        MDTopAppBar:
            title: "MsBiCNet - Skin Cancer Detection"
            elevation: 2
            md_bg_color: app.theme_cls.primary_color
            
        # Main Content
        ScrollView:
            MDBoxLayout:
                orientation: 'vertical'
                padding: dp(20)
                spacing: dp(15)
                adaptive_height: True
                
                # Welcome Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    spacing: dp(10)
                    size_hint_y: None
                    height: dp(200)
                    elevation: 3
                    radius: [15,]
                    
                    MDIcon:
                        icon: 'hospital-box'
                        font_size: '64sp'
                        halign: 'center'
                        theme_text_color: "Custom"
                        text_color: app.theme_cls.primary_color
                    
                    MDLabel:
                        text: "Welcome to MsBiCNet"
                        font_style: 'H6'
                        halign: 'center'
                        theme_text_color: "Primary"
                    
                    MDLabel:
                        text: "AI-Powered Skin Cancer Detection"
                        font_style: 'Body2'
                        halign: 'center'
                        theme_text_color: "Secondary"
                
                # Quick Actions
                MDLabel:
                    text: "Quick Actions"
                    font_style: 'H6'
                    theme_text_color: "Primary"
                    size_hint_y: None
                    height: dp(40)
                
                # Analyze Button
                MDRaisedButton:
                    text: "🔬 Analyze Skin Lesion"
                    pos_hint: {'center_x': 0.5}
                    size_hint_x: 0.9
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: app.theme_cls.primary_color
                    on_release: app.navigate_to('analysis')
                
                # History Button
                MDRaisedButton:
                    text: "📊 View History"
                    pos_hint: {'center_x': 0.5}
                    size_hint_x: 0.9
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: app.theme_cls.accent_color
                    on_release: app.navigate_to('history')
                
                # Information Section
                MDLabel:
                    text: "Information"
                    font_style: 'H6'
                    theme_text_color: "Primary"
                    size_hint_y: None
                    height: dp(40)
                
                # About Button
                MDRaisedButton:
                    text: "ℹ️ About MsBiCNet"
                    pos_hint: {'center_x': 0.5}
                    size_hint_x: 0.9
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: (0.3, 0.6, 0.9, 1)
                    on_release: app.navigate_to('about')
                
                # Publications Button
                MDRaisedButton:
                    text: "📄 Publications & Research"
                    pos_hint: {'center_x': 0.5}
                    size_hint_x: 0.9
                    size_hint_y: None
                    height: dp(56)
                    md_bg_color: (0.5, 0.7, 0.3, 1)
                    on_release: app.navigate_to('publications')
                
                # Stats Card
                MDCard:
                    orientation: 'vertical'
                    padding: dp(20)
                    spacing: dp(10)
                    size_hint_y: None
                    height: dp(180)
                    elevation: 2
                    radius: [15,]
                    
                    MDLabel:
                        text: "System Statistics"
                        font_style: 'Subtitle1'
                        bold: True
                        theme_text_color: "Primary"
                    
                    MDBoxLayout:
                        orientation: 'horizontal'
                        spacing: dp(10)
                        size_hint_y: None
                        height: dp(40)
                        
                        MDIcon:
                            icon: 'brain'
                            theme_text_color: "Custom"
                            text_color: app.theme_cls.primary_color
                        
                        MDLabel:
                            text: "6 AI Models Loaded"
                            font_style: 'Body1'
                            theme_text_color: "Secondary"
                    
                    MDBoxLayout:
                        orientation: 'horizontal'
                        spacing: dp(10)
                        size_hint_y: None
                        height: dp(40)
                        
                        MDIcon:
                            icon: 'database'
                            theme_text_color: "Custom"
                            text_color: app.theme_cls.primary_color
                        
                        MDLabel:
                            text: "10,015 Training Images"
                            font_style: 'Body1'
                            theme_text_color: "Secondary"
                    
                    MDBoxLayout:
                        orientation: 'horizontal'
                        spacing: dp(10)
                        size_hint_y: None
                        height: dp(40)
                        
                        MDIcon:
                            icon: 'chart-line'
                            theme_text_color: "Custom"
                            text_color: app.theme_cls.primary_color
                        
                        MDLabel:
                            text: "96.1% Accuracy"
                            font_style: 'Body1'
                            theme_text_color: "Secondary"
                
                # Disclaimer
                MDLabel:
                    text: "⚠️ Medical Disclaimer: This app is for educational purposes only. Always consult medical professionals for diagnosis."
                    font_style: 'Caption'
                    halign: 'center'
                    theme_text_color: "Hint"
                    size_hint_y: None
                    height: dp(60)
                
""")

class HomeScreen(Screen):
    """Home screen with navigation options."""
    pass
