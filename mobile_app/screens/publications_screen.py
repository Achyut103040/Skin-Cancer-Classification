"""
Publications Screen - Research papers and citations
"""

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

Builder.load_string("""
<PublicationsScreen>:
    name: 'publications'
    
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_light
        
        MDTopAppBar:
            title: "Publications & Research"
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
                    height: dp(800)
                    elevation: 2
                    radius: [15,]
                    
                    MDLabel:
                        text: "📄 Research Publications"
                        font_style: 'H5'
                        theme_text_color: "Primary"
                        size_hint_y: None
                        height: dp(40)
                    
                    MDSeparator:
                    
                    MDLabel:
                        text: "[b]MsBiCNet: Multi-stage Binary Cascade Network for Skin Cancer Detection[/b]\\n\\nAuthors: [Your Team]\\nJournal: [Journal Name]\\nYear: 2024\\n\\n[b]Abstract:[/b]\\nThis research presents MsBiCNet, a novel two-stage deep learning architecture for automated skin cancer detection. The system achieves 96.1% accuracy on the HAM10000 dataset using a combination of binary classification and specialized cascade models.\\n\\n[b]Key Contributions:[/b]\\n• Novel two-stage architecture\\n• 96.1% classification accuracy\\n• Real-time mobile deployment\\n• Comprehensive cascade analysis\\n\\n[b]Dataset:[/b]\\nHAM10000 - 10,015 dermatoscopic images across 7 skin lesion categories.\\n\\n[b]Citation:[/b]\\n[Your Citation Format Here]"
                        font_style: 'Body1'
                        theme_text_color: "Secondary"
                        markup: True
                        size_hint_y: None
                        height: dp(680)
""")

class PublicationsScreen(Screen):
    pass
