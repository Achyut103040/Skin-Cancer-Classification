"""
History Screen - View past analysis results
"""

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

Builder.load_string("""
<HistoryScreen>:
    name: 'history'
    
    MDBoxLayout:
        orientation: 'vertical'
        md_bg_color: app.theme_cls.bg_light
        
        MDTopAppBar:
            title: "Analysis History"
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
                    padding: dp(30)
                    spacing: dp(20)
                    size_hint_y: None
                    height: dp(300)
                    elevation: 2
                    radius: [15,]
                    
                    MDIcon:
                        icon: 'history'
                        font_size: '72sp'
                        halign: 'center'
                        theme_text_color: "Custom"
                        text_color: app.theme_cls.primary_color
                    
                    MDLabel:
                        text: "History Feature"
                        font_style: 'H5'
                        halign: 'center'
                        theme_text_color: "Primary"
                    
                    MDLabel:
                        text: "View your past analysis results here.\\n\\nComing soon in the next update!"
                        font_style: 'Body1'
                        halign: 'center'
                        theme_text_color: "Secondary"
""")

class HistoryScreen(Screen):
    pass
