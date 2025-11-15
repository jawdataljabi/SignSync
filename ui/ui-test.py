from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QIcon, QFontMetrics, QFont
from PyQt6.QtCore import Qt, QPoint, QRect
import sys
import os
import threading
from qt_material import apply_stylesheet

if sys.platform == "win32":
    try:
        import ctypes
        myappid = 'SignSync.Application.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        print(f"Warning: Could not set AppUserModelID: {e}")

# Add text-speech directory to path to import tts module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text-speech'))
from tts import speak_text, get_voice_id, find_vb_audio_device
from openai_client import get_client


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set frameless window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        self.setWindowTitle("Sign Sync")
        self.drag_position = QPoint()
        self.resize_border_width = 8  # Width of the resize border
        self.is_resizing = False
        self.resize_edge = None  # 'top', 'bottom', 'left', 'right', 'topLeft', 'topRight', 'bottomLeft', 'bottomRight'
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "mini_logo.ico")
        self.setWindowIcon(QIcon(icon_path))
        
        # Calculate dynamic minimum width and height based on font metrics
        font_metrics = QFontMetrics(self.font())
        min_width = max(
            font_metrics.horizontalAdvance("NLP interpreter") + font_metrics.horizontalAdvance("gpt-4o-mini") + 150,
            font_metrics.horizontalAdvance("Hear Voice Sample") + font_metrics.horizontalAdvance("External Play: ON") + 150,
            400  # Absolute minimum
        )
        # Calculate minimum height based on content
        min_height = max(
            int(font_metrics.height() * 15),  # Rough estimate for all elements
            250  # Absolute minimum
        )
        self.setMinimumWidth(int(min_width))
        self.setMinimumHeight(min_height)
        
        # Allow the window to resize both horizontally and vertically
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Store base font size for dynamic scaling
        self.base_font_size = self.font().pointSize() if self.font().pointSize() > 0 else 10
        self.base_width = 400  # Reference width for font scaling
        self.base_height = 500  # Reference height for font scaling
        self.main_layout = None  # Will be set in init_ui

        # Store current selections
        self.current_voice_index = 0  # Default to "Man" (index 0)
        self.current_speed = "1x"  # Default speed
        self.current_nlp_model = "gpt-4o-mini"  # Default NLP model
        self.voice_dropdown = None
        self.speed_dropdown = None
        self.nlp_dropdown = None
        self.current_voice_id = None  # Will be initialized
        self.cable_in_device_index = None  # Will be initialized if available
        self.use_cable_in_for_sample = False  # Track if voice sample should use CABLE In

        self.init_ui()
        
        # Initialize TTS after UI is set up
        self.initialize_tts()
        
        # Initialize NLP after UI is set up
        self.initialize_nlp()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Top bar with logo, title, and close button
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(10, 5, 0, 5)
        top_bar.setSpacing(10)
        
        # Logo and title on the left
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "mini_logo.ico")
        if os.path.exists(icon_path):
            logo_label = QLabel()
            logo_label.setPixmap(QIcon(icon_path).pixmap(24, 24))
            top_bar.addWidget(logo_label)
        
        title_label = QLabel("Sign Sync")
        title_label.setObjectName("title_label")
        top_bar.addWidget(title_label)
        
        top_bar.addStretch()
        
        # Close button
        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(26, 26)
        self.close_button.setObjectName("close_button")
        self.close_button.clicked.connect(self.close)
        # Force red color directly on button
        self.close_button.setStyleSheet("""
            QPushButton#close_button {
                background-color: transparent;
                border: none;
                color: #e74c3c;
                font-size: 21px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton#close_button:hover {
                background-color: #c0392b;
                color: #ffffff;
            }
        """)
        top_bar.addWidget(self.close_button)
        
        layout.addLayout(top_bar)
        
        # Main content layout
        content_layout = QVBoxLayout()
        self.main_layout = content_layout  # Store reference for dynamic updates
        # Calculate font metrics once for all dynamic sizing
        font_metrics = QFontMetrics(self.font())
        
        # Dynamic margins and spacing based on font size
        self.base_margin = max(15, int(font_metrics.height() * 0.8))
        self.base_spacing = max(10, int(font_metrics.height() * 0.6))
        content_layout.setContentsMargins(self.base_margin, self.base_margin, self.base_margin, self.base_margin)
        content_layout.setSpacing(self.base_spacing)

        voice_layout, self.voice_dropdown = self.create_dropdown("Voice:", ["Man", "Woman"], self.on_voice_changed)
        content_layout.addLayout(voice_layout)
        
        speed_layout, self.speed_dropdown = self.create_dropdown("Speed:", ["0.5x", "1x", "1.5x", "2x"], self.on_speed_changed)
        self.speed_dropdown.setCurrentText("1x")
        content_layout.addLayout(speed_layout)
                
       # Hear Voice Sample Button
        self.voice_sample_button = QPushButton("Hear Voice Sample")
        button_height = max(30, int(font_metrics.height() * 1.8))
        # Calculate width based on text content with padding
        button_width = max(120, int(font_metrics.horizontalAdvance("Hear Voice Sample") * 1.3))
        self.voice_sample_button.setMinimumHeight(button_height)
        self.voice_sample_button.setMinimumWidth(button_width)
        self.voice_sample_button.setObjectName("voice_sample_button")
        self.voice_sample_button.clicked.connect(self.play_voice_sample)
        # Allow vertical expansion when window is resized
        self.voice_sample_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)

        # External Play Button (appears in LIVE mode)
        self.external_play_button = QPushButton("External Play")
        # Use same height as voice sample button for consistency
        external_button_width = max(120, int(font_metrics.horizontalAdvance("External Play: ON") * 1.3))
        self.external_play_button.setMinimumHeight(button_height)
        self.external_play_button.setMinimumWidth(external_button_width)
        self.external_play_button.setObjectName("external_play_button")
        self.external_play_button.clicked.connect(self.toggle_external_play)
        # Allow vertical expansion when window is resized
        self.external_play_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        self.external_play_button.hide()  # Hidden by default

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.voice_sample_button)
        button_layout.addWidget(self.external_play_button)

        content_layout.addLayout(button_layout)



        nlp_layout, self.nlp_dropdown = self.create_dropdown("NLP interpreter", ["None", "gpt-3.5-turbo", "gpt-4o-mini"], self.on_nlp_changed)
        self.nlp_dropdown.setCurrentText("gpt-4o-mini")  # Default to gpt-4o-mini
        content_layout.addLayout(nlp_layout)

        # Dynamic spacer based on font size
        spacer_height = max(20, int(font_metrics.height() * 2))
        content_layout.addItem(QSpacerItem(int(font_metrics.height()), spacer_height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.start_button = QPushButton("START")
        # Dynamic button height based on font size
        start_button_height = max(40, int(font_metrics.height() * 2.5))
        self.start_button.setMinimumHeight(start_button_height)
        self.start_button.setObjectName("start_button")
        self.start_button.isStart = True
        self.start_button.clicked.connect(self.toggle_start_button)
        # Allow vertical expansion when window is resized
        self.start_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        content_layout.addWidget(self.start_button)
        
        # Add content layout to main layout
        layout.addLayout(content_layout)

        self.setLayout(layout)

    def showEvent(self, event):
        """Capture initial window size when first shown."""
        super().showEvent(event)
        if not hasattr(self, '_initial_size_set'):
            # Update base dimensions with actual window size
            self.base_width = max(self.width(), self.base_width)
            self.base_height = max(self.height(), self.base_height)
            self._initial_size_set = True
    
    def resizeEvent(self, event):
        """Handle window resize to scale fonts dynamically."""
        super().resizeEvent(event)
        # Only scale if we have initial size set
        if hasattr(self, '_initial_size_set') and self._initial_size_set:
            self.update_font_sizes()
    
    def update_font_sizes(self):
        """Update font sizes, margins, and spacing for all widgets based on window size."""
        if not hasattr(self, 'base_font_size') or not hasattr(self, 'base_width'):
            return
            
        # Calculate scale factor based on window size
        # Use geometric mean of width and height for more balanced scaling
        current_width = self.width()
        current_height = self.height()
        
        if current_width <= 0 or current_height <= 0:
            return
        
        # Calculate scale factors
        width_scale = current_width / self.base_width if self.base_width > 0 else 1.0
        height_scale = current_height / self.base_height if self.base_height > 0 else 1.0
        
        # Use geometric mean for more balanced scaling that matches button expansion
        scale_factor = (width_scale * height_scale) ** 0.5
        
        # Allow more aggressive scaling to match button expansion
        scale_factor = min(scale_factor, 3.0)  # Cap at 3x to prevent too large fonts
        scale_factor = max(scale_factor, 0.6)  # Minimum 0.6x to prevent too small fonts
        
        # Calculate new font size - scale more aggressively
        new_font_size = int(self.base_font_size * scale_factor)
        new_font_size = max(8, min(new_font_size, 72))  # Clamp between 8 and 72
        
        # Create new font with scaled size
        new_font = QFont(self.font())
        new_font.setPointSize(new_font_size)
        
        # Apply font to all widgets recursively
        self.apply_font_to_widget(self, new_font)
        
        # Update margins and spacing proportionally
        if hasattr(self, 'main_layout') and self.main_layout:
            new_margin = int(self.base_margin * scale_factor)
            new_spacing = int(self.base_spacing * scale_factor)
            self.main_layout.setContentsMargins(new_margin, new_margin, new_margin, new_margin)
            self.main_layout.setSpacing(new_spacing)
    
    def apply_font_to_widget(self, widget, font):
        """Recursively apply font to widget and all its children."""
        widget.setFont(font)
        for child in widget.findChildren(QWidget):
            child.setFont(font)
    
    def get_resize_edge(self, pos):
        """Determine which edge the mouse is near."""
        x, y = pos.x(), pos.y()
        width, height = self.width(), self.height()
        border = self.resize_border_width
        
        # Check corners first
        if x < border and y < border:
            return 'topLeft'
        elif x >= width - border and y < border:
            return 'topRight'
        elif x < border and y >= height - border:
            return 'bottomLeft'
        elif x >= width - border and y >= height - border:
            return 'bottomRight'
        # Check edges
        elif x < border:
            return 'left'
        elif x >= width - border:
            return 'right'
        elif y < border:
            return 'top'
        elif y >= height - border:
            return 'bottom'
        return None
    
    def get_cursor_for_edge(self, edge):
        """Get the appropriate cursor for the resize edge."""
        cursors = {
            'top': Qt.CursorShape.SizeVerCursor,
            'bottom': Qt.CursorShape.SizeVerCursor,
            'left': Qt.CursorShape.SizeHorCursor,
            'right': Qt.CursorShape.SizeHorCursor,
            'topLeft': Qt.CursorShape.SizeFDiagCursor,
            'topRight': Qt.CursorShape.SizeBDiagCursor,
            'bottomLeft': Qt.CursorShape.SizeBDiagCursor,
            'bottomRight': Qt.CursorShape.SizeFDiagCursor,
        }
        return cursors.get(edge, Qt.CursorShape.ArrowCursor)
    
    def mousePressEvent(self, event):
        """Handle mouse press for window dragging and resizing."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            edge = self.get_resize_edge(pos)
            
            if edge:
                self.is_resizing = True
                self.resize_edge = edge
                self.drag_position = event.globalPosition().toPoint()
                self.resize_start_geometry = self.geometry()
            else:
                self.is_resizing = False
                self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging and resizing."""
        pos = event.position().toPoint()
        
        if not self.is_resizing:
            # Update cursor based on edge proximity
            edge = self.get_resize_edge(pos)
            if edge:
                self.setCursor(self.get_cursor_for_edge(edge))
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            
            # Handle dragging
            if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position:
                self.move(event.globalPosition().toPoint() - self.drag_position)
        else:
            # Handle resizing
            if event.buttons() == Qt.MouseButton.LeftButton and self.resize_edge:
                delta = event.globalPosition().toPoint() - self.drag_position
                geom = self.resize_start_geometry
                
                if self.resize_edge == 'left':
                    new_width = max(self.minimumWidth(), geom.width() - delta.x())
                    new_x = geom.x() + (geom.width() - new_width)
                    self.setGeometry(new_x, geom.y(), new_width, geom.height())
                elif self.resize_edge == 'right':
                    new_width = max(self.minimumWidth(), geom.width() + delta.x())
                    self.setGeometry(geom.x(), geom.y(), new_width, geom.height())
                elif self.resize_edge == 'top':
                    new_height = max(self.minimumHeight(), geom.height() - delta.y())
                    new_y = geom.y() + (geom.height() - new_height)
                    self.setGeometry(geom.x(), new_y, geom.width(), new_height)
                elif self.resize_edge == 'bottom':
                    new_height = max(self.minimumHeight(), geom.height() + delta.y())
                    self.setGeometry(geom.x(), geom.y(), geom.width(), new_height)
                elif self.resize_edge == 'topLeft':
                    new_width = max(self.minimumWidth(), geom.width() - delta.x())
                    new_height = max(self.minimumHeight(), geom.height() - delta.y())
                    new_x = geom.x() + (geom.width() - new_width)
                    new_y = geom.y() + (geom.height() - new_height)
                    self.setGeometry(new_x, new_y, new_width, new_height)
                elif self.resize_edge == 'topRight':
                    new_width = max(self.minimumWidth(), geom.width() + delta.x())
                    new_height = max(self.minimumHeight(), geom.height() - delta.y())
                    new_y = geom.y() + (geom.height() - new_height)
                    self.setGeometry(geom.x(), new_y, new_width, new_height)
                elif self.resize_edge == 'bottomLeft':
                    new_width = max(self.minimumWidth(), geom.width() - delta.x())
                    new_height = max(self.minimumHeight(), geom.height() + delta.y())
                    new_x = geom.x() + (geom.width() - new_width)
                    self.setGeometry(new_x, geom.y(), new_width, new_height)
                elif self.resize_edge == 'bottomRight':
                    new_width = max(self.minimumWidth(), geom.width() + delta.x())
                    new_height = max(self.minimumHeight(), geom.height() + delta.y())
                    self.setGeometry(geom.x(), geom.y(), new_width, new_height)
        
        event.accept()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop resizing/dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_resizing = False
            self.resize_edge = None
            self.drag_position = QPoint()
            self.setCursor(Qt.CursorShape.ArrowCursor)
        event.accept()

    def create_dropdown(self, label_text, items, callback):
        layout = QHBoxLayout()

        label = QLabel(label_text)
        label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        dropdown = QComboBox()
        dropdown.addItems(items)
        dropdown.currentTextChanged.connect(callback)
        dropdown.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(label)
        layout.addWidget(dropdown)

        return layout, dropdown

    def on_voice_changed(self, value):
        print("Selected voice:", value)
        # Map "Man" -> 0, "Woman" -> 1
        if value == "Man":
            self.current_voice_index = 0
        elif value == "Woman":
            self.current_voice_index = 1
        # Update voice ID when voice changes
        self.update_voice_id()

    def on_speed_changed(self, value):
        print("Selected speed:", value)
        self.current_speed = value

    def on_nlp_changed(self, value):
        print("Selected NLP model:", value)
        # Map dropdown value to model name (or None if "None" is selected)
        if value == "None":
            self.current_nlp_model = None
        else:
            self.current_nlp_model = value
        print(f"Current NLP model set to: {self.current_nlp_model}")
    
    def get_nlp_model(self):
        """Get the current NLP model, defaulting to 'gpt-4o-mini' if None."""
        if self.current_nlp_model is None:
            return "gpt-4o-mini"
        return self.current_nlp_model

    def initialize_tts(self):
        """Initialize TTS by getting the voice ID for the current voice selection."""
        print("Initializing TTS...")
        self.update_voice_id()
        
        # Try to find CABLE In device
        try:
            self.cable_in_device_index = find_vb_audio_device()
            print(f"CABLE In device found at index {self.cable_in_device_index}")
        except Exception as e:
            print(f"CABLE In device not available: {e}")
            self.cable_in_device_index = None
        
        print("TTS initialized")
    
    def initialize_nlp(self):
        """Initialize NLP by checking OpenAI API key and creating client."""
        print("Initializing NLP...")
        try:
            # Try to get the client (will check for API key)
            client = get_client()
            print("NLP initialized successfully - OpenAI API key found")
        except ValueError as e:
            print(f"NLP initialization warning: {e}")
            print("NLP features will not be available until OPENAI_API_KEY is set")
        except Exception as e:
            print(f"NLP initialization error: {e}")
        print("NLP initialization complete")

    def update_voice_id(self):
        """Update the voice ID based on the current voice index."""
        try:
            self.current_voice_id = get_voice_id(self.current_voice_index)
            print(f"Voice ID updated for voice index {self.current_voice_index}")
        except Exception as e:
            print(f"Error getting voice ID: {e}")
            # Fallback to default voice (index 0)
            try:
                self.current_voice_id = get_voice_id(0)
            except:
                self.current_voice_id = None

    def play_voice_sample(self):
        """Play a voice sample using the currently selected voice and speed."""
        print("Playing voice sample...")
        
        # Get current selections from dropdowns if available
        if self.voice_dropdown:
            voice_text = self.voice_dropdown.currentText()
            # Map "Man" -> 0, "Woman" -> 1
            if voice_text == "Man":
                new_voice_index = 0
            elif voice_text == "Woman":
                new_voice_index = 1
            else:
                new_voice_index = 0  # Default to Man
            
            if new_voice_index != self.current_voice_index:
                self.current_voice_index = new_voice_index
                self.update_voice_id()
        
        if self.speed_dropdown:
            self.current_speed = self.speed_dropdown.currentText()
        
        # Map speed multiplier to rate (words per minute)
        # Default rate is 160 WPM
        speed_multiplier = float(self.current_speed.replace('x', ''))
        rate = int(160 * speed_multiplier)
        
        # Use the pre-initialized voice ID
        voice_id = self.current_voice_id
        if voice_id is None:
            print("Warning: Voice ID not initialized, using default")
            try:
                voice_id = get_voice_id(0)  # Fallback to first voice
            except:
                print("Error: Could not get voice ID")
                return
        
        # Speak the sample text in a separate thread to avoid event loop conflicts
        sample_text = "this is a test voice sample"
        
        # Determine which device to use
        device_index = None
        if self.use_cable_in_for_sample and self.cable_in_device_index is not None:
            device_index = self.cable_in_device_index
            print("Playing voice sample through CABLE In")
        else:
            print("Playing voice sample through system default")
        
        def speak_in_thread():
            try:
                speak_text(sample_text, rate=rate, voice_id=voice_id, sapi_device_index=device_index)
            except Exception as e:
                print(f"Error speaking: {e}")
        
        thread = threading.Thread(target=speak_in_thread, daemon=True)
        thread.start()


    def get_audio_device_index(self):
        """Get the audio device index to use based on current mode.
        
        Returns:
            Device index if in LIVE mode and CABLE In is available, None for system default
        """
        # If start button is in LIVE mode and CABLE In is available, use it
        if not self.start_button.isStart and self.cable_in_device_index is not None:
            return self.cable_in_device_index
        # Otherwise use system default (None)
        return None

    def toggle_external_play(self):
        """Toggle whether voice sample should play through CABLE In or system default."""
        self.use_cable_in_for_sample = not self.use_cable_in_for_sample
        
        if self.use_cable_in_for_sample:
            self.external_play_button.setText("External Play: ON")
            print("Voice sample set to play through CABLE In")
        else:
            self.external_play_button.setText("External Play: OFF")
            print("Voice sample set to play through system default")

    def toggle_start_button(self):
        print("toggling start button")
        if self.start_button.isStart:
            self.start_button.setObjectName("start_button_live")
            self.start_button.setText("LIVE")
            if self.cable_in_device_index is not None:
                print(f"Switched to LIVE mode - will use CABLE In device (index {self.cable_in_device_index})")
            else:
                print("Switched to LIVE mode - CABLE In not available, using system default")
            
            # Show external play button
            self.external_play_button.show()
            self.external_play_button.setText("External Play: OFF")
            self.use_cable_in_for_sample = False
            
            # Speak "SignSync initialized" on CABLE In
            if self.cable_in_device_index is not None and self.current_voice_id is not None:
                # Map speed multiplier to rate (words per minute)
                speed_multiplier = float(self.current_speed.replace('x', ''))
                rate = int(160 * speed_multiplier)
                
                def speak_initialized():
                    try:
                        speak_text("SignSync initialized", rate=rate, voice_id=self.current_voice_id, 
                                 sapi_device_index=self.cable_in_device_index)
                    except Exception as e:
                        print(f"Error speaking 'SignSync initialized': {e}")
                
                thread = threading.Thread(target=speak_initialized, daemon=True)
                thread.start()
        else:
            self.start_button.setObjectName("start_button")
            self.start_button.setText("START")
            print("Switched to START mode - using system default audio")
            
            # Hide external play button
            self.external_play_button.hide()
            self.use_cable_in_for_sample = False
            
            # Speak "SignSync off" on CABLE In
            if self.cable_in_device_index is not None and self.current_voice_id is not None:
                # Map speed multiplier to rate (words per minute)
                speed_multiplier = float(self.current_speed.replace('x', ''))
                rate = int(160 * speed_multiplier)
                
                def speak_off():
                    try:
                        speak_text("SignSync off", rate=rate, voice_id=self.current_voice_id, 
                                 sapi_device_index=self.cable_in_device_index)
                    except Exception as e:
                        print(f"Error speaking 'SignSync off': {e}")
                
                thread = threading.Thread(target=speak_off, daemon=True)
                thread.start()

        self.start_button.isStart = not self.start_button.isStart

        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(script_dir, "mini_logo.ico")
    
    if not os.path.exists(icon_path):
        print(f"Warning: Icon file not found at {icon_path}")
    else:
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)

    # Apply qt_material dark_red theme
    apply_stylesheet(app, theme="dark_red.xml")
    
    window = MainWindow()
    
    # Override background to jet black and set soft grey text
    window.setStyleSheet("""
        QWidget {
            background-color: #000000;
            color: #C0C0C0;
        }
        QLabel {
            color: #C0C0C0;
        }
        #close_button {
            background-color: transparent;
            border: none;
            color: #e74c3c;
            font-size: 24px;
            font-weight: bold;
        }
        #close_button:hover {
            background-color: #c0392b;
            color: #ffffff;
        }
        #title_label {
            color: #C0C0C0;
            font-size: 16px;
            font-weight: 600;
        }
    """)
    
    # Also set icon on window (redundant but ensures it's set)
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))

    window.show()

    sys.exit(app.exec())
