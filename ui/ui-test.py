from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
import sys
import os
import threading

# Add text-speech directory to path to import tts module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text-speech'))
from tts import speak_text, get_voice_id, find_vb_audio_device


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sign Sync")
        self.setMinimumWidth(350)
        self.setWindowIcon(QIcon("mini_logo.png"))

        # Store current selections
        self.current_voice_index = 0  # Default to "Man" (index 0)
        self.current_speed = "1x"  # Default speed
        self.voice_dropdown = None
        self.speed_dropdown = None
        self.current_voice_id = None  # Will be initialized
        self.cable_in_device_index = None  # Will be initialized if available
        self.use_cable_in_for_sample = False  # Track if voice sample should use CABLE In

        self.init_ui()
        
        # Initialize TTS after UI is set up
        self.initialize_tts()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        voice_layout, self.voice_dropdown = self.create_dropdown("Voice:", ["Man", "Woman"], self.on_voice_changed)
        layout.addLayout(voice_layout)
        
        speed_layout, self.speed_dropdown = self.create_dropdown("Speed:", ["0.5x", "1x", "1.5x", "2x"], self.on_speed_changed)
        self.speed_dropdown.setCurrentText("1x")
        layout.addLayout(speed_layout)
                
       # Hear Voice Sample Button
        self.voice_sample_button = QPushButton("Hear Voice Sample")
        self.voice_sample_button.setFixedHeight(30)
        self.voice_sample_button.setFixedWidth(150)
        self.voice_sample_button.setObjectName("voice_sample_button")
        self.voice_sample_button.clicked.connect(self.play_voice_sample)
        self.voice_sample_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # External Play Button (appears in LIVE mode)
        self.external_play_button = QPushButton("External Play")
        self.external_play_button.setFixedHeight(30)
        self.external_play_button.setFixedWidth(150)
        self.external_play_button.setObjectName("external_play_button")
        self.external_play_button.clicked.connect(self.toggle_external_play)
        self.external_play_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.external_play_button.hide()  # Hidden by default

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.voice_sample_button)
        button_layout.addWidget(self.external_play_button)

        layout.addLayout(button_layout)



        nlp_layout, _ = self.create_dropdown("NLP interpreter", ["None", "GPT-5"], self.on_nlp_changed)
        layout.addLayout(nlp_layout)

        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.start_button = QPushButton("START")
        self.start_button.setFixedHeight(50)
        self.start_button.setObjectName("start_button")
        self.start_button.isStart = True
        self.start_button.clicked.connect(self.toggle_start_button)

        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def create_dropdown(self, label_text, items, callback):
        layout = QHBoxLayout()

        label = QLabel(label_text)
        dropdown = QComboBox()
        dropdown.addItems(items)
        dropdown.currentTextChanged.connect(callback)

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

    with open("style.qss", "r") as f:
        app.setStyleSheet(f.read())
    
    window = MainWindow()
    icon = QIcon("mini_logo.ico")
    app.setWindowIcon(icon)
    
    window.setWindowIcon(icon)

    window.show()

    sys.exit(app.exec())
