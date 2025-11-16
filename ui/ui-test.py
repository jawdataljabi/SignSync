from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpacerItem, QSizePolicy, QTextEdit, QCheckBox
)
from PyQt6.QtGui import QIcon, QFontMetrics, QFont, QFontDatabase
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QObject
import sys
import os
import threading
import subprocess
from qt_material import apply_stylesheet

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('SignSync.Application.1.0')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text-speech'))
from tts import speak_text, get_voice_id, find_vb_audio_device
from openai_client import get_client, send_prompt_and_speak_streaming


class MainWindow(QWidget):
    # Signal for handling sentences from background thread
    sentence_received = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Sign Sync")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_path = os.path.join(script_dir, "mini_logo.ico")
        self.setWindowIcon(QIcon(self.icon_path))
        
        self.drag_position = QPoint()
        self.resize_border_width = 8
        self.is_resizing = False
        self.resize_edge = None
        
        font_metrics = QFontMetrics(self.font())
        min_width = max(
            font_metrics.horizontalAdvance("NLP interpreter") + font_metrics.horizontalAdvance("gpt-4o-mini") + 100,
            font_metrics.horizontalAdvance("Hear Voice Sample") + font_metrics.horizontalAdvance("External Play: ON") + 100,
            500
        )
        # Calculate minimum height including transcription box
        transcription_height = max(100, int(font_metrics.height() * 6))
        transcription_label_height = int(font_metrics.height() * 1.5)
        spacing = int(font_metrics.height() * 1.5)  # Spacing between START button and transcription
        base_height = max(int(font_metrics.height() * 15), 350)
        min_height = base_height + transcription_height + transcription_label_height + spacing
        self.setMinimumWidth(int(min_width))
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        base_font = self.font()
        if base_font.pointSize() > 0:
            base_font.setPointSize(int(base_font.pointSize() * 1.0))
        else:
            base_font.setPointSize(10)
        self.setFont(base_font)
        self.main_layout = None

        self.current_voice_index = 0
        self.current_speed = "1x"
        self.current_nlp_model = "gpt-4o-mini"
        self.voice_dropdown = None
        self.speed_dropdown = None
        self.nlp_dropdown = None
        self.current_voice_id = None
        self.cable_in_device_index = None
        self.use_cable_in_for_sample = False
        
        # Transcription setup
        self.transcription_textbox = None
        self.transcription_history = []  # Store last 3 transcriptions

        # Subprocess for ASL recognition
        self.asl_process = None
        self.asl_thread = None
        self.show_camera = False  # Camera display toggle

        self.init_ui()
        self.initialize_tts()
        self.initialize_nlp()
        
        # Connect signal to handler (thread-safe GUI update)
        self.sentence_received.connect(self.handle_line)
        
        # Start ASL recognition subprocess in background (always running)
        self.start_asl_process()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(10, 5, 0, 5)
        top_bar.setSpacing(10)
        
        if os.path.exists(self.icon_path):
            logo_label = QLabel()
            logo_label.setPixmap(QIcon(self.icon_path).pixmap(24, 24))
            top_bar.addWidget(logo_label)
        
        title_label = QLabel("Sign Sync")
        title_label.setObjectName("title_label")
        top_bar.addWidget(title_label)
        top_bar.addStretch()
        
        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(26, 26)
        self.close_button.setObjectName("close_button")
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("""
            QPushButton#close_button {
                background-color: transparent;
                border: none;
                color: #e74c3c;
                font-size: 21px;
                font-weight: bold;
                padding: 0px;
                margin-right: 5px;
            }
            QPushButton#close_button:hover {
                background-color: #c0392b;
                color: #ffffff;
            }
        """)
        top_bar.addWidget(self.close_button)
        top_bar.addSpacing(10)
        layout.addLayout(top_bar)
        
        content_layout = QVBoxLayout()
        self.main_layout = content_layout
        font_metrics = QFontMetrics(self.font())
        
        margin = max(15, int(font_metrics.height() * 0.8))
        spacing = max(10, int(font_metrics.height() * 0.6))
        content_layout.setContentsMargins(margin, margin, margin, margin)
        content_layout.setSpacing(spacing)

        voice_layout, self.voice_dropdown = self.create_dropdown("Voice:", ["Man", "Woman"], self.on_voice_changed)
        content_layout.addLayout(voice_layout)
        
        speed_layout, self.speed_dropdown = self.create_dropdown("Speed:", ["0.5x", "1x", "1.5x", "2x"], self.on_speed_changed)
        self.speed_dropdown.setCurrentText("1x")
        content_layout.addLayout(speed_layout)
        
        self.voice_sample_button = QPushButton("Hear Voice Sample")
        button_height = max(30, int(font_metrics.height() * 1.8))
        button_width = max(120, int(font_metrics.horizontalAdvance("Hear Voice Sample") * 1.3))
        self.voice_sample_button.setMinimumHeight(button_height)
        self.voice_sample_button.setMinimumWidth(button_width)
        self.voice_sample_button.setObjectName("voice_sample_button")
        self.voice_sample_button.clicked.connect(self.play_voice_sample)
        self.voice_sample_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)

        self.external_play_button = QPushButton("External Play")
        external_button_width = max(120, int(font_metrics.horizontalAdvance("External Play: ON") * 1.3))
        self.external_play_button.setMinimumHeight(button_height)
        self.external_play_button.setMinimumWidth(external_button_width)
        self.external_play_button.setObjectName("external_play_button")
        self.external_play_button.clicked.connect(self.toggle_external_play)
        self.external_play_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        self.external_play_button.hide()

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.voice_sample_button)
        button_layout.addWidget(self.external_play_button)
        content_layout.addLayout(button_layout)

        nlp_layout, self.nlp_dropdown = self.create_dropdown("NLP interpreter", ["None", "gpt-3.5-turbo", "gpt-4o-mini"], self.on_nlp_changed)
        self.nlp_dropdown.setCurrentText("gpt-4o-mini")
        content_layout.addLayout(nlp_layout)
        
        # Camera display checkbox
        camera_checkbox_layout = QHBoxLayout()
        self.camera_checkbox = QCheckBox("Show Camera")
        self.camera_checkbox.setChecked(False)
        self.camera_checkbox.stateChanged.connect(self.on_camera_checkbox_changed)
        camera_checkbox_layout.addWidget(self.camera_checkbox)
        camera_checkbox_layout.addStretch()
        content_layout.addLayout(camera_checkbox_layout)

        spacer_height = max(20, int(font_metrics.height() * 2))
        content_layout.addItem(QSpacerItem(int(font_metrics.height()), spacer_height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.start_button = QPushButton("START")
        start_button_height = max(50, int(font_metrics.height() * 2))
        self.start_button.setMinimumHeight(start_button_height)
        self.start_button.setObjectName("start_button")
        self.start_button.isStart = True
        self.start_button.clicked.connect(self.toggle_start_button)
        self.start_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        content_layout.addWidget(self.start_button)

        # Add spacing between START button and transcription box
        transcription_spacing = max(15, int(font_metrics.height() * 1.5))
        content_layout.addItem(QSpacerItem(int(font_metrics.height()), transcription_spacing, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))

        # Transcription text display (below START button)
        self.transcription_label = QLabel("Transcription:")
        self.transcription_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.transcription_label.hide()  # Hidden until START is pressed
        content_layout.addWidget(self.transcription_label)
        
        self.transcription_textbox = QTextEdit()
        self.transcription_textbox.setReadOnly(True)
        transcription_height = max(100, int(font_metrics.height() * 6))
        self.transcription_textbox.setMinimumHeight(transcription_height)
        self.transcription_textbox.setObjectName("transcription_textbox")
        self.transcription_textbox.hide()  # Hidden until START is pressed
        content_layout.addWidget(self.transcription_textbox)
        layout.addLayout(content_layout)
        self.setLayout(layout)

    
    def get_resize_edge(self, pos):
        x, y = pos.x(), pos.y()
        width, height = self.width(), self.height()
        border = self.resize_border_width
        
        if x < border and y < border:
            return 'topLeft'
        elif x >= width - border and y < border:
            return 'topRight'
        elif x < border and y >= height - border:
            return 'bottomLeft'
        elif x >= width - border and y >= height - border:
            return 'bottomRight'
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
        pos = event.position().toPoint()
        
        if not self.is_resizing:
            edge = self.get_resize_edge(pos)
            if edge:
                self.setCursor(self.get_cursor_for_edge(edge))
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            
            if event.buttons() == Qt.MouseButton.LeftButton and self.drag_position:
                self.move(event.globalPosition().toPoint() - self.drag_position)
        else:
            if event.buttons() == Qt.MouseButton.LeftButton and self.resize_edge:
                delta = event.globalPosition().toPoint() - self.drag_position
                geom = self.resize_start_geometry
                min_w, min_h = self.minimumWidth(), self.minimumHeight()
                
                if self.resize_edge == 'left':
                    new_width = max(min_w, geom.width() - delta.x())
                    self.setGeometry(geom.x() + (geom.width() - new_width), geom.y(), new_width, geom.height())
                elif self.resize_edge == 'right':
                    self.setGeometry(geom.x(), geom.y(), max(min_w, geom.width() + delta.x()), geom.height())
                elif self.resize_edge == 'top':
                    new_height = max(min_h, geom.height() - delta.y())
                    self.setGeometry(geom.x(), geom.y() + (geom.height() - new_height), geom.width(), new_height)
                elif self.resize_edge == 'bottom':
                    self.setGeometry(geom.x(), geom.y(), geom.width(), max(min_h, geom.height() + delta.y()))
                elif self.resize_edge == 'topLeft':
                    new_width = max(min_w, geom.width() - delta.x())
                    new_height = max(min_h, geom.height() - delta.y())
                    self.setGeometry(geom.x() + (geom.width() - new_width), geom.y() + (geom.height() - new_height), new_width, new_height)
                elif self.resize_edge == 'topRight':
                    new_height = max(min_h, geom.height() - delta.y())
                    self.setGeometry(geom.x(), geom.y() + (geom.height() - new_height), max(min_w, geom.width() + delta.x()), new_height)
                elif self.resize_edge == 'bottomLeft':
                    new_width = max(min_w, geom.width() - delta.x())
                    self.setGeometry(geom.x() + (geom.width() - new_width), geom.y(), new_width, max(min_h, geom.height() + delta.y()))
                elif self.resize_edge == 'bottomRight':
                    self.setGeometry(geom.x(), geom.y(), max(min_w, geom.width() + delta.x()), max(min_h, geom.height() + delta.y()))
        
        event.accept()
    
    def mouseReleaseEvent(self, event):
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
        
        font = dropdown.font()
        dropdown.setFont(font)
        label.setFont(font)
        
        layout.addWidget(label)
        layout.addWidget(dropdown)
        return layout, dropdown

    def on_voice_changed(self, value):
        self.current_voice_index = 0 if value == "Man" else 1
        self.update_voice_id()

    def on_speed_changed(self, value):
        self.current_speed = value

    def on_nlp_changed(self, value):
        self.current_nlp_model = None if value == "None" else value
    
    def on_camera_checkbox_changed(self, state):
        """Handle camera checkbox toggle - send command via stdin."""
        # state is 0 for unchecked, 2 for checked
        self.show_camera = (state == 2)
        # Send command to ASL process via stdin
        if self.asl_process and self.asl_process.stdin:
            try:
                if self.show_camera:
                    self.asl_process.stdin.write("show_camera\n")
                else:
                    self.asl_process.stdin.write("hide_camera\n")
                self.asl_process.stdin.flush()
            except Exception as e:
                print(f"Error sending camera command: {e}")
    
    def get_nlp_model(self):
        return self.current_nlp_model if self.current_nlp_model else "gpt-4o-mini"

    def initialize_tts(self):
        self.update_voice_id()
        try:
            self.cable_in_device_index = find_vb_audio_device()
        except Exception:
            self.cable_in_device_index = None
    
    def initialize_nlp(self):
        try:
            get_client()
        except Exception:
            pass

    def update_voice_id(self):
        try:
            self.current_voice_id = get_voice_id(self.current_voice_index)
        except Exception:
            try:
                self.current_voice_id = get_voice_id(0)
            except:
                self.current_voice_id = None

    def _calculate_rate(self):
        return int(160 * float(self.current_speed.replace('x', '')))

    def play_voice_sample(self):
        if self.voice_dropdown:
            voice_text = self.voice_dropdown.currentText()
            new_voice_index = 0 if voice_text == "Man" else 1
            if new_voice_index != self.current_voice_index:
                self.current_voice_index = new_voice_index
                self.update_voice_id()
        
        if self.speed_dropdown:
            self.current_speed = self.speed_dropdown.currentText()
        
        rate = self._calculate_rate()
        voice_id = self.current_voice_id or get_voice_id(0)
        if not voice_id:
            return
        
        device_index = self.cable_in_device_index if (self.use_cable_in_for_sample and self.cable_in_device_index) else None
        
        # Add to transcription box
        self.add_to_transcription_box("this is a test voice sample")
        
        def speak_in_thread():
            try:
                speak_text("this is a test voice sample", rate=rate, voice_id=voice_id, sapi_device_index=device_index)
            except Exception as e:
                print(f"Error speaking: {e}")
        
        threading.Thread(target=speak_in_thread, daemon=True).start()

    def get_audio_device_index(self):
        return self.cable_in_device_index if (not self.start_button.isStart and self.cable_in_device_index) else None

    def toggle_external_play(self):
        self.use_cable_in_for_sample = not self.use_cable_in_for_sample
        self.external_play_button.setText("External Play: ON" if self.use_cable_in_for_sample else "External Play: OFF")

    def _speak_text(self, text):
        if not (self.cable_in_device_index and self.current_voice_id):
            return
        
        rate = self._calculate_rate()
        def speak():
            try:
                speak_text(text, rate=rate, voice_id=self.current_voice_id, sapi_device_index=self.cable_in_device_index)
            except Exception as e:
                print(f"Error speaking: {e}")
        threading.Thread(target=speak, daemon=True).start()

    def start_asl_process(self):
        """Start the ASL recognition subprocess and capture its stdout."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        asl_main_path = os.path.join(script_dir, '..', 'asl-text', 'main.py')
        asl_main_path = os.path.normpath(asl_main_path)
        
        if not os.path.exists(asl_main_path):
            print(f"Error: ASL main.py not found at {asl_main_path}")
            return
        
        try:
            # Start the subprocess with unbuffered output
            # Use -u flag for unbuffered output on Windows
            cmd = [sys.executable, '-u', asl_main_path]
            # Add --show-camera flag if checkbox is checked (initial state)
            if self.show_camera:
                cmd.append('--show-camera')
            
            self.asl_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                cwd=os.path.dirname(asl_main_path)
            )
            
            # Start a thread to read stdout
            self.asl_thread = threading.Thread(
                target=self._read_asl_output,
                daemon=True
            )
            self.asl_thread.start()
            print("ASL recognition process started")
        except Exception as e:
            print(f"Error starting ASL process: {e}")
    
    def _read_asl_output(self):
        """Read stdout from the ASL subprocess and process sentences."""
        if not self.asl_process:
            return
        
        try:
            for line in iter(self.asl_process.stdout.readline, ''):
                if line:
                    line = line.rstrip()
                    # Check if this is a sentence output from ASL
                    if line.startswith("sentence:"):
                        # Extract the sentence text (everything after "sentence:")
                        sentence_text = line[len("sentence:"):].strip()
                        if sentence_text:
                            print(sentence_text)
                            # Emit signal to handle in main thread (thread-safe)
                            self.sentence_received.emit(sentence_text)
                    else:
                        # Print other output with [ASL] prefix
                        print(f"[ASL] {line}")
        except Exception as e:
            print(f"Error reading ASL output: {e}")
        finally:
            if self.asl_process:
                self.asl_process.stdout.close()
    
    def stop_asl_process(self):
        """Stop the ASL recognition subprocess."""
        if self.asl_process:
            try:
                self.asl_process.terminate()
                # Wait a bit for graceful shutdown
                try:
                    self.asl_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    self.asl_process.kill()
                    self.asl_process.wait()
                print("ASL recognition process stopped")
            except Exception as e:
                print(f"Error stopping ASL process: {e}")
            finally:
                self.asl_process = None
                self.asl_thread = None

    def toggle_start_button(self):
        if self.start_button.isStart:
            self.start_button.setObjectName("start_button_live")
            self.start_button.setText("LIVE")
            self.external_play_button.show()
            self.external_play_button.setText("External Play: OFF")
            self.use_cable_in_for_sample = False
            self._speak_text("SignSync initialized")
            
            # Show transcription box
            if self.transcription_textbox:
                self.transcription_textbox.show()
                self.transcription_textbox.clear()
                self.transcription_history = []
            if self.transcription_label:
                self.transcription_label.show()
            
            # Add to transcription box
            self.add_to_transcription_box("SignSync initialized")
        else:
            self.start_button.setObjectName("start_button")
            self.start_button.setText("START")
            self.external_play_button.hide()
            self.use_cable_in_for_sample = False
            self._speak_text("SignSync off")
            
            # Add to transcription box
            self.add_to_transcription_box("SignSync off")
            
            # Hide transcription box
            if self.transcription_textbox:
                self.transcription_textbox.hide()
                self.transcription_history = []
            if self.transcription_label:
                self.transcription_label.hide()

        self.start_button.isStart = not self.start_button.isStart
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
    
    def add_to_transcription_box(self, text):
        """Add text to the transcription box, keeping only the last 3 additions."""
        if not self.transcription_textbox:
            return
        
        # Add new text to history
        self.transcription_history.append(text)
        
        # Keep only the last 3 items
        if len(self.transcription_history) > 3:
            self.transcription_history.pop(0)
        
        # Update the textbox with the last 3 items
        self.transcription_textbox.clear()
        for item in self.transcription_history:
            self.transcription_textbox.append(item)
    
    def handle_line(self, sentence_text):
        """Handle a sentence line from ASL recognition."""
        # Only process if SignSync is ON (button shows "LIVE", not "START")
        if self.start_button.isStart:
            return

        # Add to transcription box
        self.add_to_transcription_box(sentence_text)
        
        # Run API call and TTS in background thread to avoid blocking UI
        def process_and_speak():
            try:
                rate = self._calculate_rate()
                voice_id = self.current_voice_id or get_voice_id(self.current_voice_index)
                
                # If no NLP model is set, just speak the text directly
                if self.current_nlp_model is None:
                    speak_text(sentence_text, rate=rate, voice_id=voice_id, sapi_device_index=self.cable_in_device_index)
                else:
                    # Use NLP model to process and then speak
                    send_prompt_and_speak_streaming(
                        sentence_text, 
                        model=self.current_nlp_model,
                        voice_index=self.current_voice_index, 
                        rate=rate, 
                        sapi_device_index=self.cable_in_device_index
                    )
            except Exception as e:
                print(f"Error processing and speaking: {e}")
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=process_and_speak, daemon=True).start()
    
    def closeEvent(self, event):
        """Handle window close event - cleanup subprocess."""
        self.stop_asl_process()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(script_dir, "mini_logo.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    apply_stylesheet(app, theme="dark_red.xml")

    font_path = os.path.join(script_dir, "open_sans.ttf")
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            app.setFont(QFont(font_family, 30))
        else:
            print("Failed to load Open Sans font.")
    else:
        print("open_sans.ttf not found!")
    
    window = MainWindow()
    window.setStyleSheet("""
        QWidget {
            background-color: #000000;
            color: #C0C0C0;
            font-family: 'Open Sans';
            font-size: 18px;
        }

        QLabel {
            color: #C0C0C0;
        }

        QLabel#title_label {
            font-size: 20px;
            font-weight: 600;
        }

        QComboBox {
            background-color: #1a1a1a;
            color: #ffffff;
            padding: 6px;
            border-radius: 4px;
            font-size: 14px;
        }

        QPushButton {
            font-size: 16px;
            padding: 8px 14px;
            border-radius: 6px;
        }

        QPushButton#start_button,
        QPushButton#start_button_live {
            font-size: 18px;
            font-weight: bold;
        }
        #transcription_textbox {
            font-size: 14px;
            background: #1b1b1b;
            color: #ffffff;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #333333;
        }
        """)

    
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))

    window.show()
    sys.exit(app.exec())
