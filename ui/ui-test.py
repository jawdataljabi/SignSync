from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
import sys


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sign Sync")
        self.setMinimumWidth(300)
        self.setWindowIcon(QIcon("Logo.png"))

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        layout.addLayout(self.create_dropdown("Voice:", ["Voice 1", "Voice 2", "Voice 3"], self.on_voice_changed))
        layout.addLayout(self.create_dropdown("Speed:", ["0.5x", "1x", "1.5x", "2x"], self.on_speed_changed))

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

        return layout

    def on_voice_changed(self, value):
        print("Selected voice:", value)

    def on_speed_changed(self, value):
        print("Selected speed:", value)

    def toggle_start_button(self):
        print("toggling start button")
        if self.start_button.isStart:
            self.start_button.setObjectName("start_button_live")
            self.start_button.setText("LIVE")
        else:
            self.start_button.setObjectName("start_button")
            self.start_button.setText("START")

        self.start_button.isStart = not self.start_button.isStart

        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    with open("style.qss", "r") as f:
        app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
