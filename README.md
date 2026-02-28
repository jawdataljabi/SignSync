# SignSync

SignSync is a real-time ASL → Text → Speech desktop app that lets an ASL user “speak” inside video meeting platforms (Zoom, Google Meet, Discord, etc.) by translating live signing into audio output through a virtual microphone and video through a virtual camera.

- Devpost: https://devpost.com/software/sign-sync
- Demo video: https://youtu.be/AOC3kynHwjo

## What it does

SignSync converts ASL gestures into spoken audio in real time and injects it into any meeting platform.

End-to-end pipeline:
1. **Camera Input**: user signs in front of a webcam  
2. **Virtual Camera Layer**: meeting apps see the feed as a normal webcam
3. **ASL Recognition**: frames are processed to extract landmarks and classify signs (MediaPipe + TensorFlow)
4. **NLP Cleanup**: recognized words are cleaned into readable sentences (LLM-based grammar repair)
5. **Text-to-Speech**: sentence is synthesized to speech (pyttsx3)
6. **Virtual Audio Output**: audio is routed into a virtual microphone so meeting apps receive it as live speech

## Repo structure

- `asl-text/` - ASL recognition pipeline (webcam frames → predicted tokens/words)
- `text-speech/` - text cleanup + TTS output
- `ui/` - desktop UI / orchestration
- `speech.mp3` - tts

## Tech stack

- Python, PyQt6
- OpenCV (video capture + frame processing)
- MediaPipe Holistic (landmark extraction)
- TensorFlow (gesture classification)
- ZMQ (inter-process messaging)
- PyVirtualCam (virtual webcam output)
- pyttsx3 (offline TTS)
- Virtual Audio Cable (virtual mic routing on Windows)

## Setup (local)

### Prerequisites
- Python 3.10+ recommended
- A webcam
- **Windows** (recommended for the demo setup) with a virtual audio driver (e.g., Virtual Audio Cable)
- A virtual camera sink (handled by `pyvirtualcam`)

### Install Python dependencies
Create a virtual environment, then install the core deps:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install opencv-python mediapipe tensorflow pyqt6 pyvirtualcam pyttsx3 pyzmq
