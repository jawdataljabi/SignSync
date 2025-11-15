import pyttsx3
import sys


def get_voice_id(voice_index=1):
    """Get the voice ID for the specified voice index.
    
    Args:
        voice_index: Index of the voice to use (default: 1)
    
    Returns:
        Voice ID string
    """
    temp_engine = pyttsx3.init()
    voices = temp_engine.getProperty("voices")
    for i, v in enumerate(voices):
        print(i, v.id, v.name, flush=True)
    voice_id = voices[voice_index].id if len(voices) > voice_index else voices[0].id
    del temp_engine
    return voice_id


def speak_text(text, rate=160, volume=0.9, voice_id=None):
    """Speak the given text using TTS.
    
    Args:
        text: String containing the sentence to speak
        rate: Words per minute (default: 160)
        volume: Volume level 0.0 to 1.0 (default: 0.9)
        voice_id: Voice ID to use (default: None, uses default voice)
    """
    # Create a new engine instance for each utterance to avoid Windows pyttsx3 issues
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    if voice_id:
        engine.setProperty("voice", voice_id)
    engine.say(text)
    engine.runAndWait()
    del engine


def main():
    """Main function that reads from stdin and speaks text."""
    # Get voice ID at startup
    voice_id = get_voice_id(1)
    
    print("TTS process started, waiting for input...", file=sys.stderr, flush=True)
    
    while True:
        try:
            # readline() will block until a newline is received or EOF
            print("Waiting for input...", file=sys.stderr, flush=True)
            text = sys.stdin.readline()
            print(f"Received: {repr(text)}", file=sys.stderr, flush=True)
            
            if not text:  # EOF - pipe is closed, exit
                print("EOF detected, exiting", file=sys.stderr, flush=True)
                break
            
            text = text.strip()
            if text:  # Only speak non-empty lines
                print(f"Speaking: {text}", file=sys.stderr, flush=True)
                speak_text(text, voice_id=voice_id)
                print("Finished speaking", file=sys.stderr, flush=True)
                
        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting", file=sys.stderr, flush=True)
            break
        except (EOFError, BrokenPipeError) as e:
            # On EOF or broken pipe, exit
            print(f"Pipe error: {e}, exiting", file=sys.stderr, flush=True)
            break
        except Exception as e:
            # Log other exceptions but continue
            print(f"Error: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue


if __name__ == "__main__":
    main()

# from gtts import gTTS
# import os

# tts = gTTS("Hello! This is a test using gTTS.", lang='en')
# tts.save("speech.mp3")

# # Windows: open the MP3 with default player
# os.startfile("speech.mp3")

