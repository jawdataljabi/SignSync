import pyttsx3
import sys

# Try to import comtypes for SAPI device selection
try:
    import comtypes.client
    COMTYPES_AVAILABLE = True
except ImportError:
    COMTYPES_AVAILABLE = False

# Cache for VB-Audio device index (searched once at module load)
_VB_AUDIO_DEVICE_INDEX = None
_VB_AUDIO_SEARCHED = False


def find_vb_audio_device():
    """Find and return the index of a device containing "CABLE In" in its name.
    
    Returns:
        Device index if found
        
    Raises:
        RuntimeError: If "CABLE In" device is not found or comtypes is unavailable
    """
    global _VB_AUDIO_DEVICE_INDEX, _VB_AUDIO_SEARCHED
    
    # Return cached value if already searched
    if _VB_AUDIO_SEARCHED:
        if _VB_AUDIO_DEVICE_INDEX is None:
            raise RuntimeError("CABLE In device not found (cached result)")
        return _VB_AUDIO_DEVICE_INDEX
    
    _VB_AUDIO_SEARCHED = True
    
    if not COMTYPES_AVAILABLE:
        raise RuntimeError("comtypes not available - cannot search for CABLE In device")
    
    try:
        category = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
        category.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\AudioOutput", False)
        tokens = category.EnumerateTokens()
        
        # Look for "CABLE In" device (required)
        for i in range(tokens.Count):
            token = tokens.Item(i)
            description = token.GetDescription()
            # Must have "CABLE In" in the description
            if "CABLE In" in description:
                _VB_AUDIO_DEVICE_INDEX = i
                print(f"[DEBUG] Found CABLE In device: [{i}] {description}", file=sys.stderr, flush=True)
                return i
        
        # If we get here, no CABLE In device was found
        available_devices = []
        for i in range(tokens.Count):
            token = tokens.Item(i)
            description = token.GetDescription()
            available_devices.append(f"  [{i}] {description}")
        
        error_msg = (
            "CABLE In device not found. Available SAPI audio output devices:\n" +
            "\n".join(available_devices) +
            "\n\nPlease ensure VB-Audio Virtual Cable is installed and configured."
        )
        raise RuntimeError(error_msg)
        
    except RuntimeError:
        # Re-raise RuntimeError (our custom errors)
        raise
    except Exception as e:
        raise RuntimeError(f"Error searching for CABLE In device: {e}") from e


def list_sapi_devices():
    """List all available SAPI audio output devices.
    
    Returns:
        List of tuples (index, token_id, description, token)
    """
    if not COMTYPES_AVAILABLE:
        print("comtypes not available, cannot list SAPI devices", file=sys.stderr, flush=True)
        return []
    
    devices = []
    try:
        category = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
        category.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\AudioOutput", False)
        tokens = category.EnumerateTokens()
        
        print("\nAvailable SAPI Audio Output Devices:", file=sys.stderr, flush=True)
        for i in range(tokens.Count):
            token = tokens.Item(i)
            token_id = token.Id
            description = token.GetDescription()
            devices.append((i, token_id, description, token))
            print(f"  [{i}] {description}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Error listing SAPI devices: {e}", file=sys.stderr, flush=True)
    
    return devices


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


def speak_text(text, rate=160, volume=0.9, voice_id=None, sapi_device_index=None):
    """Speak the given text using TTS.
    
    Args:
        text: String containing the sentence to speak
        rate: Words per minute (default: 160)
        volume: Volume level 0.0 to 1.0 (default: 0.9)
        voice_id: Voice ID to use (default: None, uses default voice)
        sapi_device_index: SAPI audio output device index (default: None, auto-finds VB-Audio or uses default)
    """
    # Create a new engine instance for each utterance to avoid Windows pyttsx3 issues
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    if voice_id:
        engine.setProperty("voice", voice_id)
    
    # Set SAPI audio output device
    if COMTYPES_AVAILABLE:
        try:
            # Determine which device to use
            device_to_use = sapi_device_index
            
            # If no device specified, try to find CABLE In device (uses cache, required)
            if device_to_use is None:
                try:
                    device_to_use = find_vb_audio_device()
                    print(f"[DEBUG] Auto-selected CABLE In device at index {device_to_use}", 
                          file=sys.stderr, flush=True)
                except RuntimeError as e:
                    # CABLE In device is required - fail the program
                    print(f"ERROR: {e}", file=sys.stderr, flush=True)
                    raise
            else:
                print(f"[DEBUG] Using explicitly specified SAPI device at index {device_to_use}", 
                      file=sys.stderr, flush=True)
            
            # Set the device if we have one
            if device_to_use is not None:
                # Get SAPI audio output tokens using comtypes
                category = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
                category.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\AudioOutput", False)
                tokens = category.EnumerateTokens()
                
                if 0 <= device_to_use < tokens.Count:
                    selected_token = tokens.Item(device_to_use)
                    device_description = selected_token.GetDescription()
                    print(f"[DEBUG] Setting SAPI device: [{device_to_use}] {device_description}", 
                          file=sys.stderr, flush=True)
                    # Access the SAPI Voice object and set AudioOutput
                    voice = engine.proxy._driver._tts
                    voice.AudioOutput = selected_token
                    print("[DEBUG] SAPI device set successfully", file=sys.stderr, flush=True)
                else:
                    print(f"[DEBUG] Warning: Device index {device_to_use} out of range (0-{tokens.Count-1}), using default", 
                          file=sys.stderr, flush=True)
        except Exception as e:
            # If device selection fails, continue with default device
            print(f"[DEBUG] Warning: Could not set SAPI device: {e}", 
                  file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
    
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
