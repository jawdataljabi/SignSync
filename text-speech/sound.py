import pyttsx3
import win32com.client as wincl
import ctypes


def inspect_engine_structure():
    """Inspect the pyttsx3 engine structure to find SAPI access path."""
    engine = pyttsx3.init()
    
    print("Engine attributes:")
    print(f"  dir(engine): {[x for x in dir(engine) if not x.startswith('__')]}")
    
    if hasattr(engine, '_driver'):
        print("\nEngine._driver found!")
        print(f"  dir(_driver): {[x for x in dir(engine._driver) if not x.startswith('__')]}")
        
        if hasattr(engine._driver, '_tts'):
            print("\nEngine._driver._tts found!")
            print(f"  dir(_tts): {[x for x in dir(engine._driver._tts) if not x.startswith('__')]}")
            
            if hasattr(engine._driver._tts, '_engine'):
                print("\nEngine._driver._tts._engine found!")
                return engine._driver._tts._engine
    
    if hasattr(engine, 'proxy'):
        print("\nEngine.proxy found!")
        print(f"  dir(proxy): {[x for x in dir(engine.proxy) if not x.startswith('__')]}")
        
        if hasattr(engine.proxy, '_driver'):
            print("\nEngine.proxy._driver found!")
            print(f"  dir(_driver): {[x for x in dir(engine.proxy._driver) if not x.startswith('__')]}")
            
            if hasattr(engine.proxy._driver, '_tts'):
                print("\nEngine.proxy._driver._tts found!")
                print(f"  dir(_tts): {[x for x in dir(engine.proxy._driver._tts) if not x.startswith('__')]}")
                
                if hasattr(engine.proxy._driver._tts, '_engine'):
                    print("\nEngine.proxy._driver._tts._engine found!")
                    return engine.proxy._driver._tts._engine
    
    # Try to find any attribute that might be the SAPI voice
    print("\nSearching for SAPI-related attributes...")
    for attr in dir(engine):
        if not attr.startswith('__'):
            try:
                obj = getattr(engine, attr)
                if hasattr(obj, 'AudioOutput'):
                    print(f"Found AudioOutput in: engine.{attr}")
                    return obj
            except:
                pass
    
    return None


def list_sapi_devices():
    """List all available SAPI audio output devices.
    
    Returns:
        List of tuples (index, token_id, description) using comtypes
    """
    import comtypes.client
    
    # Use comtypes to get tokens (compatible with pyttsx3)
    category = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
    category.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\AudioOutput", False)
    tokens = category.EnumerateTokens()
    
    devices = []
    print("\nAvailable SAPI Audio Output Devices:")
    for i in range(tokens.Count):
        token = tokens.Item(i)
        token_id = token.Id
        description = token.GetDescription()
        devices.append((i, token_id, description, token))
        print(f"  [{i}] {description}")
    
    return devices


def set_sapi_output_device(device_index):
    """Set the SAPI audio output device for pyttsx3.
    
    Args:
        device_index: Index of the SAPI audio output device to use
    
    Returns:
        Configured pyttsx3 engine
    """
    import comtypes.client
    
    # Get SAPI audio output tokens using comtypes (compatible with pyttsx3)
    category = comtypes.client.CreateObject("SAPI.SpObjectTokenCategory")
    category.SetId("HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\AudioOutput", False)
    tokens = category.EnumerateTokens()
    
    if 0 <= device_index < tokens.Count:
        selected_token = tokens.Item(device_index)
        description = selected_token.GetDescription()
        print(f"\nSelected device: {description}")
        
        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        
        # Access the SAPI Voice object directly
        # Based on inspection: engine.proxy._driver._tts IS the SAPI Voice object
        voice = engine.proxy._driver._tts
        
        if hasattr(voice, 'AudioOutput'):
            # Set the AudioOutput property directly (both are comtypes objects now)
            voice.AudioOutput = selected_token
            print(f"✓ Successfully set AudioOutput to: {description}")
            return engine
        else:
            raise AttributeError("SAPI Voice object does not have AudioOutput property")
    else:
        raise ValueError(f"Invalid device index: {device_index}. Available devices: 0-{tokens.Count-1}")


if __name__ == "__main__":
    # First, inspect the engine structure
    print("=" * 60)
    print("INSPECTING PYTTSX3 ENGINE STRUCTURE")
    print("=" * 60)
    inspect_engine_structure()
    
    # List devices
    print("\n" + "=" * 60)
    print("SAPI DEVICES")
    print("=" * 60)
    devices = list_sapi_devices()
    
    # Try to set device and speak
    if len(devices) > 0:
        print("\n" + "=" * 60)
        print("TESTING DEVICE SELECTION")
        print("=" * 60)
        device_index = int(input("\nEnter device index to use: "))
        
        try:
            engine = set_sapi_output_device(device_index)
            
            print("\nSpeaking test...")
            engine.say("Audio is now routed through the selected SAPI device without temporary files.")
            engine.runAndWait()
            print("✓ Success!")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
