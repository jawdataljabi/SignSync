from tts import speak_text, get_voice_id, list_sapi_devices


def main():
    """Main function that prompts for input and speaks text directly."""
    # List SAPI devices if available
    sapi_devices = list_sapi_devices()
    sapi_device_index = None
    
    if sapi_devices:
        print("\nSelect SAPI audio output device:")
        print("  Enter device number (e.g., '0', '1', '2')")
        print("  Or press Enter to use default device")
        device_choice = input("Device index (or Enter for default): ").strip()
        
        if device_choice:
            try:
                device_index = int(device_choice)
                if 0 <= device_index < len(sapi_devices):
                    sapi_device_index = device_index
                    print(f"Selected: [{device_index}] {sapi_devices[device_index][2]}")
                else:
                    print(f"Invalid device number, using default device")
            except ValueError:
                print(f"Invalid input, using default device")
        else:
            print("Using default SAPI audio device")
    else:
        print("SAPI device selection not available (comtypes not installed)")
    
    # Get voice ID at startup
    print("\nSelecting TTS voice...")
    voice_id = get_voice_id(1)
    
    print("\nTTS ready. Enter text to speak (or 'exit' to quit)")
    
    while True:
        try:
            user_input = input("Enter text to send (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            
            if user_input.strip():
                print(f"Speaking: {user_input}")
                speak_text(user_input, voice_id=voice_id, sapi_device_index=sapi_device_index)
                print("Finished speaking")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

