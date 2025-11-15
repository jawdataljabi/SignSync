import os
import time
from openai import OpenAI
from tts import speak_text, get_voice_id

# Global client instance (initialized on first use)
_client = None
# Global voice ID (initialized on first use)
_voice_id = None

# Prefix to prepend to all prompts
# Using a clear structure to separate instruction from user input
PROMPT_PREFIX = "You are a text clarity assistant. Your task is to rewrite the user's sentence to be clearer and more understandable. Respond with exactly one improved sentence, nothing else.\n\nUser's sentence to rewrite: "


def get_client(api_key=None):
    """Get or create an OpenAI client instance.
    
    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
    
    Returns:
        OpenAI client instance
    """
    global _client
    if _client is None:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        _client = OpenAI(api_key=api_key)
    return _client


def send_prompt(prompt, model="gpt-3.5-turbo", temperature=0.7, system_message=None):
    """Send a prompt to OpenAI and receive the response as a string.
    
    Args:
        prompt: The user's prompt/question as a string
        model: Model to use (default: "gpt-3.5-turbo")
        temperature: Sampling temperature 0.0-2.0 (default: 0.7)
        system_message: Optional system message to set context
    
    Returns:
        Response string from OpenAI
    """
    client = get_client()
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    # Prepend the prompt with the instruction prefix
    prefixed_prompt = PROMPT_PREFIX + prompt
    messages.append({"role": "user", "content": prefixed_prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content


def send_prompt_and_speak_streaming(prompt, model="gpt-4o-mini", temperature=0.7, system_message=None, voice_index=1, min_chunk_length=10, sapi_device_index=None):
    """Send a prompt to OpenAI with streaming, collect the full response, then speak it all at once.
    
    Args:
        prompt: The user's prompt/question as a string
        model: Model to use (default: "gpt-4o-mini")
        temperature: Sampling temperature 0.0-2.0 (default: 0.7)
        system_message: Optional system message to set context
        voice_index: Voice index to use for TTS (default: 1)
        min_chunk_length: Deprecated parameter, kept for backward compatibility (not used)
        sapi_device_index: SAPI audio output device index (default: None, uses system default)
    
    Returns:
        Tuple of (full response string, timing dict with metrics:
            - 'api_first_token_ms': Time from API call to first token received
            - 'api_total_ms': Time from API call to last token received (total API time)
            - 'api_to_speech_start_ms': Time from API call to when speaking starts
            - 'speaking_total_ms': Total time spent speaking
            - 'function_total_ms': Total function execution time)
    """
    global _voice_id
    
    # Start comprehensive timing
    api_call_start = time.time()
    first_token_time = None
    last_token_time = None
    first_speech_start = None
    last_speech_end = None
    
    client = get_client()
    
    # Pre-initialize voice ID to avoid delay during streaming
    if _voice_id is None:
        _voice_id = get_voice_id(voice_index)
    
    # Prepare messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    # Prepend the prompt with the instruction prefix
    prefixed_prompt = PROMPT_PREFIX + prompt
    messages.append({"role": "user", "content": prefixed_prompt})
    
    # Stream the response and collect it all
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True
    )
    
    # Collect the full response
    full_response = ""
    
    # Process stream chunks to collect full response
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            # Record first token time
            if first_token_time is None:
                first_token_time = time.time()
            
            # Record last token time (updated with each chunk)
            last_token_time = time.time()
            
            # Add chunk to full response
            chunk_text = chunk.choices[0].delta.content
            full_response += chunk_text
    
    # Speak the entire response at once
    if full_response.strip():
        first_speech_start = time.time()
        speak_start = time.time()
        speak_text(full_response.strip(), voice_id=_voice_id, sapi_device_index=sapi_device_index)
        speak_end = time.time()
        last_speech_end = speak_end
    
    # End of function timing
    function_end = time.time()
    
    # Calculate comprehensive timing metrics
    timing = {}
    
    # API timing
    if first_token_time is not None:
        timing['api_first_token_ms'] = round((first_token_time - api_call_start) * 1000, 2)
    else:
        timing['api_first_token_ms'] = None
    
    if last_token_time is not None:
        timing['api_total_ms'] = round((last_token_time - api_call_start) * 1000, 2)
    else:
        timing['api_total_ms'] = None
    
    # Speech timing
    if first_speech_start is not None:
        timing['api_to_speech_start_ms'] = round((first_speech_start - api_call_start) * 1000, 2)
    else:
        timing['api_to_speech_start_ms'] = None
    
    if first_speech_start is not None and last_speech_end is not None:
        timing['speaking_total_ms'] = round((last_speech_end - first_speech_start) * 1000, 2)
    else:
        timing['speaking_total_ms'] = None
    
    # Total function timing
    timing['function_total_ms'] = round((function_end - api_call_start) * 1000, 2)
    
    return full_response, timing


def main():
    """Interactive benchmarking loop."""
    global _voice_id
    
    # Initialize voice ID once at startup to avoid wasting time in benchmarks
    print("Initializing TTS voice...")
    if _voice_id is None:
        _voice_id = get_voice_id(1)
    print("Ready! Enter prompts to benchmark (type 'exit' to quit)\n")
    
    while True:
        try:
            # Get user input
            prompt = input("Enter prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() == 'exit':
                print("Exiting...")
                break
            
            # Send prompt using streaming
            print(f"\nSending prompt: {prompt}")
            print("Processing...")
            
            response, timing = send_prompt_and_speak_streaming(
                prompt,
                voice_index=1  # Voice already initialized, but pass for consistency
            )
            
            # Display timing results
            print(f"\n{'='*60}")
            print("TIMING BENCHMARKS")
            print(f"{'='*60}")
            
            # Show first token time
            first_token = timing.get('api_first_token_ms')
            if first_token is not None:
                print(f"API Start → First Token:     {first_token:>10.2f} ms")
            
            # Show API total time
            api_total = timing.get('api_total_ms', 0)
            print(f"API Total Time:                {api_total:>10.2f} ms")
            
            # Show API to speech start
            api_to_speech = timing.get('api_to_speech_start_ms', 0)
            print(f"API Start → Speech Start:     {api_to_speech:>10.2f} ms")
            
            # Show breakdown for streaming: first token to speech start
            if first_token is not None:
                time_from_first_token = api_to_speech - first_token
                print(f"  (First Token → Speech):     {time_from_first_token:>10.2f} ms")
            
            # Show speaking time
            speaking = timing.get('speaking_total_ms', 0)
            print(f"Speaking Total:                {speaking:>10.2f} ms")
            
            # Show function total time
            function_total = timing.get('function_total_ms', 0)
            print(f"Function Total:                 {function_total:>10.2f} ms")
            
            print(f"{'='*60}")
            print(f"\nResponse: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

