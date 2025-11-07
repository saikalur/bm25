#!/usr/bin/env python3
"""
OpenAI Realtime Voice Agent - Self-sufficient Python script
Uses OpenAI Realtime API for voice conversations on your laptop
"""

import json
import os
import asyncio
import logging
import base64
from pathlib import Path
import signal
import sys

try:
    from openai import OpenAI
    import pyaudio
except ImportError:
    print("Missing required packages. Install with: pip install openai pyaudio")
    sys.exit(1)

# Configuration
LOG_DIR = Path("log")
LOG_DIR.mkdir(exist_ok=True)
DEBUG_LOG = LOG_DIR / "realtime_agent.log"

# Prompt configuration file
PROMPT_CONFIG_FILE = Path("prompt_config.json")

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # INFO level to see important events
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DEBUG_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000  # OpenAI Realtime API uses 24kHz

# Default prompts (can be overridden in prompt_config.json)
DEFAULT_PROMPTS = {
    "system_prompt": """You are a helpful and friendly AI assistant with a natural, human-like conversation style. 
You speak clearly and expressively, using natural pauses and intonation. 
You are warm, empathetic, and engaging in your conversations.""",
    
    "voice": "alloy",  # Options: alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar
    "temperature": 0.8,
    "max_tokens": 4096,
    "first_sentence": "",  # Optional: First sentence the agent will say at the start
    "instructions": """When having a conversation:
- Be natural and conversational
- Use appropriate pauses and emphasis
- Match the user's energy and tone
- Ask clarifying questions when needed
- Be concise but thorough"""
}

# Global variables
audio = None
is_running = False


def load_prompt_config() -> dict:
    """Load prompt configuration from JSON file or use defaults"""
    if PROMPT_CONFIG_FILE.exists():
        try:
            with open(PROMPT_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                prompts = DEFAULT_PROMPTS.copy()
                prompts.update(config)
                logger.info(f"Loaded prompt configuration from {PROMPT_CONFIG_FILE}")
                return prompts
        except Exception as e:
            logger.error(f"Error loading prompt config: {e}, using defaults")
    
    # Create default config file if it doesn't exist
    try:
        with open(PROMPT_CONFIG_FILE, 'w') as f:
            json.dump(DEFAULT_PROMPTS, f, indent=2)
        logger.info(f"Created default prompt config file: {PROMPT_CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error creating prompt config: {e}")
    
    return DEFAULT_PROMPTS.copy()


def get_openai_config() -> str:
    """Get OpenAI API key from config file or environment"""
    api_key = os.getenv('OPENAI_API_KEY', '')
    
    config_file = Path('openai_config.json')
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                api_key = config.get('api_key', api_key)
                logger.info("Loaded OpenAI configuration from openai_config.json")
        except Exception as e:
            logger.error(f"Error loading OpenAI config: {e}")
    
    if not api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable or create openai_config.json")
        return None
    
    return api_key


def initialize_audio():
    """Initialize PyAudio for microphone input and speaker output"""
    global audio
    try:
        audio = pyaudio.PyAudio()
        logger.info("Audio system initialized")
        return True
    except Exception as e:
        logger.error(f"Error initializing audio: {e}")
        return False


def list_audio_devices():
    """List available audio input/output devices"""
    if not audio:
        return
    
    print("\n=== Available Audio Devices ===")
    print("Input devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} ({info['maxInputChannels']} channels)")
    
    print("\nOutput devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']} ({info['maxOutputChannels']} channels)")
    print()


def get_audio_device_index(device_type='input'):
    """Get audio device index from config or use default"""
    config_file = Path('openai_config.json')
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                key = f'{device_type}_device_index'
                if key in config:
                    return config[key]
        except:
            pass
    return None


async def create_realtime_session(api_key: str, prompts: dict):
    """Create and manage OpenAI Realtime API session"""
    global is_running
    
    try:
        # Create OpenAI client (use AsyncOpenAI for async operations)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        
        # Connect to Realtime API (only model parameter allowed)
        async with client.beta.realtime.connect(
            model="gpt-4o-realtime-preview-2024-10-01"
        ) as session:
            
            # Build instructions - include first sentence if configured
            instructions = prompts.get('system_prompt', '') + "\n\n" + prompts.get('instructions', '')
            first_sentence = prompts.get('first_sentence', '')
            if first_sentence:
                instructions += f"\n\nIMPORTANT: When the conversation starts, you must begin by saying exactly this: {first_sentence}. This is your opening greeting. Do not refuse or modify it."
            
            # Configure the session by sending events
            await session.send({
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": instructions,
                    "voice": prompts.get('voice', 'alloy'),
                    "temperature": prompts.get('temperature', 0.8),
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.4,  # Lower threshold = more sensitive to speech
                        "prefix_padding_ms": 200,  # Shorter padding = faster detection
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True  # Enable interruption
                    },
                    "max_response_output_tokens": prompts.get('max_tokens', 4096)
                }
            })
            
            logger.info("Connected to OpenAI Realtime API")
            logger.info(f"Voice: {prompts.get('voice', 'alloy')}")
            print("\n=== Conversation Started ===")
            print("Speak into your microphone. Press Ctrl+C to end the call.\n")
            
            is_running = True
            
            # Set up audio streams first (before sending first sentence)
            input_device = get_audio_device_index('input')
            output_device = get_audio_device_index('output')
            
            # Input stream (microphone)
            input_stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=CHUNK
            )
            
            # Output stream (speakers)
            output_stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                output_device_index=output_device,
                frames_per_buffer=CHUNK
            )
            
            # Wait a moment for session to be ready, then trigger first sentence
            await asyncio.sleep(1.0)
            
            # Trigger the first sentence by creating an empty response
            # The session instructions will tell it what to say
            if first_sentence:
                logger.info(f"First sentence configured: {first_sentence}")
                print(f"BM25: {first_sentence}\n")
                # Create a response - the system instructions will make it say the first sentence
                await session.send({
                    "type": "response.create"
                })
            
            async def send_audio():
                """Send audio from microphone to OpenAI"""
                try:
                    while is_running:
                        data = input_stream.read(CHUNK, exception_on_overflow=False)
                        # Convert to base64 for Realtime API
                        audio_data = base64.b64encode(data).decode('utf-8')
                        await session.send({
                            "type": "input_audio_buffer.append",
                            "audio": audio_data
                        })
                        await asyncio.sleep(0.01)
                except Exception as e:
                    logger.error(f"Error sending audio: {e}")
                    import traceback
                    traceback.print_exc()
            
            async def receive_events():
                """Receive events from OpenAI and handle them"""
                # Track current response ID for interruption
                current_response_id = None
                
                try:
                    async for event in session:
                        # Get event type - handle both dict and Pydantic models
                        if isinstance(event, dict):
                            event_type = event.get("type")
                        else:
                            # Try to get type from object
                            event_type = getattr(event, 'type', None)
                            if event_type is None:
                                # Try to get class name
                                class_name = event.__class__.__name__.lower()
                                if 'audio' in class_name and 'delta' in class_name:
                                    event_type = "response.audio.delta"
                                elif 'transcript' in class_name and 'delta' in class_name:
                                    event_type = "response.audio_transcript.delta"
                                else:
                                    logger.debug(f"Unknown event class: {event.__class__.__name__}")
                                    continue
                        
                        # Log important events (not all, to reduce noise)
                        if event_type not in ["input_audio_buffer.append", "response.audio.delta"]:
                            logger.debug(f"Event received: {event_type}")
                        
                        # Handle audio output
                        if event_type == "response.audio.delta":
                            # Play audio chunk
                            if isinstance(event, dict):
                                delta = event.get("delta")
                            else:
                                delta = getattr(event, 'delta', None)
                            
                            if delta:
                                try:
                                    audio_bytes = base64.b64decode(delta)
                                    output_stream.write(audio_bytes)
                                    logger.debug(f"Played audio chunk: {len(audio_bytes)} bytes")
                                except Exception as e:
                                    logger.error(f"Error decoding audio: {e}")
                        
                        # Handle transcription events
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            # Show user's transcribed speech
                            if isinstance(event, dict):
                                transcript = event.get("transcript")
                            else:
                                transcript = getattr(event, 'transcript', None)
                            if transcript:
                                print(f"You: {transcript}")
                        elif event_type == "conversation.item.input_audio_transcription.delta":
                            # Handle incremental transcription (optional - can show partial results)
                            logger.debug("Transcription delta received")
                        
                        # Handle AI response text
                        elif event_type == "response.audio_transcript.delta":
                            # Show AI's response text
                            if isinstance(event, dict):
                                delta = event.get("delta")
                            else:
                                delta = getattr(event, 'delta', None)
                            if delta:
                                print(delta, end='', flush=True)
                        
                        elif event_type == "response.audio_transcript.done":
                            print("\n")
                        
                        # Handle response events
                        elif event_type == "response.done":
                            logger.debug("Response done")
                            current_response_id = None  # Clear response ID when done
                        elif event_type == "response.audio.done":
                            logger.debug("Audio response completed")
                        
                        # Handle errors (but ignore input_audio_buffer_commit_empty since server handles commits)
                        elif event_type == "error":
                            if isinstance(event, dict):
                                error_msg = event.get("error")
                                error_code = event.get("error", {}).get("code") if isinstance(event.get("error"), dict) else None
                            else:
                                error_msg = getattr(event, 'error', None)
                                error_code = getattr(error_msg, 'code', None) if hasattr(error_msg, 'code') else None
                            
                            if isinstance(error_msg, dict):
                                error_code = error_msg.get("code")
                                error_msg = error_msg.get("message", str(error_msg))
                            
                            # Ignore the "buffer too small" error - server VAD handles commits automatically
                            if error_code == "input_audio_buffer_commit_empty":
                                logger.debug(f"Ignoring expected error: {error_msg}")
                            else:
                                logger.error(f"API error: {error_msg}")
                        
                        # Handle session events
                        elif event_type == "session.created":
                            logger.info("Session created successfully")
                        elif event_type == "session.updated":
                            logger.info("Session updated successfully")
                        
                        # Handle response events to track when AI is speaking
                        elif event_type == "response.created":
                            if isinstance(event, dict):
                                current_response_id = event.get("response", {}).get("id")
                            else:
                                current_response_id = getattr(getattr(event, 'response', None), 'id', None)
                            logger.debug("Response created - AI is responding!")
                        
                        # Handle input audio buffer events
                        elif event_type == "input_audio_buffer.speech_started":
                            logger.debug("Speech started - listening...")
                            # Manually cancel response if AI is speaking to ensure interruption works
                            if current_response_id:
                                try:
                                    await session.send({
                                        "type": "response.cancel",
                                        "response_id": current_response_id
                                    })
                                    logger.debug(f"Cancelled response {current_response_id} due to user interruption")
                                    current_response_id = None
                                except Exception as e:
                                    logger.debug(f"Could not cancel response: {e}")
                        elif event_type == "input_audio_buffer.speech_stopped":
                            logger.debug("Speech stopped - server will auto-commit and create response")
                            # Don't manually commit - server VAD handles this automatically with create_response: true
                        elif event_type == "input_audio_buffer.committed":
                            logger.debug("Audio buffer committed by server")
                        elif event_type == "response.cancelled":
                            logger.debug("Response was cancelled/interrupted")
                        
                        # Handle conversation and response events
                        elif event_type == "conversation.item.created":
                            logger.debug("Conversation item created")
                        elif event_type == "response.output_item.added":
                            logger.debug("Response output item added")
                        elif event_type == "response.content_part.added":
                            logger.debug("Response content part added")
                        
                        # Handle other events (for debugging)
                        else:
                            logger.debug(f"Unhandled event type: {event_type}")
                            
                except Exception as e:
                    logger.error(f"Error receiving events: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Run both tasks concurrently
            await asyncio.gather(
                send_audio(),
                receive_events()
            )
            
            # Cleanup streams
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Session error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_running = False
        print("\n=== Conversation Ended ===")


def cleanup():
    """Cleanup resources"""
    global audio, is_running
    is_running = False
    if audio:
        audio.terminate()
    logger.info("Cleaned up resources")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nEnding conversation...")
    cleanup()
    sys.exit(0)


async def main():
    """Main function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== OpenAI Realtime Voice Agent ===")
    print(f"Log directory: {LOG_DIR.absolute()}")
    print(f"Prompt config: {PROMPT_CONFIG_FILE.absolute()}\n")
    
    # Load configuration
    prompts = load_prompt_config()
    api_key = get_openai_config()
    
    if not api_key:
        print("\nError: OpenAI API key required!")
        print("Create openai_config.json with:")
        print('  {"api_key": "your-api-key-here"}')
        print("Or set OPENAI_API_KEY environment variable")
        return
    
    # Initialize audio
    if not initialize_audio():
        print("\nError: Could not initialize audio system")
        print("Make sure your microphone and speakers are connected")
        return
    
    # List audio devices (optional, for debugging)
    list_audio_devices()
    
    # Start conversation
    try:
        await create_realtime_session(api_key, prompts)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == '__main__':
    print("\nStarting OpenAI Realtime Voice Agent...")
    print("Press Ctrl+C to exit\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        cleanup()
