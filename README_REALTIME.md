# OpenAI Realtime Voice Agent

A self-sufficient Python script that uses OpenAI's Realtime API to have voice conversations on your laptop using your microphone and speakers.

## Features

- **Voice Conversations**: Real-time voice conversations with OpenAI's most human-like voices
- **Prompt Engineering**: Fully configurable prompts via `prompt_config.json`
- **Audio Input/Output**: Uses your laptop's microphone and speakers
- **Transcription**: Shows both your speech and AI responses in real-time
- **Customizable**: Configure voice, temperature, and conversation style

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_realtime.txt
```

**Note**: On macOS, you may need to install PortAudio first:
```bash
brew install portaudio
```

### 2. Configure OpenAI API Key

Create `openai_config.json`:
```json
{
  "api_key": "your-openai-api-key-here"
}
```

Or set environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Configure Prompts (Optional)

The script will create a default `prompt_config.json` on first run. You can customize:

- **system_prompt**: Defines the agent's role and personality
- **voice**: Choose from `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- **temperature**: Controls randomness (0.0-2.0)
- **instructions**: Specific conversation guidelines

Example:
```json
{
  "system_prompt": "You are a friendly customer service agent...",
  "voice": "nova",
  "temperature": 0.9,
  "instructions": "Be warm and helpful..."
}
```

## Usage

Run the script:
```bash
python openai_realtime_agent.py
```

The script will:
1. Show available audio devices
2. Connect to OpenAI Realtime API
3. Start listening to your microphone
4. Speak responses through your speakers
5. Display transcriptions in real-time

Press `Ctrl+C` to end the conversation.

## Audio Device Configuration

If you need to specify specific input/output devices, add to `openai_config.json`:
```json
{
  "api_key": "your-api-key",
  "input_device_index": 0,
  "output_device_index": 1
}
```

The script will list available devices on startup to help you find the right indices.

## Prompt Engineering Tips

### System Prompt
Define the agent's identity and purpose:
- "You are a helpful assistant..."
- "You are a friendly customer service agent..."
- "You are a knowledgeable teacher..."

### Instructions
Provide specific conversation guidelines:
- Tone and style
- Response length preferences
- Special behaviors or rules
- Context about the user or situation

### Voice Selection
- **alloy**: Balanced, neutral
- **echo**: Clear and professional
- **fable**: Expressive and engaging
- **onyx**: Deep and authoritative
- **nova**: Warm and friendly
- **shimmer**: Soft and gentle


