# BakerMatcher - AI Voice Conversation Platform

A web-based application that enables real-time voice conversations with an AI assistant (BakerMatcher) using OpenAI's Realtime API. The platform includes a React frontend and a Flask backend server.

## Features

- **Real-time Voice Conversations**: Interactive voice conversations with OpenAI's Realtime API
- **Web Interface**: Modern React-based UI with video integration
- **Conversation Analysis**: AI-powered analysis of conversations with sentiment and insights
- **Configurable Prompts**: Customize the AI's personality and behavior via `prompt_config.json`
- **Multiple Voice Options**: Choose from various OpenAI voices

## Project Structure

```
BM25/
├── vapi_controller.py          # Flask backend server (main API)
├── openai_realtime_agent.py    # Standalone CLI voice agent (optional)
├── ui/                         # React frontend application
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   └── styles.css         # Styling
│   └── package.json           # Frontend dependencies
├── prompt_config.json         # AI personality and behavior configuration
├── openai_config.json         # OpenAI API key configuration
├── requirements.txt           # Python backend dependencies
└── README.md                  # This file
```

## Prerequisites

- Python 3.11+ with virtual environment support
- Node.js 16+ and npm
- OpenAI API key
- Microphone and speakers/headphones

## Setup

### 1. Install Python Dependencies

```bash
# Create and activate virtual environment (if not already created)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: `pyaudio` is only needed for the standalone CLI script (`openai_realtime_agent.py`). The web application doesn't require it.

**Installing PortAudio (required for pyaudio):**

- **On macOS:**
  ```bash
  brew install portaudio
  ```

- **On Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install portaudio19-dev
  ```

- **On other Linux distributions:**
  Install the PortAudio development package for your distribution.

### 2. Install Frontend Dependencies

```bash
cd ui
npm install
cd ..
```

### 3. Configure OpenAI API Key

Create `openai_config.json` in the project root:

```json
{
  "api_key": "your-openai-api-key-here"
}
```

Alternatively, you can set it as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 4. Configure Prompts (Optional)

The `prompt_config.json` file controls the AI's personality and behavior. It will be created automatically on first run with defaults, or you can create it manually:

```json
{
  "system_prompt": "You are BakerMatcher, an empathetic financial guide...",
  "voice": "nova",
  "temperature": 0.8,
  "max_tokens": 4096,
  "first_sentence": "Hi! I'm Baker Matcher, and I'm here to chat with you...",
  "instructions": "Be warm, engaging, and helpful..."
}
```

## Running the Application

### Main Web Application

The application consists of two components that need to run simultaneously:

#### 1. Start the Backend Server

**Option A: Run in foreground (for development/debugging)**

In one terminal:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start Flask server
python vapi_controller.py
```

**Option B: Run in background (recommended for production)**

Using the provided script:
```bash
./start_backend.sh
```

Or manually:
```bash
# Activate virtual environment
source venv/bin/activate

# Run in background with nohup
nohup python3 vapi_controller.py > log/backend.log 2>&1 &
```

To stop the background server:
```bash
./stop_backend.sh
```

Or manually find and kill the process:
```bash
# Find process on port 5000
lsof -ti:5000 | xargs kill
```

The server will start on `http://localhost:5000`

#### 2. Start the Frontend

In a second terminal:

```bash
cd ui
npm run dev
```

The frontend will start on `http://localhost:8801` (or another port if 5173 is busy)

#### 3. Access the Application

Open your browser and navigate to:
```
http://localhost:8801
```

### Usage Flow

1. Watch the video on the page
2. Click **"Give your thoughts"** button
3. Grant microphone permissions when prompted
4. Start speaking - the AI will respond after you finish
5. Click **"Analyze my thoughts"** to get conversation insights

## Standalone CLI Voice Agent (Optional)

The `openai_realtime_agent.py` script is a standalone command-line tool for voice conversations. It doesn't require the web interface and runs independently.

**Note**: This is separate from the main web application. You don't need to run this when using the web UI.

To use it:

```bash
# Activate virtual environment
source venv/bin/activate

# Run the standalone agent
python openai_realtime_agent.py
```

Press `Ctrl+C` to end the conversation.

## API Endpoints

The Flask backend (`vapi_controller.py`) provides the following endpoints:

- `POST /api/realtime-token` - Create a Realtime API session token
- `POST /api/analyze` - Analyze conversation messages
- `GET /status` - Server status
- `POST /api/chat` - Chat with AI (text-based)
- `POST /api/voice` - Voice interaction endpoint

## Configuration Files

### `openai_config.json`
Contains your OpenAI API key:
```json
{
  "api_key": "sk-..."
}
```

### `prompt_config.json`
Controls AI behavior:
- `system_prompt`: Defines the AI's role and personality
- `voice`: Voice selection (alloy, echo, fable, onyx, nova, shimmer, etc.)
- `temperature`: Controls randomness (0.0-2.0)
- `max_tokens`: Maximum response length
- `first_sentence`: Opening greeting
- `instructions`: Conversation guidelines

## Troubleshooting

### "Failed to fetch" or "ERR_CONNECTION_REFUSED"

- Make sure the Flask backend server (`vapi_controller.py`) is running on port 5000
- Check that no other process is using port 5000
- Verify the backend started successfully (check terminal output)

### Microphone Not Working

- Grant microphone permissions in your browser
- Check system microphone settings
- Try refreshing the page and granting permissions again

### AI Speaks Before Button Click

- This should be fixed in the current version
- The AI waits for you to click "Give your thoughts" before speaking
- If it still speaks early, check browser console for errors

### Port Already in Use

If port 5000 is in use:
```bash
# Find process using port 5000
lsof -ti:5000

# Kill the process (replace PID with actual process ID)
kill -9 PID
```

Or change the port in `vapi_controller.py` (line 797) and update `API_BASE_URL` in `ui/src/App.jsx` (line 3).

## Development

### Backend Development

The Flask server runs in debug mode by default. Logs are written to:
- `log/vapi_debug.log` - Server logs
- `log/calls.db` - Call logs database

### Frontend Development

The React app uses Vite for hot module replacement. Changes to `ui/src/` files will automatically reload in the browser.

## Environment Variables

You can use a `.env` file (loaded via `python-dotenv`) or set environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key
- `BM25_REALTIME_MODEL` - Realtime model (default: `gpt-4o-realtime-preview-2024-10-01`)
- `BM25_TTS_VOICE` - TTS voice (default: `alloy`)

## Notes

- The web application requires both the Flask backend and React frontend to be running
- The standalone `openai_realtime_agent.py` script is independent and doesn't need the web components
- All audio processing happens in the browser (WebRTC) - the backend only provides session tokens
- Conversation analysis uses OpenAI's chat API, not the Realtime API

## License

[Add your license information here]

