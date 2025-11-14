#!/usr/bin/env python3
"""
Vapi Webhook Controller - Self-sufficient Python script
Handles Vapi webhooks and conversation services for BakerMatcher.
"""

import base64
import json
import os
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List
import sqlite3

import httpx

from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from openai import OpenAI
from pytz import timezone

# Load environment variables as early as possible
load_dotenv()

# Configuration
LOG_DIR = Path("log")
LOG_FILE = LOG_DIR / "call_logs.json"
DEBUG_LOG = LOG_DIR / "vapi_debug.log"
DB_FILE = LOG_DIR / "calls.db"
PROMPT_CONFIG_FILE = Path("prompt_config.json")
VIDEO_CONFIG_FILE = Path("video_config.json")
REALTIME_MODEL = os.getenv("BM25_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-10-01")
TTS_MODEL = os.getenv("BM25_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("BM25_TTS_VOICE", "alloy")
TRANSCRIPTION_MODEL = os.getenv("BM25_TRANSCRIPTION_MODEL", "whisper-1")
TTS_AUDIO_FORMAT = os.getenv("BM25_TTS_FORMAT", "mp3")

# Create log directory if it doesn't exist
LOG_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(DEBUG_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# OpenAI configuration
_openai_client: Optional[OpenAI] = None
_session_create_fn: Optional[Any] = None
_prompt_config: Dict[str, Any] = {}
VIDEO_URL = "https://youtu.be/dyaG7zEtJ7U"
VIDEO_CONVERSATION_PRIMER = (
    "You are BakerMatcher, an empathetic financial guide discussing insights from the video "
    f"available here: {VIDEO_URL}. Encourage thoughtful reflection, focus on individuals "
    "finance takeaways, and ask clarifying questions before offering advice. "
    "You DO NOT elaborate or give your thoughts. Your role is to extract information "
    "from the user. Always acknowledge when you rely on the viewer's interpretation "
    "rather than the video's direct content. Keep responses concise, action-oriented, and supportive."
)
ANALYSIS_SYSTEM_PROMPT = (
    "You analyze conversations and provide insights. Given a transcript, return a JSON "
    "object with keys: summary (string), sentiment (string: positive/neutral/concerned), "
    "keyInsights (array of up to 4 short bullet strings), nextSteps (array of up to 3 "
    "actionable suggestions), and personalityScore (object with keys: openness (0-100), "
    "conscientiousness (0-100), extraversion (0-100), agreeableness (0-100), neuroticism (0-100), "
    "and overallScore (0-100)). Base insights solely on the transcript provided. "
    "Personality scores should reflect the user's communication style, engagement level, "
    "and responses during the conversation."
)

# Initialize database
def init_database():
    """Initialize SQLite database for call logs"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_free_phone_call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT,
            call_type TEXT,
            recording_url TEXT,
            status TEXT,
            req_host TEXT,
            inbound_info_message_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize on startup
init_database()


def get_openai_api_key() -> Optional[str]:
    """Retrieve the OpenAI API key from environment or configuration file."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    config_file = Path("openai_config.json")
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as cfg:
                config_data = json.load(cfg)
                api_key = config_data.get("api_key", "").strip()
                if api_key:
                    return api_key
        except Exception as exc:
            logger.error(f"Error loading OpenAI config: {exc}")
    return None


def get_openai_client() -> Optional[OpenAI]:
    """Lazy-initialize the OpenAI client."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    api_key = get_openai_api_key()
    if not api_key:
        logger.warning("OpenAI API key missing; /api/chat and /api/analyze will return an error")
        return None

    try:
        _openai_client = OpenAI(api_key=api_key)
    except Exception as exc:
        logger.error(f"Unable to initialize OpenAI client: {exc}")
        _openai_client = None
    return _openai_client


def load_prompt_config() -> Dict[str, Any]:
    global _prompt_config
    if _prompt_config:
        return _prompt_config

    system_prompt = VIDEO_CONVERSATION_PRIMER
    instructions = ""
    voice = TTS_VOICE
    temperature = 0.7
    first_sentence = "Hi there, I'm BakerMatcher."
    max_tokens = 4096

    if PROMPT_CONFIG_FILE.exists():
        try:
            with open(PROMPT_CONFIG_FILE, "r", encoding="utf-8") as cfg:
                loaded = json.load(cfg)
                if isinstance(loaded, dict):
                    system_prompt = loaded.get("system_prompt", system_prompt)
                    instructions = loaded.get("instructions", instructions)
                    voice = loaded.get("voice", voice)
                    temperature = loaded.get("temperature", temperature)
                    first_sentence = loaded.get("first_sentence", first_sentence)
                    max_tokens = loaded.get("max_tokens", max_tokens)
        except Exception as exc:
            logger.error(f"Error loading prompt config: {exc}")

    instruction_parts = [system_prompt]
    if instructions:
        instruction_parts.append(instructions)
    if first_sentence:
        instruction_parts.append(
            f"IMPORTANT: When the conversation begins, your very first spoken sentence must be exactly:\n\"{first_sentence}\"\nDo not change the wording, and do not introduce yourself in any other way."
        )
    combined_instructions = "\n\n".join(instruction_parts)

    _prompt_config = {
        "system_prompt": system_prompt,
        "instructions": instructions,
        "combined_instructions": combined_instructions,
        "voice": voice,
        "temperature": temperature,
        "first_sentence": first_sentence,
        "max_tokens": max_tokens
    }
    return _prompt_config


def load_video_config(video_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load video configuration from video_config.json."""
    if not VIDEO_CONFIG_FILE.exists():
        logger.warning(f"Video config file not found: {VIDEO_CONFIG_FILE}")
        return None

    try:
        with open(VIDEO_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            videos = config.get("videos", [])

            if video_id:
                # Find specific video by ID
                for video in videos:
                    if video.get("id") == video_id:
                        return video
                logger.warning(f"Video ID '{video_id}' not found in config")
                return None

            # Return first video if no ID specified
            return videos[0] if videos else None
    except Exception as exc:
        logger.error(f"Error loading video config: {exc}")
        return None


def build_topic_enforcement_instructions(video_config: Dict[str, Any], prompt_config: Dict[str, Any]) -> tuple:
    """Build instructions that include topic enforcement and script guidance.
    Returns: (first_sentence, instructions)"""
    topic = video_config.get("topic", "")
    script = video_config.get("script", [])
    followup_questions = video_config.get("followup_questions", [])

    # Build first sentence with topic
    first_sentence = f"Hi! I'm Baker Matcher, and I'm here to chat with you about the video on {topic} that you've watched a few minutes ago. What are your thoughts about what was presented in the video? Let us have an open interactive conversation. Share your thoughts openly and I will patiently listen and engage with you"

    topic_enforcement = prompt_config.get("topic_enforcement",
        "If the user goes off-topic (e.g., talking about movies when the topic is Finance), "
        "politely redirect them: 'I appreciate you sharing that, but let's stay focused on [TOPIC]. "
        "How does that relate to [TOPIC]?' Be gentle and conversational, not rigid.")

    # Topic enforcement
    strict_topic_enforcement = f"""
CRITICAL: TOPIC ENFORCEMENT - YOU MUST ENFORCE THIS STRICTLY

The conversation topic is: {topic}. The ENTIRE conversation must stay on {topic}.

If the user tries to talk about ANYTHING else (movies, Hollywood, sports, unrelated topics, etc.), you MUST immediately and firmly redirect them:

EXAMPLES:
- User: "Hollywood movies"
  You: "I appreciate you sharing that, but we're here to discuss {topic}. Let's stay focused on {topic}. What are your thoughts on {topic}?"

- User: "I want to talk about movies"
  You: "That's interesting, but we're discussing {topic} today. Can you tell me more about your thoughts on {topic}?"

- User mentions any unrelated topic
  You: "I understand, but let's keep our conversation about {topic}. What's your perspective on {topic}?"

DO NOT allow the conversation to drift to other topics. DO NOT discuss movies, Hollywood, or any unrelated topics. 
Gently but FIRMLY redirect them back to {topic} immediately. This is non-negotiable.
"""

    # Build follow-up questions list as a clear numbered list
    followup_list = "\n".join([f"QUESTION {i+1}: {q}" for i, q in enumerate(followup_questions)])
    
    script_guidance = (
        "CRITICAL RULES:\n\n"
        "1. DO NOT ELABORATE: After user speaks, give 1-2 word acknowledgment ('Interesting,' 'I see'), then ask next question. Keep responses under 15 words.\n\n"
        "2. DO NOT EXPLAIN VIDEO: Your role is to EXTRACT information, not provide it.\n\n"
        "3. CONVERSATION FLOW:\n\n"
        "   STEP 1: After your first sentence, user shares thoughts. Give brief acknowledgment, then ask: 'Do you want to discuss something else in the video you watched?'\n\n"
        "   STEP 2: If user says YES, let them share more, then ask 'Do you want to discuss something else in the video you watched?' again. Repeat until they say NO.\n\n"
        "   STEP 3 - CRITICAL TRANSITION TO FOLLOW-UP QUESTIONS:\n"
        "   - After asking 'Do you want to discuss something else in the video you watched?', watch for these responses:\n"
        "     * Direct: 'no,' 'not really,' 'that's all,' 'I'm done,' 'nothing else'\n"
        "     * Indirect: 'I don't have anything else to discuss,' 'I don't have much to talk about,' 'nothing more to say,' 'that's everything,' 'I think that's it'\n"
        "     * Any response that indicates the user is finished discussing the video\n"
        "   - WHEN YOU HEAR ANY OF THESE, you MUST IMMEDIATELY transition to asking follow-up questions\n"
        "   - Do NOT ask 'Do you want to discuss something else' again\n"
        "   - Do NOT wait, pause, or ask anything else\n"
        "   - IMMEDIATELY say: 'That's an interesting point, but how about [read QUESTION 1 exactly as written below]?'\n"
        "   - Read the question VERBATIM - do not rephrase it\n"
        "   - After user answers, say: 'Interesting. That's an interesting point, but how about [read QUESTION 2 exactly as written]?'\n"
        "   - Continue this pattern: read QUESTION 3, then QUESTION 4, then QUESTION 5, etc.\n"
        "   - You MUST ask ALL {len(followup_questions)} questions in order\n"
        "   - Read each question EXACTLY as it appears below - verbatim, word for word\n"
        "   - After each answer, brief acknowledgment, then next question\n"
        "   - Do NOT stop until all questions are asked\n"
        "   - CRITICAL: If user says 'I don't have much to talk about' or similar, this means START asking follow-up questions NOW\n\n"
        "4. RESPONSE EXAMPLES:\n"
        "   - 'I see. Do you want to discuss something else in the video you watched?'\n"
        "   - When user says 'no': 'That's an interesting point, but how about What does financial security mean to you personally?'\n"
        "   - After answer: 'Interesting. That's an interesting point, but how about How do you typically make decisions about spending money?'\n\n"
        "5. NEVER: Elaborate, explain, give opinions, go quiet, stop early, or skip follow-up questions\n\n"
        "6. ALWAYS: Be proactive, ask next question immediately, read questions verbatim, continue through all questions"
    ).replace('{topic}', topic)

    instructions = f"""
{script_guidance}

{strict_topic_enforcement}

FOLLOW-UP QUESTIONS - READ THESE VERBATIM AFTER USER SAYS 'NO' OR 'NOT REALLY':

{followup_list}

CRITICAL INSTRUCTIONS - READ CAREFULLY:

TRIGGER DETECTION - When user is done discussing the video, IMMEDIATELY start asking follow-up questions:
- Direct: 'no,' 'not really,' 'that's all,' 'I'm done,' 'nothing else'
- Indirect: 'I don't have anything else to discuss,' 'I don't have much to talk about,' 'nothing more to say,' 'that's everything,' 'I think that's it'
- ANY response that indicates they're finished = START asking follow-up questions NOW

IMMEDIATE ACTION REQUIRED:
When you detect the trigger, you MUST:
1. Say: 'That's an interesting point, but how about [read QUESTION 1 above exactly as written]?'
2. Wait for answer
3. Say: 'Interesting. That's an interesting point, but how about [read QUESTION 2 above exactly as written]?'
4. Continue through ALL {len(followup_questions)} questions in order

READ QUESTIONS VERBATIM - word for word, do not rephrase

ASK IN ORDER: QUESTION 1 → QUESTION 2 → QUESTION 3 → QUESTION 4 → QUESTION 5 → QUESTION 6 → QUESTION 7 → QUESTION 8

FORMAT: Always use 'That's an interesting point, but how about [question text]?' where [question text] is EXACTLY as written above

AFTER EACH ANSWER: Brief acknowledgment ('Interesting,' 'I see'), then immediately next question

DO NOT STOP: You MUST ask all {len(followup_questions)} questions - continue until all are asked

EXAMPLE TRANSITIONS - COPY THIS EXACT BEHAVIOR:
User: 'not really'
You: 'That's an interesting point, but how about What does financial security mean to you personally?'

User: 'I don't have much to talk about'
You: 'That's an interesting point, but how about What does financial security mean to you personally?'

User: 'I don't have anything else to discuss'
You: 'That's an interesting point, but how about What does financial security mean to you personally?'

[After user answers]
You: 'Interesting. That's an interesting point, but how about How do you typically make decisions about spending money?'
[Continue with remaining questions...]
"""

    return first_sentence, instructions


def get_realtime_sessions() -> Optional[Any]:
    """Return a callable that can create OpenAI Realtime sessions."""
    global _session_create_fn
    if _session_create_fn is not None:
        return _session_create_fn

    client = get_openai_client()
    if client is None:
        return None

    try:
        # Try the helper if available (newer SDKs)
        realtime = getattr(client, "realtime", None)
        if realtime is not None and hasattr(realtime, "sessions"):
            sessions = realtime.sessions
            if hasattr(sessions, "create"):
                _session_create_fn = sessions.create
                return _session_create_fn

        # Fallback to the generic request method
        def _create_session(**kwargs):
            body = {
                "model": kwargs.get("model", REALTIME_MODEL),
                "voice": kwargs.get("voice", TTS_VOICE),
                "modalities": kwargs.get("modalities", ["audio", "text"]),
                "instructions": kwargs.get("instructions", VIDEO_CONVERSATION_PRIMER),
                "input_audio_format": kwargs.get("input_audio_format", "pcm16"),
                "output_audio_format": kwargs.get("output_audio_format", "pcm16"),
                "turn_detection": kwargs.get("turn_detection", {
                    "type": "server_vad",
                    "threshold": 0.4,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 600
                })
            }
            try:
                api_key = get_openai_api_key()
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is not configured.")

                response = httpx.post(
                    "https://api.openai.com/v1/realtime/sessions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=body,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                raise RuntimeError("Unable to create realtime session via REST fallback.") from exc

        _session_create_fn = _create_session
        return _session_create_fn
    except Exception as exc:
        logger.error(f"Unable to initialize OpenAI Realtime sessions: {exc}")
        _session_create_fn = None
        return None


def _sanitize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    """Filter inbound message payloads to valid chat messages."""
    sanitized: List[Dict[str, str]] = []
    if not isinstance(raw_messages, list):
        return sanitized

    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        sanitized.append({"role": role, "content": content.strip()})
    return sanitized


def _extract_openai_text(message: Any) -> str:
    """Normalize OpenAI message content to plain text."""
    if message is None:
        return ""

    content = None
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_value = part.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "".join(text_parts).strip()

    return ""


def _build_cors_preflight_response() -> Any:
    response = make_response("", 204)
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@app.after_request
def _add_cors_headers(response):
    response.headers.setdefault("Access-Control-Allow-Origin", request.headers.get("Origin", "*"))
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    response.headers.setdefault("Access-Control-Max-Age", "86400")
    return response


def _build_audio_payload(audio_bytes: bytes, mime_type: str) -> Dict[str, str]:
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    return {"base64": encoded, "mime_type": mime_type}


def synthesize_speech(text: str) -> Optional[Dict[str, str]]:
    """Generate speech audio for the provided text using OpenAI TTS."""
    if not text or not text.strip():
        return None

    prompt_config = load_prompt_config()
    client = get_openai_client()
    if client is None:
        return None

    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=prompt_config.get("voice", TTS_VOICE),
            input=text,
            format=TTS_AUDIO_FORMAT
        )

        audio_bytes = b""
        if hasattr(response, "read"):
            audio_bytes = response.read()
        elif hasattr(response, "stream"):
            audio_bytes = b"".join(response.stream)
        elif hasattr(response, "content") and isinstance(response.content, (bytes, bytearray)):
            audio_bytes = bytes(response.content)
        elif hasattr(response, "audio") and isinstance(response.audio, (bytes, bytearray)):
            audio_bytes = bytes(response.audio)

        if not audio_bytes:
            logger.warning("Speech synthesis returned empty audio stream")
            return None

        mime_type = "audio/mpeg" if TTS_AUDIO_FORMAT == "mp3" else f"audio/{TTS_AUDIO_FORMAT}"
        return _build_audio_payload(audio_bytes, mime_type)
    except Exception as exc:
        logger.error(f"Speech synthesis error: {exc}")
        return None


def transcribe_audio(file_storage) -> Optional[str]:
    """Transcribe uploaded audio using OpenAI Whisper."""
    if file_storage is None:
        return None

    client = get_openai_client()
    if client is None:
        return None

    try:
        audio_bytes = file_storage.read()
        if not audio_bytes:
            return None

        audio_buffer = BytesIO(audio_bytes)
        audio_buffer.name = file_storage.filename or "input.webm"

        transcription = client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio_buffer
        )
        text = getattr(transcription, "text", "")
        return text.strip()
    except Exception as exc:
        logger.error(f"Transcription error: {exc}")
        return None


def generate_chat_reply(messages: List[Dict[str, str]]) -> Optional[str]:
    """Generate chat completion based on existing messages."""
    client = get_openai_client()
    if client is None:
        return None

    prompt_config = load_prompt_config()
    limited_history = messages[-12:]
    chat_messages = [{'role': 'system', 'content': prompt_config.get("combined_instructions", VIDEO_CONVERSATION_PRIMER)}] + limited_history
    model = os.getenv('BM25_CHAT_MODEL', 'gpt-4o-mini')

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            temperature=0.7,
            max_tokens=512,
        )
        choice = completion.choices[0].message if completion.choices else None
        reply_text = _extract_openai_text(choice)
        return reply_text or "I'm thinking about that. Could you rephrase your insight?"
    except Exception as exc:
        logger.error(f"Chat completion error: {exc}")
        return None


def load_logs() -> list:
    """Load call logs from JSON file"""
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading logs: {e}")
        return []


def save_logs(logs: list):
    """Save call logs to JSON file"""
    try:
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving logs: {e}")


def store_call_in_db(phone_number: str, call_type: str, recording_url: str, 
                     status: str, req_host: str) -> Optional[int]:
    """Store call information in database. Returns the call log ID."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ai_free_phone_call_logs 
            (phone_number, call_type, recording_url, status, req_host)
            VALUES (?, ?, ?, ?, ?)
        ''', (phone_number, call_type, recording_url, status, req_host))
        call_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return call_id
    except Exception as e:
        logger.error(f"Database error: {e}")
        return None



@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Vapi Webhook Controller',
        'status': 'running',
        'endpoints': {
            'POST /webhook': 'Receive Vapi webhook events',
            'GET /logs': 'Get all call logs (JSON)',
            'GET /logs/view': 'View logs in HTML format',
            'GET /status': 'Get server status'
        }
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle Vapi webhook requests"""
    logger.info("=== Received webhook request ===")
    logger.info(f"Headers: {dict(request.headers)}")
    
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        print("-------payload---------")
        event_type = payload.get('message', {}).get('type') or payload.get('type')
        print(f"Event type: {event_type}")
        print("---------------")
        print("********************************************************")
        logger.info(f"Event type: {event_type}")
        print("********************************************************")
        
        # Create log entry
        chicago_tz = timezone('America/Chicago')
        timestamp = datetime.now(chicago_tz).strftime('%Y-%m-%d %I:%M:%S %p %Z')
        
        log_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'data': payload,
            'ip': request.remote_ip
        }
        
        # Save to logs
        call_logs = load_logs()
        call_logs.append(log_entry)
        save_logs(call_logs)
        
        # Handle end-of-call-report
        if event_type == 'end-of-call-report':
            call_data = payload.get('message', {})
            
            # Store in database
            phone_number = call_data.get('customer', {}).get('number', '')
            recording_url = call_data.get('recordingUrl', '')
            status = call_data.get('status', 'completed')
            req_host = request.host
            
            call_id = store_call_in_db(
                phone_number=phone_number,
                call_type='Inbound',
                recording_url=recording_url,
                status=status,
                req_host=req_host
            )
            
            if call_id:
                logger.info(f"Stored end-of-call-report in database for call_id={call_data.get('callId')}")
        
        # Handle call.received event
        if event_type == 'call.received':
            return jsonify({
                'assistantId': 'c2eb2fbf-23fa-4a9c-a6cc-0fcae52b3faa'
            })
        else:
            return jsonify({'status': 'processed'})
            
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/logs', methods=['GET'])
def logs():
    """Get all call logs"""
    return jsonify(load_logs())


@app.route('/logs/view', methods=['GET'])
def logs_view():
    """View logs in HTML format"""
    logs = load_logs()
    logs.reverse()
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vapi Call Logs</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            pre { max-width: 500px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>Vapi Call Logs</h1>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Event Type</th>
                <th>IP Address</th>
                <th>Data</th>
            </tr>
    """
    
    for log in logs:
        html += f"""
            <tr>
                <td>{log.get('timestamp', 'N/A')}</td>
                <td>{log.get('event_type', 'N/A')}</td>
                <td>{log.get('ip', 'N/A')}</td>
                <td><pre>{json.dumps(log.get('data', {}), indent=2)}</pre></td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    return html


@app.route('/status', methods=['GET'])
def status():
    """Get server status"""
    logs = load_logs()
    chicago_tz = timezone('America/Chicago')
    server_time = datetime.now(chicago_tz).isoformat()
    
    return jsonify({
        'status': 'running',
        'total_calls': len(logs),
        'last_call': logs[-1] if logs else None,
        'server_time': server_time
    })


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat_with_ai():
    """Handle chat messages coming from the React UI."""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    payload = request.get_json(silent=True) or {}
    messages = _sanitize_messages(payload.get('messages'))

    if not messages:
        return jsonify({'error': 'At least one user message is required.'}), 400

    reply_text = generate_chat_reply(messages)
    if reply_text is None:
        return jsonify({'error': 'Unable to generate a response right now. Please try again.'}), 500

    audio_payload = synthesize_speech(reply_text)

    return jsonify({'reply': reply_text, 'audio': audio_payload})


@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze_conversation():
    """Provide a finance-focused summary of the conversation."""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    payload = request.get_json(silent=True) or {}
    messages = _sanitize_messages(payload.get('messages'))
    if not messages:
        return jsonify({'error': 'Conversation history is required for analysis.'}), 400

    client = get_openai_client()
    if client is None:
        return jsonify({'error': 'OpenAI API key is not configured.'}), 500

    transcript_lines = []
    for message in messages[-20:]:
        speaker = 'User' if message['role'] == 'user' else 'AI'
        transcript_lines.append(f"{speaker}: {message['content']}")
    transcript = "\n".join(transcript_lines)

    model = os.getenv('BM25_ANALYSIS_MODEL', os.getenv('BM25_CHAT_MODEL', 'gpt-4o-mini'))

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': ANALYSIS_SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Conversation transcript:\n{transcript}"}
            ],
            temperature=0.4,
            max_tokens=512,
        )
        choice = completion.choices[0].message if completion.choices else None
        raw_text = _extract_openai_text(choice)
        analysis_payload: Dict[str, Any]
        try:
            analysis_payload = json.loads(raw_text)
        except Exception:
            analysis_payload = {
                'summary': raw_text or 'Unable to analyze the conversation at this time.',
                'sentiment': 'unknown',
                'keyInsights': [],
                'nextSteps': [],
                'personalityScore': {
                    'openness': 50,
                    'conscientiousness': 50,
                    'extraversion': 50,
                    'agreeableness': 50,
                    'neuroticism': 50,
                    'overallScore': 50
                }
            }
        return jsonify({'analysis': analysis_payload})
    except Exception as exc:
        logger.error(f"Analysis error: {exc}")
        return jsonify({'error': 'Unable to analyze the conversation right now. Please try again.'}), 500


@app.route('/api/realtime-token', methods=['POST', 'OPTIONS'])
def create_realtime_token():
    """Create an ephemeral token for establishing a Realtime WebRTC session."""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    prompt_config = load_prompt_config()
    session_create = get_realtime_sessions()
    if session_create is None:
        return jsonify({'error': 'OpenAI Realtime is not configured.'}), 500

    request_data = request.get_json(silent=True) or {}
    requested_voice = (request_data.get('voice') or '').strip() or prompt_config.get("voice", TTS_VOICE)
    requested_model = (request_data.get('model') or '').strip() or REALTIME_MODEL
    video_id = request_data.get('video_id')  # Get video_id from request

    # Load video config if video_id is provided
    video_config = None
    instructions = prompt_config.get("combined_instructions", VIDEO_CONVERSATION_PRIMER)

    video_first_sentence = None
    if video_id:
        video_config = load_video_config(video_id)
        if video_config:
            # Build enhanced instructions with topic enforcement and script guidance
            video_first_sentence, topic_instructions = build_topic_enforcement_instructions(video_config, prompt_config)
            # Put topic enforcement and follow-up question instructions FIRST for highest priority
            # Then system prompt, then other instructions
            instructions = f"{topic_instructions}\n\n{prompt_config.get('system_prompt', '')}\n\n{prompt_config.get('instructions', '')}"
        else:
            logger.warning(f"Video config not found for video_id: {video_id}, using default instructions")

    try:
        session = session_create(
            model=requested_model,
            voice=requested_voice,
            modalities=["audio", "text"],
            instructions=instructions,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            input_audio_transcription={
                "model": "whisper-1"
            },
            turn_detection={
                "type": "server_vad",
                "threshold": 0.4,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 600
            }
        )

        client_secret = session.get("client_secret") or {}
        secret_value = client_secret.get("value")
        if not secret_value:
            logger.error("Realtime session created without client_secret")
            return jsonify({'error': 'Unable to issue realtime token.'}), 500

        # Use video-specific first sentence if available, otherwise use default
        first_sentence_to_use = video_first_sentence if video_first_sentence else prompt_config.get("first_sentence")

        response_data = {
            'clientSecret': secret_value,
            'expiresAt': client_secret.get('expires_at'),
            'session': {
                'id': session.get('id'),
                'model': session.get('model', requested_model),
                'voice': session.get('voice', requested_voice),
                'instructions': instructions[:200] + '...' if len(instructions) > 200 else instructions,
                'firstSentence': first_sentence_to_use,
                'temperature': prompt_config.get("temperature"),
                'maxTokens': prompt_config.get("max_tokens")
            }
        }

        # Include video config if available
        if video_config:
            response_data['video'] = {
                'id': video_config.get('id'),
                'title': video_config.get('title'),
                'video_url': video_config.get('video_url'),
                'topic': video_config.get('topic')
            }

        return jsonify(response_data)
    except Exception as exc:
        logger.error(f"Realtime token error: {exc}")
        return jsonify({'error': 'Unable to create realtime session token.'}), 500


@app.route('/api/voice', methods=['POST', 'OPTIONS'])
def handle_voice_interaction():
    """Accept audio input, transcribe, and respond with synthesized speech."""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({'error': 'Audio file is required.'}), 400

    messages_str = request.form.get('messages', '[]')
    try:
        raw_messages = json.loads(messages_str)
    except json.JSONDecodeError:
        raw_messages = []

    messages = _sanitize_messages(raw_messages)

    transcript = transcribe_audio(audio_file)
    if not transcript:
        return jsonify({'error': 'Unable to transcribe the audio.'}), 422

    updated_messages = messages + [{"role": "user", "content": transcript}]
    reply_text = generate_chat_reply(updated_messages)
    if reply_text is None:
        return jsonify({'error': 'Unable to generate a response right now. Please try again.'}), 500

    audio_payload = synthesize_speech(reply_text)

    return jsonify({
        'transcript': transcript,
        'reply': reply_text,
        'audio': audio_payload
    })


@app.route('/api/speak', methods=['POST', 'OPTIONS'])
def speak_text():
    """Synthesize speech for arbitrary text."""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    payload = request.get_json(silent=True) or {}
    text = (payload.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Text is required.'}), 400

    audio_payload = synthesize_speech(text)
    if audio_payload is None:
        return jsonify({'error': 'Unable to synthesize audio at the moment.'}), 500

    return jsonify({'audio': audio_payload})


if __name__ == '__main__':
    print("Starting Vapi Webhook Controller...")
    print(f"Log directory: {LOG_DIR.absolute()}")
    print(f"Database: {DB_FILE.absolute()}")
    print("\nAvailable endpoints:")
    print("  POST /webhook - Receive Vapi webhooks")
    print("  GET  /logs - Get all logs (JSON)")
    print("  GET  /logs/view - View logs (HTML)")
    print("  GET  /status - Get server status")
    print("\nConfiguration:")
    print("  HOST - Your application host URL (default: http://localhost:5000)")
    print("\nStarting server on http://localhost:5000\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

