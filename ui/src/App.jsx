import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000';
const VIDEO_URL = 'https://youtu.be/dyaG7zEtJ7U';
const ASSISTANT_NAME = 'BakerMatcher';
const DEFAULT_FIRST_SENTENCE = "Hi! I'm Baker Matcher, and I'm here to chat with you about the videos you've watched a few minutes ago. What did you watch today, or what's on your mind? Let us have an open interactive conversation. Speak whatever is on your mind and I will patiently listen and engage with you";

function Message({ role, content }) {
  return (
    <div
      className={`message message-${role}`}
      aria-label={`${role === 'assistant' ? ASSISTANT_NAME : 'You'} message`}
    >
      <div className="message-meta">{role === 'assistant' ? ASSISTANT_NAME : 'You'}</div>
      <div className="message-content">{content}</div>
    </div>
  );
}

function AnalysisCard({ analysis }) {
  if (!analysis) return null;

  const keyInsights = Array.isArray(analysis.keyInsights) ? analysis.keyInsights : [];
  const nextSteps = Array.isArray(analysis.nextSteps) ? analysis.nextSteps : [];

  return (
    <section className="analysis-card" aria-live="polite">
      <h3>Finance Reflection</h3>
      <p className="analysis-summary">{analysis.summary}</p>
      <div className="analysis-columns">
        <div>
          <strong>Sentiment</strong>
          <p className="analysis-tag">{analysis.sentiment ?? 'neutral'}</p>
        </div>
        {keyInsights.length > 0 && (
          <div>
            <strong>Key Insights</strong>
            <ul>
              {keyInsights.map((item, index) => (
                <li key={`insight-${index}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {nextSteps.length > 0 && (
          <div>
            <strong>Next Steps</strong>
            <ul>
              {nextSteps.map((item, index) => (
                <li key={`step-${index}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

function App() {
  const [hasStarted, setHasStarted] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [liveAssistantText, setLiveAssistantText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [micPermissionGranted, setMicPermissionGranted] = useState(true);

  const peerConnectionRef = useRef(null);
  const dataChannelRef = useRef(null);
  const localStreamRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const messagesRef = useRef(messages);
  const assistantBufferRef = useRef('');
  const userBufferRef = useRef('');

  const embedUrl = useMemo(() => {
    if (VIDEO_URL.includes('youtu.be/')) {
      const [, videoIdWithParams = ''] = VIDEO_URL.split('youtu.be/');
      const [videoId] = videoIdWithParams.split('?');
      return `https://www.youtube.com/embed/${videoId}`;
    }
    if (VIDEO_URL.includes('watch?v=')) {
      return VIDEO_URL.replace('watch?v=', 'embed/');
    }
    return VIDEO_URL;
  }, []);

  const hasUserMessages = useMemo(() => {
    if (messages.some((message) => message.role === 'user')) {
      return true;
    }
    messagesRef.current = messages;
    return messagesRef.current.some((message) => message.role === 'user');
  }, [messages]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const stopRealtime = useCallback(() => {
    setIsConnecting(false);
    setIsConnected(false);
    assistantBufferRef.current = '';
    userBufferRef.current = '';
    setLiveAssistantText('');

    if (dataChannelRef.current) {
      try {
        dataChannelRef.current.close();
      } catch (err) {
        console.debug('Data channel close error', err);
      }
      dataChannelRef.current = null;
    }

    if (peerConnectionRef.current) {
      try {
        peerConnectionRef.current.getSenders().forEach((sender) => {
          if (sender.track) sender.track.stop();
        });
        peerConnectionRef.current.close();
      } catch (err) {
        console.debug('Peer connection close error', err);
      }
      peerConnectionRef.current = null;
    }

    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }

    if (remoteAudioRef.current) {
      remoteAudioRef.current.pause();
      remoteAudioRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => stopRealtime, [stopRealtime]);

  const handleRealtimeEvent = useCallback((event) => {
    if (!event || !event.type) {
      return;
    }

    const extractText = (item) => {
      if (!item || !Array.isArray(item.content)) return '';
      return item.content
        .filter((part) => part?.type === 'text')
        .map((part) => part.text || '')
        .join('')
        .trim();
    };

    switch (event.type) {
      case 'response.output_text.delta': {
        assistantBufferRef.current += event.delta || '';
        setLiveAssistantText(assistantBufferRef.current);
        break;
      }
      case 'response.output_text.done': {
        const text = assistantBufferRef.current.trim();
        assistantBufferRef.current = '';
        setLiveAssistantText('');
        if (text) {
          setMessages((prev) => [...prev, { role: 'assistant', content: text }]);
        }
        break;
      }
      case 'conversation.item.input_audio_transcription.delta': {
        userBufferRef.current += event.delta || '';
        break;
      }
      case 'conversation.item.input_audio_transcription.completed': {
        const transcript = (event.transcript || userBufferRef.current).trim();
        userBufferRef.current = '';
        if (transcript) {
          setMessages((prev) => [...prev, { role: 'user', content: transcript }]);
        }
        break;
      }
      case 'conversation.item.completed': {
        const item = event.item;
        if (item?.type === 'message' && item.role === 'assistant') {
          const textParts = [];
          (item.content || []).forEach((part) => {
            if (part.type === 'output_text' && part.text) {
              textParts.push(part.text);
            }
            if (part.type === 'text' && part.text) {
               textParts.push(part.text);
            }
          });
          const text = textParts.join('').trim();
          if (text) {
            setMessages((prev) => [...prev, { role: 'assistant', content: text }]);
            assistantBufferRef.current = '';
            setLiveAssistantText('');
          }
        }
        break;
      }
      case 'conversation.item.created': {
        const item = event.item;
        if (item?.type === 'message') {
          const text = extractText(item);
          if (text) {
            if (item.role === 'assistant') {
              setMessages((prev) => [...prev, { role: 'assistant', content: text }]);
            }
            if (item.role === 'user') {
              setMessages((prev) => [...prev, { role: 'user', content: text }]);
            }
          }
        }
        break;
      }
      case 'response.error':
      case 'error': {
        const message = event.error?.message || event.message || 'Realtime session error.';
        setError(message);
        break;
      }
      default:
        break;
    }
  }, [setMessages]);

  const startRealtime = useCallback(async () => {
    if (isConnecting || isConnected) {
      return;
    }

    setError(null);
    setIsConnecting(true);

    let localStream;
    try {
      localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setMicPermissionGranted(true);
    } catch (err) {
      console.error(err);
      setMicPermissionGranted(false);
      setError('Microphone access denied. Enable mic permissions and try again.');
      setIsConnecting(false);
      return;
    }

    try {
      const tokenResponse = await fetch(`${API_BASE_URL}/api/realtime-token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (!tokenResponse.ok) {
        throw new Error('Unable to obtain realtime session token.');
      }
      const tokenData = await tokenResponse.json();
      const clientSecret = tokenData.clientSecret;
      if (!clientSecret) {
        throw new Error('Realtime token is missing a client secret.');
      }
      const model = tokenData.session?.model || 'gpt-4o-realtime-preview-2024-10-01';
      const voice = tokenData.session?.voice;
      const instructions = tokenData.session?.instructions;
      const firstSentence = tokenData.session?.firstSentence?.trim() || DEFAULT_FIRST_SENTENCE;
      const temperature = tokenData.session?.temperature;
      const maxTokens = tokenData.session?.maxTokens;

      const pc = new RTCPeerConnection();
      peerConnectionRef.current = pc;
      localStreamRef.current = localStream;

      localStream.getTracks().forEach((track) => pc.addTrack(track, localStream));

      pc.ontrack = (event) => {
        if (!remoteAudioRef.current) return;
        const [remoteStream] = event.streams;
        if (remoteStream) {
          remoteAudioRef.current.srcObject = remoteStream;
          const playPromise = remoteAudioRef.current.play();
          if (playPromise) {
            playPromise.catch((err) => console.debug('Autoplay blocked until interaction', err));
          }
        }
      };

      pc.onconnectionstatechange = () => {
        if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
          stopRealtime();
        }
      };

      const dc = pc.createDataChannel('oai-events');
      dataChannelRef.current = dc;

      dc.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          handleRealtimeEvent(payload);
        } catch (err) {
          console.error('Realtime payload parse error', err, event.data);
        }
      };

      dc.onclose = () => {
        stopRealtime();
      };

      dc.onopen = () => {
        const sessionUpdate = {
          type: 'session.update',
          session: {
            instructions: instructions || VIDEO_CONVERSATION_PRIMER,
            voice,
            modalities: ['audio', 'text'],
            input_audio_format: 'pcm16',
            output_audio_format: 'pcm16',
            temperature: temperature,
            max_response_output_tokens: maxTokens
          }
        };
        dc.send(JSON.stringify(sessionUpdate));
        // Send first sentence only after button is clicked and connection is established
        // This ensures AI speaks after user interaction, not before
        if (firstSentence) {
          // Small delay to ensure session is fully ready
          setTimeout(() => {
            if (dataChannelRef.current && dataChannelRef.current.readyState === 'open') {
              dataChannelRef.current.send(
                JSON.stringify({
                  type: 'response.create',
                  response: {
                    modalities: ['audio', 'text'],
                    instructions: `Say exactly the following sentence and nothing else: ${firstSentence}`,
                    conversation: 'none'
                  }
                })
              );
            }
          }, 500);
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const waitForIceGatheringComplete = () => new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') {
          resolve();
        } else {
          const checkState = () => {
            if (pc.iceGatheringState === 'complete') {
              pc.removeEventListener('icegatheringstatechange', checkState);
              resolve();
            }
          };
          pc.addEventListener('icegatheringstatechange', checkState);
        }
      });

      await waitForIceGatheringComplete();

      const answerResponse = await fetch(`https://api.openai.com/v1/realtime?model=${encodeURIComponent(model)}`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${clientSecret}`,
          'Content-Type': 'application/sdp'
        },
        body: pc.localDescription?.sdp || ''
      });

      if (!answerResponse.ok) {
        throw new Error('Realtime handshake failed.');
      }

      const answer = await answerResponse.text();
      await pc.setRemoteDescription({ type: 'answer', sdp: answer });

      setIsConnected(true);
      setIsConnecting(false);
    } catch (err) {
      console.error(err);
      setError(err.message ?? 'Unable to start realtime session.');
      stopRealtime();
    }
  }, [API_BASE_URL, handleRealtimeEvent, isConnected, isConnecting, stopRealtime]);

  const handleStartConversation = async () => {
    if (isConnecting || isConnected) {
      return;
    }
    setHasStarted(true);
    setAnalysis(null);
    setError(null);
    setMessages([]);
    messagesRef.current = [];
    await startRealtime();
  };

  const handleToggleRealtime = async () => {
    if (isConnecting) {
      return;
    }
    if (isConnected) {
      stopRealtime();
    } else {
      await startRealtime();
    }
  };

  const handleAnalyze = async () => {
    const userMessages = messagesRef.current.filter((m) => m.role === 'user');
    if (userMessages.length === 0 || isAnalyzing || isConnecting) {
      return;
    }
    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: messagesRef.current })
      });

      if (!response.ok) {
        throw new Error('Unable to analyze the conversation right now.');
      }

      const data = await response.json();
      setAnalysis(data.analysis ?? null);
    } catch (err) {
      console.error(err);
      setError(err.message ?? 'Something went wrong.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const statusMessage = error
    ? error
    : !micPermissionGranted
      ? 'Microphone access is blocked. Update permissions and reconnect.'
      : isConnecting
        ? 'Negotiating a secure audio link…'
        : isConnected
          ? 'Live conversation active. Share what stood out to you.'
          : 'Tap to connect your microphone and begin chatting with BakerMatcher.';

  return (
    <div className="page">
      <audio ref={remoteAudioRef} className="sr-only" autoPlay playsInline />
      <header className="header">
        <div className="logo-stack">
          <div className="logo-ring" aria-hidden="true">
            <img src="/BakerMatcher_logo.png" alt="BakerMatcher" />
          </div>
          <div>
            <h1>BakerMatcher Personality Analysis</h1>
            <p>
              Watch the featured video, share your thoughts, and let {ASSISTANT_NAME} surface how your
              perspective shapes your financial mindset.
            </p>
          </div>
        </div>
      </header>

      <main className="content">
        <section className="video-panel">
          <div className="video-wrapper">
            <iframe
              src={`${embedUrl}?rel=0`}
              title="Finance conversation prompt video"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
          {!hasStarted ? (
            <button className="primary" onClick={handleStartConversation} disabled={isConnecting}>
              {isConnecting ? 'Connecting…' : 'Give your thoughts'}
            </button>
          ) : (
            <p className="helper-text">
              BakerMatcher listens in real time—share anything the video sparked for you.
            </p>
          )}
        </section>

        {hasStarted && (
          <section className="chat-panel">
            <div className="chat-log" role="log" aria-live="polite">
              {messages.map((msg, idx) => (
                <Message key={`msg-${idx}`} role={msg.role} content={msg.content} />
              ))}
              {liveAssistantText && (
                <Message role="assistant" content={liveAssistantText} />
              )}
            </div>

            <div className="mic-control">
              <button
                className={`mic-button ${isConnected ? 'recording' : ''}`}
                onClick={handleToggleRealtime}
                disabled={isConnecting}
              >
                <span className="mic-icon" aria-hidden="true" />
                {isConnected ? 'Stop conversation' : isConnecting ? 'Connecting…' : 'Tap to speak'}
              </button>
              <div className="mic-status">{statusMessage}</div>
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || isConnecting || !hasUserMessages}
              >
                {isAnalyzing ? 'Analyzing…' : 'Analyze my thoughts'}
              </button>
            </div>

            {error && <div className="error-banner">{error}</div>}
            <AnalysisCard analysis={analysis} />
          </section>
        )}
      </main>

      <footer className="footer">
        {ASSISTANT_NAME} · Finance reflections powered by OpenAI
      </footer>
    </div>
  );
}

export default App;

