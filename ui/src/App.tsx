import { useState, useEffect, useRef, useCallback } from 'react'

const BASE_URL = window.location.origin

type AppState = 'idle' | 'connected' | 'listening' | 'speaking'

interface Message {
  id: number
  role: 'user' | 'agent'
  text: string
}

const NUM_BARS = 12

export default function App() {
  const [appState, setAppState] = useState<AppState>('idle')
  const [messages, setMessages] = useState<Message[]>([])
  const [sessionInfo, setSessionInfo] = useState('')
  const [statusText, setStatusText] = useState('Not connected')
  const [barHeights, setBarHeights] = useState<number[]>(Array(NUM_BARS).fill(6))

  // ── Refs ──────────────────────────────────────────────────────────────────
  const wsRef = useRef<WebSocket | null>(null)

  const micCtxRef = useRef<AudioContext | null>(null)

  const playCtxRef = useRef<AudioContext | null>(null)

  const analyserRef = useRef<AnalyserNode | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const animFrameRef = useRef<number>(0)
  const agentQueueRef = useRef<ArrayBuffer[]>([])
  const isPlayingRef = useRef(false)
  // Tracks scheduled end time for gapless chunk-to-chunk playback
  const nextStartRef = useRef(0)
  const appStateRef = useRef<AppState>('idle')
  const msgIdRef = useRef(0)
  const transcriptRef = useRef<HTMLDivElement>(null)

  // ── State sync ────────────────────────────────────────────────────────────

  useEffect(() => { appStateRef.current = appState }, [appState])

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight
    }
  }, [messages])

  const addMessage = useCallback((role: 'user' | 'agent', text: string) => {
    setMessages(prev => [...prev, { id: msgIdRef.current++, role, text }])
  }, [])

  // ── Waveform animation ────────────────────────────────────────────────────

  const idlePulse = useCallback(() => {
    const animate = () => {
      const t = Date.now() / 1000
      setBarHeights(Array.from({ length: NUM_BARS }, (_, i) =>
        6 + Math.sin(t * 1.2 + i * 0.6) * 3
      ))
      animFrameRef.current = requestAnimationFrame(animate)
    }
    animFrameRef.current = requestAnimationFrame(animate)
  }, [])

  const micAnalyse = useCallback(() => {
    const animate = () => {
      if (!analyserRef.current) return
      const data = new Uint8Array(analyserRef.current.frequencyBinCount)
      analyserRef.current.getByteFrequencyData(data)
      const step = Math.floor(data.length / NUM_BARS)
      setBarHeights(Array.from({ length: NUM_BARS }, (_, i) =>
        Math.max(6, (data[i * step] / 255) * 56)
      ))
      animFrameRef.current = requestAnimationFrame(animate)
    }
    animFrameRef.current = requestAnimationFrame(animate)
  }, [])

  useEffect(() => {
    idlePulse()
    return () => cancelAnimationFrame(animFrameRef.current)
  }, [idlePulse])

  // ── Agent audio playback ──────────────────────────────────────────────────

  const drainQueue = useCallback(async () => {
    if (isPlayingRef.current) return
    isPlayingRef.current = true

    if (!playCtxRef.current) {
      playCtxRef.current = new AudioContext()
    }
    const ctx = playCtxRef.current
    if (ctx.state === 'suspended') await ctx.resume()

    setAppState('speaking')
    setStatusText('Agent speaking…')

    while (agentQueueRef.current.length > 0) {
      const wav = agentQueueRef.current.shift()!
      let buf: AudioBuffer
      try {
        buf = await ctx.decodeAudioData(wav.slice(0))
      } catch (e) {
        console.error('decodeAudioData failed:', e)
        continue
      }
      const src = ctx.createBufferSource()
      src.buffer = buf
      src.connect(ctx.destination)

      const now = ctx.currentTime
      const start = Math.max(now, nextStartRef.current)
      src.start(start)
      nextStartRef.current = start + buf.duration
    }

    isPlayingRef.current = false

    if (appStateRef.current === 'speaking') {
      setAppState('connected')
      setStatusText('Connected — hold mic to speak')
    }
  }, [])

  // ── WebSocket connection ──────────────────────────────────────────────────

  const connectSession = useCallback(async () => {
    try {
      setStatusText('Creating session…')

      const res = await fetch(`${BASE_URL}/browser-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}'
      })
      if (!res.ok) throw new Error(`Session failed: ${res.status}`)
      const { session_id } = await res.json()
      setSessionInfo(`Session ${session_id.slice(0, 8)}…`)

      const ws = new WebSocket(
        `${BASE_URL.replace('http', 'ws')}/browser-stream/${session_id}`
      )
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => {
        setAppState('connected')
        setStatusText('Connected — hold mic to speak')
        cancelAnimationFrame(animFrameRef.current)
        idlePulse()
      }

      ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) {
          agentQueueRef.current.push(e.data)
          void drainQueue()
        } else {
          const evt = JSON.parse(e.data as string)
          if (evt.type === 'transcript') addMessage('user', evt.text)
          if (evt.type === 'agent_text') addMessage('agent', evt.text)
          if (evt.type === 'status') {
            setAppState('idle')
            setStatusText(`Call ended: ${evt.value}`)
          }
        }
      }

      ws.onclose = () => {
        setAppState('idle')
        setStatusText('Disconnected')
        wsRef.current = null
        cancelAnimationFrame(animFrameRef.current)
        idlePulse()
      }

      ws.onerror = () => {
        setAppState('idle')
        setStatusText('Connection error — is the server running?')
        wsRef.current = null
      }

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setAppState('idle')
      setStatusText(`Error: ${msg}`)
    }
  }, [idlePulse, drainQueue, addMessage])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
  }, [])

  // ── Mic capture ───────────────────────────────────────────────────────────

  const startListening = useCallback(async () => {

    if (!micCtxRef.current) {
      micCtxRef.current = new AudioContext()
    }
    const ctx = micCtxRef.current
    if (ctx.state === 'suspended') await ctx.resume()

    micStreamRef.current = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    })

    await ctx.audioWorklet.addModule('/pcm-processor.js')

    const src = ctx.createMediaStreamSource(micStreamRef.current)

    analyserRef.current = ctx.createAnalyser()
    analyserRef.current.fftSize = 256
    src.connect(analyserRef.current)

    workletNodeRef.current = new AudioWorkletNode(ctx, 'pcm-processor')
    src.connect(workletNodeRef.current)

    const ratio = ctx.sampleRate / 16000

    workletNodeRef.current.port.onmessage = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      const input: Float32Array = e.data
      const outLen = Math.floor(input.length / ratio)
      const pcm16 = new Int16Array(outLen)
      for (let i = 0; i < outLen; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, input[Math.floor(i * ratio)] * 32767))
      }
      wsRef.current.send(pcm16.buffer)
    }

    cancelAnimationFrame(animFrameRef.current)
    micAnalyse()
    setAppState('listening')
    setStatusText('Listening…')
  }, [micAnalyse])

  const stopListening = useCallback(() => {
    workletNodeRef.current?.disconnect()
    workletNodeRef.current = null
    micStreamRef.current?.getTracks().forEach(t => t.stop())
    micStreamRef.current = null
    analyserRef.current?.disconnect()
    analyserRef.current = null
    cancelAnimationFrame(animFrameRef.current)
    idlePulse()
    setAppState('connected')
    setStatusText('Connected — hold mic to speak')
  }, [idlePulse])

  // ── Mic button handlers ───────────────────────────────────────────────────

  const handleMicDown = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    if (appStateRef.current !== 'listening') startListening()
  }, [startListening])

  const handleMicUp = useCallback(() => {
    if (appStateRef.current === 'listening') stopListening()
  }, [stopListening])

  // ── Derived style helpers ─────────────────────────────────────────────────
  const isConnected = appState === 'connected' || appState === 'listening' || appState === 'speaking'
  const micDisabled = !isConnected

  const barColor = appState === 'listening'
    ? '#C8A96E'
    : appState === 'speaking'
    ? '#4F8EF7'
    : 'rgba(255,255,255,0.15)'

  const dotColor = appState === 'listening'
    ? '#C8A96E'
    : appState === 'speaking'
    ? '#4F8EF7'
    : appState === 'connected'
    ? '#4CAF7D'
    : 'rgba(255,255,255,0.2)'

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={styles.shell}>
      <div style={styles.card}>

        {/* Header */}
        <div style={styles.companyLabel}>Billing Agent Demo</div>

        {/* Waveform — animates gold when listening, blue when agent speaking */}
        <div style={styles.waveformArea}>
          <div style={styles.waveform}>
            {barHeights.map((h, i) => (
              <div
                key={i}
                style={{
                  ...styles.bar,
                  height: h,
                  background: barColor,
                  transition: 'height 0.08s ease, background 0.3s ease'
                }}
              />
            ))}
          </div>
          <div style={styles.statusRow}>
            <div style={{ ...styles.statusDot, background: dotColor }} />
            <span style={{
              ...styles.statusText,
              color: appState !== 'idle' ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.45)'
            }}>
              {statusText}
            </span>
          </div>
        </div>

        {/* Transcript — scrollable chat bubbles, user left / agent right */}
        <div style={styles.transcript} ref={transcriptRef}>
          {messages.length === 0 ? (
            <div style={styles.emptyMsg}>Connect to start a session</div>
          ) : (
            messages.map(m => (
              <div key={m.id} style={{
                ...styles.bubble,
                ...(m.role === 'user' ? styles.bubbleUser : styles.bubbleAgent)
              }}>
                <div style={{
                  ...styles.bubbleLabel,
                  color: m.role === 'user' ? 'rgba(255,255,255,0.3)' : 'rgba(79,142,247,0.5)',
                  textAlign: m.role === 'agent' ? 'right' : 'left'
                }}>
                  {m.role === 'user' ? 'You' : 'Agent'}
                </div>
                <div>{m.text}</div>
              </div>
            ))
          )}
        </div>

        {/* Controls — mic hold-to-speak button + connect/disconnect button */}
        <div style={styles.controls}>
          <button
            style={{
              ...styles.micBtn,
              ...(appState === 'listening' ? styles.micBtnListening
                : appState === 'speaking' ? styles.micBtnSpeaking
                : {}),
              opacity: micDisabled ? 0.35 : 1,
              cursor: micDisabled ? 'not-allowed' : 'pointer'
            }}
            disabled={micDisabled}
            onMouseDown={handleMicDown}
            onMouseUp={handleMicUp}
            onTouchStart={handleMicDown}
            onTouchEnd={handleMicUp}
            title="Hold to speak"
          >
            <i className="ti ti-microphone" aria-hidden="true" />
          </button>

          <button
            style={{
              ...styles.connectBtn,
              ...(isConnected ? styles.connectBtnConnected : {})
            }}
            onClick={isConnected ? disconnect : connectSession}
          >
            {isConnected ? 'Disconnect' : 'Connect to Agent'}
          </button>

          {sessionInfo && (
            <div style={styles.sessionInfo}>{sessionInfo}</div>
          )}
        </div>

      </div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  shell: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem 1rem',
    background: '#050A18',
  },
  card: {
    width: '100%',
    maxWidth: 480,
    background: 'rgba(13, 21, 38, 0.9)',
    border: '0.5px solid rgba(255,255,255,0.08)',
    borderRadius: 24,
    padding: '2.5rem 2rem 2rem',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    animation: 'fadeIn 0.4s ease forwards',
  },
  companyLabel: {
    fontSize: 11,
    fontWeight: 500,
    letterSpacing: '0.12em',
    color: 'rgba(255,255,255,0.3)',
    textTransform: 'uppercase',
    marginBottom: '2rem',
  },
  waveformArea: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '1.5rem',
    marginBottom: '2rem',
    width: '100%',
  },
  waveform: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    height: 64,
  },
  bar: {
    width: 4,
    borderRadius: 2,
    background: 'rgba(255,255,255,0.15)',
  },
  statusRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: '50%',
    flexShrink: 0,
    transition: 'background 0.3s ease',
  },
  statusText: {
    fontSize: 13,
    letterSpacing: '0.02em',
    fontWeight: 400,
    transition: 'color 0.3s ease',
  },
  transcript: {
    width: '100%',
    maxHeight: 220,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
    marginBottom: '2rem',
    padding: '0 0.25rem',
    scrollbarWidth: 'none',
  },
  emptyMsg: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.2)',
    textAlign: 'center',
    padding: '1.5rem 0',
    letterSpacing: '0.01em',
  },
  bubble: {
    maxWidth: '85%',
    padding: '10px 14px',
    borderRadius: 14,
    fontSize: 13.5,
    lineHeight: 1.5,
    animation: 'fadeUp 0.3s ease forwards',
  },
  bubbleUser: {
    background: 'rgba(255,255,255,0.07)',
    color: 'rgba(255,255,255,0.75)',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 4,
  },
  bubbleAgent: {
    background: 'rgba(79,142,247,0.12)',
    color: 'rgba(200,220,255,0.85)',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 4,
  },
  bubbleLabel: {
    fontSize: 10,
    fontWeight: 500,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  controls: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '1rem',
    width: '100%',
  },
  micBtn: {
    width: 64,
    height: 64,
    borderRadius: '50%',
    border: '0.5px solid rgba(255,255,255,0.12)',
    background: 'rgba(255,255,255,0.05)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 22,
    color: 'rgba(255,255,255,0.6)',
    transition: 'background 0.2s ease, border-color 0.2s ease, color 0.2s ease',
    outline: 'none',
  },
  micBtnListening: {
    background: 'rgba(200,169,110,0.15)',
    borderColor: 'rgba(200,169,110,0.5)',
    color: '#C8A96E',
  },
  micBtnSpeaking: {
    background: 'rgba(79,142,247,0.15)',
    borderColor: 'rgba(79,142,247,0.4)',
    color: '#4F8EF7',
  },
  connectBtn: {
    width: '100%',
    padding: '13px 0',
    borderRadius: 12,
    border: '0.5px solid rgba(255,255,255,0.1)',
    background: 'rgba(255,255,255,0.05)',
    color: 'rgba(255,255,255,0.6)',
    fontSize: 14,
    fontWeight: 500,
    letterSpacing: '0.02em',
    cursor: 'pointer',
    transition: 'background 0.2s ease, color 0.2s ease',
    outline: 'none',
    fontFamily: 'inherit',
  },
  connectBtnConnected: {
    background: 'rgba(255,80,80,0.08)',
    borderColor: 'rgba(255,80,80,0.2)',
    color: 'rgba(255,120,120,0.8)',
  },
  sessionInfo: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.2)',
    letterSpacing: '0.04em',
  },
}