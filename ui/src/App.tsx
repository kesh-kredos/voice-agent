import { useState, useEffect, useRef, useCallback } from 'react'

const BASE_URL = window.location.origin

type AppState = 'idle' | 'connected' | 'listening' | 'speaking'

interface Message {
  id: number
  role: 'user' | 'agent'
  text: string
}

const NUM_BARS = 12

// Energy gate for the mic feed. While push-to-talk is held we otherwise
// forward every ~2.6ms chunk to the server; near-silence then accumulates
// into VAD-flagged "utterances" that Whisper hallucinates (e.g. "Thank
// you."). Values are in PCM16 units where full-scale = 32767.
const GATE_RMS_THRESHOLD = 200   // ~-44 dBFS; ~2x the observed noise floor
const GATE_HANGOVER_MS = 300     // bridge inter-word pauses / soft onsets

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
  const playbackNodeRef = useRef<AudioWorkletNode | null>(null)
  const playInitRef = useRef<Promise<void> | null>(null)

  // Monotonic playback epoch. Stamped onto every 'samples' message and echoed
  // back by the worklet in 'ended', so a drain notification queued from a
  // previous utterance can't flip the UI to 'connected' after newer samples
  // have already been posted. Bumped in resetPlayback to invalidate it.
  const playbackEpochRef = useRef(0)
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const micPressedRef = useRef(false)
  const connectingRef = useRef(false)
  const listeningStartRef = useRef(false)
  const appStateRef = useRef<AppState>('idle')
  const msgIdRef = useRef(0)
  const transcriptRef = useRef<HTMLDivElement>(null)

  // ── State sync ────────────────────────────────────────────────────────────

  // Keep the ref in lock-step with state so event handlers that fire between
  // the setState call and the next render (e.g. a quick tap/release on the mic
  // button) see the latest value immediately.
  const setAppStateAndRef = useCallback((s: AppState) => {
    appStateRef.current = s
    setAppState(s)
  }, [])

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

  // Create + unlock the playback AudioContext. MUST be called from a user
  // gesture (the Connect click). Chrome's autoplay policy otherwise leaves the
  // context 'suspended', which freezes currentTime at 0 and the worklet never
  // renders. We also load the playback AudioWorklet module here so the node
  // is ready the moment the first binary frame arrives.
  const ensurePlayCtx = useCallback(async () => {
    if (!playCtxRef.current) {
      playCtxRef.current = new AudioContext({ latencyHint: 'interactive' })
    }
    const ctx = playCtxRef.current
    if (ctx.state === 'suspended') await ctx.resume()

    if (!playbackNodeRef.current) {
      // Serialize worklet creation: concurrent callers await the same promise
      // so rapid clicks can't spawn orphaned AudioWorkletNodes.
      if (!playInitRef.current) {
        playInitRef.current = (async () => {
          await ctx.audioWorklet.addModule('/playback-processor.js')
          const node = new AudioWorkletNode(ctx, 'playback-processor', {
            processorOptions: { prebufferSec: 0.15 },
            outputChannelCount: [1]
          })
          node.connect(ctx.destination)
          // The worklet posts 'ended' when its ring buffer drains to empty
          // after playback — flip back to 'connected' then, instead of on a
          // fixed timer that fires before the tail audio finishes.
          node.port.onmessage = (e) => {
            const msg = e.data
            // Reject stale 'ended' from a prior drain: the worklet stamps it
            // with the epoch of the samples that were playing. If newer samples
            // have since been posted the epoch won't match, so we don't flip
            // to 'connected' while the next utterance is starting.
            if (msg && msg.type === 'ended' && appStateRef.current === 'speaking' && msg.epoch === playbackEpochRef.current) {
              setAppStateAndRef('connected')
              setStatusText('Connected — hold mic to speak')
            }
          }
          playbackNodeRef.current = node
        })().catch((err) => {
          playInitRef.current = null
          throw err
        })
      }
      await playInitRef.current
    }

    // Play a silent buffer to fully unlock output on the gesture.
    const silent = ctx.createBuffer(1, 1, ctx.sampleRate)
    const s = ctx.createBufferSource()
    s.buffer = silent
    s.connect(ctx.destination)
    s.start(0)
    return ctx
  }, [setAppStateAndRef])

  // Clear the worklet's ring buffer. Called on disconnect / error so stale
  // audio from a previous session never leaks into the next one.
  const resetPlayback = useCallback(() => {
    // Bump the epoch so any 'ended' still queued from before the reset can't
    // match the current epoch and flip the UI after a disconnect/reconnect.
    playbackEpochRef.current++
    playbackNodeRef.current?.port.postMessage({ type: 'clear' })
  }, [])

  // Release every mic-capture resource (worklet, MediaStream tracks,
  // analyser) and restart the idle waveform. Shared by stopListening and
  // every session-teardown path so a mid-listening disconnect can't leak
  // the microphone or its animation frame.
  const teardownMic = useCallback(() => {
    workletNodeRef.current?.disconnect()
    workletNodeRef.current = null
    micSourceRef.current?.disconnect()
    micSourceRef.current = null
    micStreamRef.current?.getTracks().forEach(t => t.stop())
    micStreamRef.current = null
    analyserRef.current?.disconnect()
    analyserRef.current = null
    cancelAnimationFrame(animFrameRef.current)
    idlePulse()
  }, [idlePulse])

  // Parse a raw PCM16LE 24 kHz binary frame, convert to float32, resample to
  // the playback context's sample rate if needed, and push into the worklet's
  // ring buffer.
  const handleBinaryFrame = useCallback((data: ArrayBuffer) => {
    if (!wsRef.current) return
    const node = playbackNodeRef.current
    const ctx = playCtxRef.current
    if (!node || !ctx) return

    // Decode strictly as little-endian PCM16 — Int16Array would silently
    // mis-decode on big-endian hosts.
    const view = new DataView(data)
    const numSamples = data.byteLength >> 1
    if (numSamples === 0) return

    const float32 = new Float32Array(numSamples)
    for (let i = 0; i < numSamples; i++) {
      float32[i] = view.getInt16(i * 2, true) / 32768
    }

    // Linear-interpolation resampler from 24 kHz → ctx.sampleRate.
    const srcRate = 24000
    const dstRate = ctx.sampleRate
    let samples: Float32Array
    if (srcRate === dstRate) {
      samples = float32
    } else {
      const ratio = dstRate / srcRate
      const outLen = Math.floor(float32.length * ratio)
      samples = new Float32Array(outLen)
      for (let i = 0; i < outLen; i++) {
        const srcIdx = i / ratio
        const idx0 = Math.floor(srcIdx)
        const idx1 = Math.min(idx0 + 1, float32.length - 1)
        const frac = srcIdx - idx0
        samples[i] = float32[idx0] * (1 - frac) + float32[idx1] * frac
      }
    }

    // Stamp this chunk with a fresh playback epoch; the worklet echoes it back
    // on 'ended' so we can tell a stale drain from the current one.
    const epoch = ++playbackEpochRef.current
    node.port.postMessage({ type: 'samples', data: samples, epoch }, [samples.buffer])

    // Mark agent as speaking; the worklet posts 'ended' when its buffer
    // drains, which flips us back to 'connected'.
    if (appStateRef.current !== 'speaking') {
      setAppStateAndRef('speaking')
      setStatusText('Agent speaking…')
    }
  }, [setAppStateAndRef])

  // ── WebSocket connection ──────────────────────────────────────────────────

  const connectSession = useCallback(async () => {
    // Guard against re-entrant clicks while the session fetch / socket
    // handshake is in flight — otherwise each click leaks a stale audio
    // context and an orphan WebSocket.
    if (connectingRef.current) return
    if (wsRef.current) return
    connectingRef.current = true
    try {
      // Unlock audio output now, while we're still inside the click gesture,
      // so the agent's opening line plays as soon as it arrives.
      await ensurePlayCtx()

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
        setAppStateAndRef('connected')
        setStatusText('Connected — hold mic to speak')
        cancelAnimationFrame(animFrameRef.current)
        idlePulse()
      }

      ws.onmessage = (e) => {
        if (e.data instanceof ArrayBuffer) {
          handleBinaryFrame(e.data)
        } else {
          const evt = JSON.parse(e.data as string)
          if (evt.type === 'transcript') addMessage('user', evt.text)
          if (evt.type === 'agent_text') addMessage('agent', evt.text)
          if (evt.type === 'status') {
            if (wsRef.current === ws) {
              wsRef.current = null
              ws.onopen = null
              ws.onmessage = null
              ws.onclose = null
              ws.onerror = null
              resetPlayback()
              teardownMic()
              ws.close()
            }
            setAppStateAndRef('idle')
            setStatusText(`Call ended: ${evt.value}`)
          }
        }
      }

      ws.onclose = () => {
        if (wsRef.current !== ws) return
        wsRef.current = null
        resetPlayback()
        teardownMic()
        setAppStateAndRef('idle')
        setStatusText('Disconnected')
      }

      ws.onerror = () => {
        if (wsRef.current !== ws) return
        wsRef.current = null
        resetPlayback()
        teardownMic()
        setAppStateAndRef('idle')
        setStatusText('Connection error — is the server running?')
      }

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setAppStateAndRef('idle')
      setStatusText(`Error: ${msg}`)
    } finally {
      connectingRef.current = false
    }
  }, [idlePulse, handleBinaryFrame, addMessage, ensurePlayCtx, resetPlayback, teardownMic, setAppStateAndRef])

  const disconnect = useCallback(() => {
    const ws = wsRef.current
    if (!ws) return
    wsRef.current = null
    // Detach handlers before close so any frames the browser already queued
    // are discarded rather than processed into playback after teardown.
    ws.onopen = null
    ws.onmessage = null
    ws.onclose = null
    ws.onerror = null
    resetPlayback()
    teardownMic()
    setAppStateAndRef('idle')
    setStatusText('Disconnected')
    ws.close()
  }, [resetPlayback, teardownMic, setAppStateAndRef])

  // ── Mic capture ───────────────────────────────────────────────────────────

  const startListening = useCallback(async () => {
    if (listeningStartRef.current) return
    listeningStartRef.current = true
    try {

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
    micSourceRef.current = src

    analyserRef.current = ctx.createAnalyser()
    analyserRef.current.fftSize = 256
    src.connect(analyserRef.current)

    workletNodeRef.current = new AudioWorkletNode(ctx, 'pcm-processor')
    src.connect(workletNodeRef.current)

    const ratio = ctx.sampleRate / 16000

    // Per-listening hangover deadline for the energy gate. Held in this
    // closure so it resets each time the mic graph is rebuilt.
    let gateHangoverUntil = 0

    workletNodeRef.current.port.onmessage = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
      const input: Float32Array = e.data
      if (!input || input.length === 0) return
      const outLen = Math.floor(input.length / ratio)
      const pcm16 = new Int16Array(outLen)
      let sumSq = 0
      for (let i = 0; i < outLen; i++) {
        // Box-average the source samples spanning this output sample. Cheap
        // anti-alias filter so 48k→16k decimation doesn't garble the mic feed
        // and STT can reliably pick up the user turn after turn.
        const start = Math.floor(i * ratio)
        const end = Math.min(input.length, Math.floor((i + 1) * ratio))
        let sum = 0
        let n = 0
        for (let j = start; j < end; j++) { sum += input[j]; n++ }
        const s = n > 0 ? sum / n : input[start]
        const v = Math.max(-32768, Math.min(32767, s * 32767))
        pcm16[i] = v
        sumSq += v * v
      }
      // Energy gate: drop near-silence/noise-floor chunks so they can't
      // accumulate into hallucinated utterances server-side. A short hangover
      // keeps sending after speech ends so inter-word pauses and soft onsets
      // aren't chopped.
      const rms = outLen > 0 ? Math.sqrt(sumSq / outLen) : 0
      const now = performance.now()
      if (rms >= GATE_RMS_THRESHOLD) {
        gateHangoverUntil = now + GATE_HANGOVER_MS
      } else if (now >= gateHangoverUntil) {
        return
      }
      wsRef.current.send(pcm16.buffer)
    }

    // If the user released the mic button while async setup was in flight,
    // tear the graph back down immediately so the mic never becomes active
    // on a button that's no longer held.
    if (!micPressedRef.current) {
      teardownMic()
      return
    }

    // If the websocket died while async mic setup was in flight, don't
    // transition to 'listening' — fall back to idle/disconnected cleanup.
    if (!wsRef.current) {
      teardownMic()
      setAppStateAndRef('idle')
      setStatusText('Disconnected')
      return
    }

    cancelAnimationFrame(animFrameRef.current)
    micAnalyse()
    setAppStateAndRef('listening')
    setStatusText('Listening…')

    } catch (err: unknown) {
      teardownMic()
      const msg = err instanceof Error ? err.message : 'Unknown error'
      if (!wsRef.current) {
        setAppStateAndRef('idle')
        setStatusText('Disconnected')
      } else {
        setAppStateAndRef('connected')
        setStatusText(`Mic error: ${msg}`)
      }
    } finally {
      listeningStartRef.current = false
    }
  }, [micAnalyse, teardownMic, setAppStateAndRef])

  const stopListening = useCallback(() => {
    // Send an explicit end-of-turn signal before tearing down the mic so
    // the server can finalize the utterance immediately — the RMS gate
    // suppresses the trailing silence VAD needs to do this on its own.
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_of_turn' }))
    }
    teardownMic()
    setAppStateAndRef('connected')
    setStatusText('Connected — hold mic to speak')
  }, [teardownMic, setAppStateAndRef])

  // ── Mic button handlers ───────────────────────────────────────────────────

  const handleMicDown = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    micPressedRef.current = true
    if (appStateRef.current !== 'listening') startListening()
  }, [startListening])

  const handleMicUp = useCallback(() => {
    micPressedRef.current = false
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
            onMouseLeave={handleMicUp}
            onTouchStart={handleMicDown}
            onTouchEnd={handleMicUp}
            onTouchCancel={handleMicUp}
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