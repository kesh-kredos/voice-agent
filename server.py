"""
FastAPI server built to mimic 11labs architecture

Features:
    - Twilio message handling for future cases
    - Website UI demo similar to 11labs


Built using FastAPI and localhost hosting
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import date
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from models.voice_activity import VADClient
from models.speech_to_text import STTClient
from models.llm import LLMClient
from models.text_to_speech import TTSClient
from agent import VoiceAgent
from utils.audio import twilio_media_message

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

logger = logging.getLogger("Server")

_vad: VADClient = None
_stt: STTClient = None
_llm: LLMClient = None
_tts: TTSClient = None

_sessions: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vad, _stt, _llm, _tts
    start = time.perf_counter()

    logger.info("Loading models...")
    _vad = VADClient(threshold=float(os.getenv("VAD_THRESHOLD", "0.5")))
    _stt = STTClient(device=os.getenv("WHISPER_DVICE", "cuda:0"))
    _llm = LLMClient(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.getenv("VLLM_MODEL", "meta-llama/Meta-Llama-3.2-1B-Instruct")
    )
    _tts = TTSClient(
        voice=os.getenv("KOKORO_VOICE", "am_adam"),
        lang_code=os.getenv("KOKORO_LANG", 'a')
    )

    et = time.perf_counter() - start
    logger.info(f"Model initialization complete in {et:.0f}ms")
    yield
    logger.info("Shutting down server")

app = FastAPI(lifespan=lifespan)


def get_today_verbal() -> str:
    today = date.today()
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(today.day % 10, "th")
    if today.day in (11, 12, 13):
        suffix = "th"
    
    return f"{today.strftime('%B')} {today.day}{suffix}, {today.year}"

def build_customer_ctx(override: dict | None = None) -> dict:
    ctx = {
        "customer_name": "Sarah Johnson",
        "company": "Acme Telecom",
        "today_date": get_today_verbal(),
        "account_id": "ACM-88421",
        "balance": "820.57",
        "due_date": "July 1st, 2026",
        "last_payment": "June 1st, 2026"
    }
    if override:
        ctx.update(override)
    
    return ctx

@app.post('/browser-session')
async def create_browser_session(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    session_id = str(uuid.uuid4())
    _sessions[session_id] = build_customer_ctx(body)
    logger.info(f"New browser session created with ID: {session_id}")
    return JSONResponse({"session_id": session_id})
    
@app.websocket("/browser-stream/{session_id}")
async def browser_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"Browser websocket connected for session: {session_id}")

    customer_ctx = _sessions.pop(session_id, build_customer_ctx())

    agent = VoiceAgent(
        vad=_vad,
        stt=_stt,
        llm=_llm,
        tts=_tts,
        customer_ctx=customer_ctx,
        mode="browser"
    )

    async def send_audio(wav_bytes: bytes):
        await websocket.send_bytes(wav_bytes)

    async def send_event(event_type: str, **kwargs):
        await websocket.send_text(json.dumps({'type': event_type, **kwargs}))
    
    _original_handle_turn = agent._handle_turn

    async def _handle_turn_with_events(utterance, send_audio_cb):
        import asyncio
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(None, agent.stt.transcribe, utterance)
        if transcript.strip():
            await send_event("transcript", text=transcript)

        await _original_handle_turn(utterance, send_audio_cb)

    agent._handle_turn = _handle_turn_with_events
    agent_task = asyncio.create_task(agent.run(send_audio))

    try:
        async for raw_msg in websocket.iter.bytes():
            agent.enqueue_audio(raw_msg)
    
    except WebSocketDisconnect:
        logger.warning(f'Browser websocket disconnected for session {session_id}')

    finally:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            logger.warning(f"Asyncio cancel error for session: {session_id}")
        
        if agent.call_status:
            try:
                await send_event("status", value=agent.call_status)
            except Exception as e:
                logger.warning(f"Exception when trying to send event details in session {session_id}"
                               f"Error: {str(e)}")
        
        logger.info(f"Browser session ended for session {session_id}")

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Return TwiML to open a bidirectional media stream for this call."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    from_number = form.get("From", "unknown")
    logger.info(f"Incoming call: {call_sid} from {from_number}")

    host = request.headers.get("host", "localhost:8765")
    ws_url = f"wss://{host}/media-stream/{call_sid}?from={from_number}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="{ws_url}" />
        </Connect>
    </Response>"""

    return PlainTextResponse(content=twiml, media_type="text/xml")


@app.websocket("/media-stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str, from_number: str = "unknown"):
    """Bidirectional μ-law audio handler for one active Twilio call."""
    await websocket.accept()
    logger.info(f"Twilio WebSocket connected: {call_sid}")

    customer_ctx = build_customer_ctx() 

    agent = VoiceAgent(
        vad=_vad,
        stt=_stt,
        llm=_llm,
        tts=_tts,
        customer_ctx=customer_ctx,
        mode="twilio",      
    )

    stream_sid = None

    async def send_audio(mulaw_bytes: bytes):
        if stream_sid:
            await websocket.send_text(twilio_media_message(mulaw_bytes, stream_sid))

    agent_task = asyncio.create_task(agent.run(send_audio))

    try:
        async for raw_message in websocket.iter_text():
            data = json.loads(raw_message)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                logger.info(f"Twilio stream started: {stream_sid}")

            elif event == "media":
                mulaw = base64.b64decode(data["media"]["payload"])
                agent.enqueue_audio(mulaw)

            elif event == "stop":
                logger.info(f"Twilio stream stopped: {call_sid}")
                break

    except WebSocketDisconnect:
        logger.info(f"Twilio disconnected: {call_sid}")

    finally:
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        logger.info(f"Twilio call ended: {call_sid}")


@app.get("/health")
async def health():
    return {
        "status": "OK",
        "models": {
            "vad": _vad is not None,
            "stt": _stt is not None,
            "llm": _llm is not None,
            "tts": _tts is not None
        }
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")