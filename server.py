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
    
