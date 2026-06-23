import asyncio 
import logging
import time
import numpy as np
import torch 
from typing import AsyncGenerator

from asyncio import Queue

from models.voice_activity import VADClient
from models.text_to_speech import TTSClient
from models.llm import LLMClient, EOC_SIGNALS
from models.speech_to_text import STTClient
from utils.audio import mulaw_to_float32, float32_to_mulaw, pcm16_to_float32, float32_to_wav

logger = logging.getLogger("Agent")

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_CHUNKS = 22

class VoiceAgent:
    def __init__(
            self,
            vad: VADClient,
            stt: STTClient,
            llm: LLMClient,
            tts: TTSClient,
            customer_ctx: dict,
            mode: str = "twilio"
        ):
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.customer_ctx = customer_ctx
        self.mode = mode

        self.history: list[dict] = []
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._call_ended: bool = False
        self.call_status: str | None = None

        logger.info(f"New agent initialized for {customer_ctx.get("name", "TEST CUSTOMER")}")


    def enqueue_audio(self, raw_bytes: bytes):
        self._audio_queue.put_nowait(raw_bytes)

    async def run(self, send_audio_callback):
        logger.info(
            f'New voice session started for {self.customer_ctx.get('customer_name', 'TEST CUSTOMER')}'
            f'[mode={self.mode}]'
        )

        async for utterance in self._utterance_stream():
            if self._call_ended:
                break
            await self._handle_turn(utterance, send_audio_callback)
            if self._call_ended:
                break

    def _decode_audio(self, raw_bytes: bytes) -> np.ndarray:
        if self.mode == 'twilio':
            return mulaw_to_float32(raw_bytes)
        else:
            return pcm16_to_float32(raw_bytes)

    async def _utterance_stream(self) -> AsyncGenerator[np.ndarray, None]:
        vad_buff: list[np.ndarray] = []
        silence_count = 0
        speaking = False

        while not self._call_ended:
            try:
                raw_audio = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            
            audio_chunk = self._decode_audio(raw_audio)
            frames = [audio_chunk[i:i + CHUNK_SIZE] for i in range(0, len(audio_chunk), CHUNK_SIZE)]

            for frame in frames:
                if len(frame) < CHUNK_SIZE:
                    continue

                is_speech = self.vad.is_speech()

                if is_speech:
                    in_speech = True
                    silence_count = 0
                    vad_buff.append(frame)

                elif in_speech:
                    silence_count += 1
                    vad_buff.append(frame)

                    if silence_count >= SILENCE_CHUNKS:
                        yield np.concatenate(vad_buff)
                        vad_buff = []
                        silence_count = 0
                        in_speech = False


            
    
    async def _handle_turn(self, utterance: np.ndarray, send_audio):
        start = time.perf_counter()

        transcript = await asyncio.get_event_loop().run_in_executor(
            None, self.stt.transcribe, utterance
        )
        if not transcript.strip():
            return
        
        logger.info(f"Agent - Customer {self.customer_ctx.get("name", "TEST CUSTOMER")} said: '{transcript}'")

        if self._should_escalate(transcript):
            await self._escalate(send_audio)
            self._call_ended = True
            self.call_status = "escalated"
            return
        
        tokens: list[str] = []

        async def token_stream_with_collection():
            async for tok in self.llm.stream(transcript, self.history, self.customer_ctx):
                tokens.append(tok)
                yield tok
        
        async for audio_chunk in self.tts.stream(token_stream_with_collection):
            encoded = self._audio_queue(audio_chunk)
            await send_audio(encoded)
        
        et = (time.perf_counter() - start) * 1000
        logger.info(f'Outgoing message completed in {et:.0f}ms')
        
        response = "".join(tokens)
        for sig in EOC_SIGNALS:
            if sig in response:
                status = sig.split(":")[-1]
                logger.info(f"End-of-call signal detected: {status}")
                self.call_status = status
                self._call_ended = True
                return
        
        clean_response = response
        for sig in EOC_SIGNALS:
            clean_response = clean_response.replace(sig, "").strip()
        
        self.history.append({"role": "user", "content": transcript})
        self.history.append({"role": "assistant", "content": clean_response})
        if len(self.history) > 20:
            self.history = self.history[-20:]

    
    def _audio_encode(self, audio: np.ndarray) -> bytes:
        if self.mode == 'twilio':
            return float32_to_mulaw(audio)
        else:
            return float32_to_wav(audio)
    
    def _should_escalate(self, transcript: str) -> bool:
        phrases = [
            "speak to a human", "speak to someone", "real person",
            "agent please", "transfer me", "supervisor"
        ]

        return any(p in transcript.lower() for p in phrases)
    
    async def _escalate(self, send_audio):
        message = (
            "Of course, let me transfer you to one of our agents right now",
            "Please hold for just a moment."
        )

        async for audio in self.tts.stream(self._as_token_stream(message)):
            await send_audio(self._audio_encode(audio))
        
        logger.info("Escalation was detected")
    
    @staticmethod
    async def _as_token_stream(text: str) -> AsyncGenerator[str, None]:
        yield text
