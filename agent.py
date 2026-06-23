import asyncio 
import logging
import time
import numpy as np
import torch 
from typing import AsyncGenerator

from models.voice_activity import VADClient
from models.text_to_speech import TTSClient
from models.llm import LLMClient, EOC_SIGNALS
from models.speech_to_text import STTClient
from utils.audio import mulaw_to_float32, float32_to_mulaw

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_LIMIT = 22

class VoiceAgent:
    def __init__(
            self,
            vad: VADClient,
            stt: STTClient,
            llm: LLMClient,
            tts: TTSClient,
            customer_ctx: dict
        ):
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.customer_ctx = customer_ctx
        self.history: list[dict] = []
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._call_ended: bool = False
        self.call_status: str | None = None

        logger.info(f"Agent - New agent initialized for {customer_ctx.get("name", "TEST CUSTOMER")}")

    async def _utterance_stream(self) -> AsyncGenerator[np.ndarray, None]:
        vad_buff = []
        silence_count = 0
        speaking = False

        while True:
            if self._call_ended:
                return

            mulaw_chunk = await self._audio_queue.get()
            audio_chunk = mulaw_to_float32(mulaw_chunk)

            for i in range(0, len(audio_chunk) - 511, CHUNK_SIZE):
                frame = audio_chunk[i:i + CHUNK_SIZE]
                tensor = torch.from_numpy(frame)
                speech_prob = self.vad.model(tensor, SAMPLE_RATE).item()

                if speech_prob >= self.vad.threshold:
                    speaking = True
                    silence_count = 0
                    vad_buff.append(frame)
                
                elif speaking: 
                    vad_buff.append(frame)
                    silence_count += 1

                    if silence_count >= SILENCE_LIMIT:
                        yield np.concatenate(vad_buff)
                        vad_buff = []
                        silence_count = 0
                        speaking = False
    
    async def _handle_turn(self, utterance: np.ndarray, send_audio):
        start = time.perf_counter()

        transcript = self.stt.transcribe(utterance)
        if not transcript.strip():
            return
        
        logger.info(f"Customer {self.customer_ctx.get("name", "TEST CUSTOMER")} said: '{transcript}'")

        tokens = []

        async def token_stream_with_collection():
            async for tok in self.llm.stream(transcript, self.history, self.customer_ctx):
                tokens.append(tok)
                yield tok
        
        async def clean_token_stream():
            buff = ""
            async for tok in token_stream_with_collection():
                buff += tok

                if "SIGNAL" in buff:
                    if "\n" in buff.split("SIGNAL")[-1] or buff.endswith(tuple(EOC_SIGNALS)):
                        buff = buff[:buff.index("SIGNAL")].rstrip()
                        yield buff
                        return
                    continue
                yield tok
                buff = ""
        
        audio_stream = self.tts.stream(clean_token_stream())

        first_audio = False
        async for audio_chunk in audio_stream:
            mulaw = float32_to_mulaw(audio_chunk)
            await send_audio(mulaw)

            if not first_audio:
                et = (time.perf_counter() - start) * 1000
                logger.info("")
