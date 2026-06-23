import numpy as np
import logging
import time
from typing import AsyncGenerator
from kokoro import KPipeline

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000
TWILIO_SAMPLE_RATE = 8000

FLUSH_CHARS = {'.', '?', '!', ',', ';', ':'}

class TTSClient:

    """
    From HuggingFace Kokoro repo:
        Voice options:
         - af_heart -- default voice
         - af_sarah -- warm, professional
         - af_nova -- confident
         - am_adam -- male, professional
    """

    def __init__(self, voice: str = "am_adam", lang_code: str = "a"):
        logger.info(f"TTSClient - Initializing Kokoro TTS with voice: {voice}")
        start = time.perf_counter()
        
        self.voice = voice
        self.pipeline = KPipeline(lang_code=lang_code)
        et = time.perf_counter() - start
        logger.info(f"TTSClient - Kokoro TTS initialized in {et:.2f} seconds")

    
    def _synthesize_chunk(self, text: str) -> list[np.ndarray]:

        if not text.strip():
            return []

        start = time.perf_counter()
        chunks = []

        generator = self.pipeline(text, voice=self.voice)
        for i, (gs, ps, audio) in enumerate(generator):
            logger.debug(f"TTSClient - Segment {i} | gs='{gs}' | ps='{ps}'")
            chunks.append(audio)
        
        et = time.perf_counter() - start
        total_duration = sum(len(c) for c in chunks) / KOKORO_SAMPLE_RATE
        logger.debug(
            f"TTSClient - Synthesized {total_duration:.2f}s of audio from"
            f" {len(text)} chars in {et:.2f}ms"
        )
        
        return chunks

    async def stream(
            self, token_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[np.ndarray, None]:

        buff = ""

        async for token in token_stream:
            buff += token
            if any(buffer.rstrip().endswith(char) for char in FLUSH_CHARS):
                for audio in self._synthesize_chunk(buff):
                    yield audio
                buff = ""
        
        # Flush any remaining text in the buffer after the stream ends
        if buff.strip():
            for audio in self._synthesize_chunk(buff):
                yield audio

    def to_mulaw(self, audio: np.ndarray) -> bytes:
        from scipy.signal import resample_poly

        downsampled = resample_poly(audio, up=1, down=3)
        cliped = np.clip(downsampled, -1.0, 1.0)
        pcm16 = (cliped * 32767).astype(np.int16)

        return self._linear_to_mulaw(pcm16).tobytes()
    
    @staticmethod
    def _linear_to_mulaw(pcm: np.ndarray) -> np.ndarray:
    
        BIAS = 33
        pcm = pcm.astype(np.int32)
        sign = np.where(pcm < 0, 0x80, 0x00)
        pcm = np.abs(pcm)
        pcm = np.clip(pcm, 0, 32635) + BIAS
        exp = np.clip(np.floor(np.log2(pcm)).astype(np.int32) - 5, 0, 7)
        mantissa = (pcm >> (exp + 1)) & 0x0F

        return (~(sign | (exp << 4) | mantissa)).astype(np.uint8)
    

