import numpy as np
import logging
import time
from typing import AsyncGenerator
from kokoro import KPipeline
import asyncio

logger = logging.getLogger("Kokoro Client")

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
        logger.info(f"Initializing Kokoro TTS with voice: {voice}")
        start = time.perf_counter()
        
        self.voice = voice
        self.pipeline = KPipeline(lang_code=lang_code)
        et = time.perf_counter() - start
        logger.info(f"Kokoro TTS initialized in {et:.2f} seconds")

    
    async def stream(
            self,
            token_stream: AsyncGenerator[str, None]
        ) -> AsyncGenerator[np.ndarray, None]:

        buff = ""
        async for token in token_stream:
            buff += token
            if any(ch in buff for ch in FLUSH_CHARS):
                sentences = self._split_on_flush(buff)
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        async for chunk in self._synthesize(sentence):
                            yield chunk
                buff = sentences[-1]
        
        if buff.strip():
            async for chunk in self._synthesize(buff.strip()):
                yield chunk

    
    async def _synthesize(self, text: str) -> AsyncGenerator[np.ndarray, None]:

        loop = asyncio.get_event_loop()

        def _run():
            generator = self.pipeline(text, voice=self.voice)
            chunks = []
            for gs, ps, audio in generator:
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                    chunks.append(audio)
            
            return chunks
        
        chunks = await loop.run_in_executor(None, _run)
        for chunk in chunks:
            yield chunk
    
    @staticmethod
    def _split_on_flush(text: str) -> list[str]:
        import re
        parts = re.split(r'(?<=[.?!,;:])\s*', text)
        return parts if parts else [text]

    

    