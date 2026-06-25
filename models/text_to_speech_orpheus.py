"""
models/tts_orpheus.py
─────────────────────
Text-to-speech using Orpheus-3B served via vLLM on port 8001.

Does NOT use the orpheus-speech pip package — that package hardcodes
gpu_memory_utilization=0.92 and assumes it owns the whole GPU. Instead
this module calls the Orpheus vLLM server directly over the OpenAI-
compatible HTTP API, exactly the same pattern as llm.py calls LLaMA.

The only in-process dependency is the SNAC decoder, which converts the
raw audio token stream from vLLM into float32 PCM audio.

Install:
    pip install snac openai

Architecture:
    text → vLLM (Orpheus, port 8001) → SNAC token stream → float32 PCM
    agent.py then encodes PCM to mulaw (Twilio) or WAV (browser).

Orpheus token format:
    The model outputs special tokens in the range 128266–156938.
    Every 7 tokens = 1 SNAC frame = a small chunk of audio at 24kHz.
    We accumulate tokens in groups of 7, decode each group immediately,
    and yield the resulting audio — this is what gives streaming latency.
"""

import asyncio
import logging
import re
import numpy as np
import torch
from typing import AsyncGenerator
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000          # Orpheus / SNAC output rate
AUDIO_TOKEN_OFFSET = 128266  # Orpheus audio tokens start here
TOKENS_PER_FRAME = 7         # SNAC frame size

# Sentence-boundary flush thresholds — same logic as the Kokoro version.
HARD_FLUSH = {".", "?", "!", "\n"}
SOFT_FLUSH = {",", ";", ":"}
MIN_SOFT_FLUSH_CHARS = 60
MIN_HARD_FLUSH_CHARS = 2

# Strip any "voice: " prefix the model might accidentally echo back
_LEADING_VOICE = re.compile(
    r"^\s*(tara|leah|jess|leo|dan|mia|zac|zoe)\s*:\s*", re.I
)


class TTSClient:
    """
    Streams audio from Orpheus-3B running on a separate vLLM process.
    Same public interface as the Kokoro TTSClient:
      - __init__(voice, ...)
      - stream(token_stream) -> AsyncGenerator[np.ndarray, None]
    """

    AVAILABLE_VOICES = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")

    def __init__(
        self,
        voice: str = "tara",
        base_url: str = "http://localhost:8001/v1",
        model_name: str = "canopylabs/orpheus-3b-0.1-ft",
    ):
        if voice not in self.AVAILABLE_VOICES:
            logger.warning(
                f"Voice '{voice}' not in known Orpheus voices "
                f"{self.AVAILABLE_VOICES}; passing through anyway."
            )
        self.voice = voice
        self.model_name = model_name

        # OpenAI-compatible client pointed at Orpheus vLLM on port 8001
        self.client = AsyncOpenAI(base_url=base_url, api_key="not-needed")

        # Load SNAC decoder in-process — it's small (~50MB) and fast
        logger.info("Loading SNAC decoder...")
        from snac import SNAC
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        if torch.cuda.is_available():
            self.snac = self.snac.cuda()
        logger.info(f"Orpheus TTSClient ready — voice={self.voice}")

    # ── Public API ────────────────────────────────────────────────────────────

    async def stream(
        self,
        token_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Consume a stream of LLM text tokens, synthesise at clause boundaries,
        and yield 24kHz float32 numpy arrays as they're ready.

        agent.py encodes the float32 output to mulaw (Twilio) or WAV (browser).
        """
        buffer = ""

        async for token in token_stream:
            if not token:
                continue
            buffer += token

            flush_text = None
            last = buffer[-1]
            stripped_len = len(buffer.strip())

            if last in HARD_FLUSH and stripped_len >= MIN_HARD_FLUSH_CHARS:
                flush_text = buffer
                buffer = ""
            elif last in SOFT_FLUSH and stripped_len >= MIN_SOFT_FLUSH_CHARS:
                flush_text = buffer
                buffer = ""

            if flush_text:
                async for audio in self._synthesise(flush_text):
                    yield audio

        # Flush remainder at end of stream
        if buffer.strip():
            async for audio in self._synthesise(buffer):
                yield audio

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _synthesise(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """
        Send one clause to the Orpheus vLLM server and stream back decoded
        audio chunks as float32 arrays.
        """
        clean = _LEADING_VOICE.sub("", text).strip()
        if not clean:
            return

        # Orpheus finetune-prod expects "voice: text" as the prompt
        prompt = f"<|audio|>{self.voice}: {clean}<|eot_id|>"

        logger.debug(f"TTS <- '{clean}'")

        token_buffer: list[int] = []

        try:
            # Stream raw completion tokens from Orpheus vLLM.
            # We use /completions (not /chat/completions) because Orpheus
            # is not a chat model — it's a raw token generator.
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1200,
                temperature=0.6,
                top_p=0.9,
                extra_body={"repetition_penalty": 1.1},
                stream=True,
            )

            async for chunk in stream:
                text_piece = chunk.choices[0].text
                logger.info(f"CHUNK: {repr(text_piece)}")
                if not text_piece:
                    continue

                # Parse audio token IDs from the token text.
                # Orpheus audio tokens appear as <custom_token_NNNNN>.
                token_ids = self._parse_token_ids(text_piece)
                if not token_ids:
                    continue

                token_buffer.extend(token_ids)

                # Decode every complete 7-token SNAC frame immediately
                while len(token_buffer) >= TOKENS_PER_FRAME:
                    frame_tokens = token_buffer[:TOKENS_PER_FRAME]
                    token_buffer = token_buffer[TOKENS_PER_FRAME:]
                    audio = self._decode_frame(frame_tokens)
                    if audio is not None and audio.size > 0:
                        yield audio

            # Decode any leftover complete frames
            while len(token_buffer) >= TOKENS_PER_FRAME:
                frame_tokens = token_buffer[:TOKENS_PER_FRAME]
                token_buffer = token_buffer[TOKENS_PER_FRAME:]
                audio = self._decode_frame(frame_tokens)
                if audio is not None and audio.size > 0:
                    yield audio

        except Exception as e:
            logger.error(f"Orpheus synthesis error: {e}")
            raise

    def _parse_token_ids(self, text: str) -> list[int]:
        """
        Orpheus audio tokens come back as strings like '<custom_token_28631>'.
        Extract the numeric ID and return only tokens in the audio range.
        """
        ids = []
        for match in re.finditer(r"<custom_token_(\d+)>", text):
            token_id = int(match.group(1))
            if token_id >= AUDIO_TOKEN_OFFSET:
                ids.append(token_id)
        return ids

    def _decode_frame(self, frame_tokens: list[int]) -> np.ndarray | None:
        """
        Decode one 7-token SNAC frame to float32 PCM.

        SNAC uses a 3-level residual codec. The 7 tokens per frame map to:
          token[0]                        -> layer 1 (1 code)
          token[1], token[4]              -> layer 2 (2 codes)
          token[2], token[3], token[5], token[6] -> layer 3 (4 codes)
        Each token is offset by a per-layer base to get the actual code index.
        """
        try:
            codes = [t - AUDIO_TOKEN_OFFSET for t in frame_tokens]

            # Skip frames with out-of-range codes
            if any(c < 0 or c > 4096 * 7 for c in codes):
                return None

            layer1 = torch.tensor([codes[0]],                          dtype=torch.long)
            layer2 = torch.tensor([codes[1] - 4096,
                                   codes[4] - 4096 * 4],               dtype=torch.long)
            layer3 = torch.tensor([codes[2] - 4096 * 2,
                                   codes[3] - 4096 * 3,
                                   codes[5] - 4096 * 5,
                                   codes[6] - 4096 * 6],               dtype=torch.long)

            layer1 = layer1.clamp(0, 4095)
            layer2 = layer2.clamp(0, 4095)
            layer3 = layer3.clamp(0, 4095)

            if torch.cuda.is_available():
                layer1 = layer1.cuda()
                layer2 = layer2.cuda()
                layer3 = layer3.cuda()

            with torch.no_grad():
                audio_tensor = self.snac.decode(
                    [layer1.unsqueeze(0),
                     layer2.unsqueeze(0),
                     layer3.unsqueeze(0)]
                )

            return audio_tensor.cpu().squeeze().numpy().astype(np.float32)

        except Exception as e:
            logger.debug(f"SNAC decode error on frame {frame_tokens}: {e}")
            return None