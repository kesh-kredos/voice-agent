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

SAMPLE_RATE = 24000    # Orpheus / SNAC output rate
TOKENS_PER_FRAME = 7   # SNAC frame size

# vLLM outputs Orpheus audio tokens as relative indices, not absolute
# vocabulary IDs (128266–156937). Each position in a 7-token SNAC frame maps to
# a specific sub-codebook range; these ranges are used for frame validation.
#
# The official Orpheus decoder adds a per-position +10 offset on top of the
# codebook base, so the valid range at position p is [4096*p + 10, 4096*p + 4105].
_SNAC_RANGES = [
    (10,     4105),   # position 0: layer-1 coarse code
    (4106,   8201),   # position 1: layer-2a
    (8202,   12297),  # position 2: layer-3a
    (12298,  16393),  # position 3: layer-3b
    (16394,  20489),  # position 4: layer-2b
    (20490,  24585),  # position 5: layer-3c
    (24586,  28681),  # position 6: layer-3d
]


def _frame_to_codebooks(frame: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Map one 7-token SNAC frame to (layer1, layer2, layer3) codebook ids.

    Orpheus emits relative token indices with a per-position +10 offset on top
    of the codebook base, so each code is ``token - 10 - 4096 * position``.
    """
    layer1 = [frame[0] - 10]
    layer2 = [frame[1] - 10 - 4096,       frame[4] - 10 - 4096 * 4]
    layer3 = [frame[2] - 10 - 4096 * 2,   frame[3] - 10 - 4096 * 3,
              frame[5] - 10 - 4096 * 5,   frame[6] - 10 - 4096 * 6]
    return layer1, layer2, layer3

# Sentence-boundary flush thresholds — same logic as the Kokoro version.
HARD_FLUSH = {".", "?", "!", "\n"}
SOFT_FLUSH = {",", ";", ":"}
MIN_SOFT_FLUSH_CHARS = 30   # lowered from 60 — flush shorter clauses for faster TTFA
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
            # Check the last *non-whitespace* character so that punctuation
            # followed by trailing spaces (e.g. "Hello. ") still triggers a
            # flush instead of stalling until more text arrives.
            stripped = buffer.rstrip()
            last = stripped[-1] if stripped else ""
            stripped_len = len(stripped)

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
        carry = ""  # holds a partial <custom_token_N> split across SSE chunks

        try:
            # Stream raw completion tokens from Orpheus vLLM.
            # We use /completions (not /chat/completions) because Orpheus
            # is not a chat model — it's a raw token generator.
            stream = await self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1200,
                temperature=0.6,
                top_p=0.8,
                extra_body={"repetition_penalty": 1.3},
                stream=True,
            )

            async for chunk in stream:
                text_piece = chunk.choices[0].text
                if not text_piece:
                    continue

                text_piece = carry + text_piece
                ids, carry = self._parse_token_ids(text_piece)
                token_buffer.extend(ids)

            # Parse any remaining carry (unlikely to be complete, but try)
            if carry:
                ids, _ = self._parse_token_ids(carry)
                token_buffer.extend(ids)

            # Decode the entire clause in one snac.decode call.
            # Calling decode frame-by-frame causes the non-causal dilated
            # ResNet blocks to run without surrounding context on every
            # boundary, producing distortion artifacts at every ~13ms edge.
            # One batch call confines artifacts to the start/end of the clause.
            audio = self._decode_tokens(token_buffer)
            if audio is not None and audio.size > 0:
                yield audio

        except Exception as e:
            logger.error(f"Orpheus synthesis error: {e}")
            raise

    def _parse_token_ids(self, text: str) -> tuple[list[int], str]:
        """Extract all <custom_token_N> IDs from a vLLM completion chunk.

        Returns (ids, carry) where *carry* is any trailing text that looks
        like an incomplete ``<custom_token_N>`` tag split across the SSE
        chunk boundary.  The caller prepends *carry* to the next chunk so
        tokens are never silently dropped.
        """
        ids = [
            int(m.group(1))
            for m in re.finditer(r"<custom_token_(\d+)>", text)
        ]

        # Detect a trailing partial token tag that may complete in the next
        # chunk.  We look for the last '<' in the text and check whether the
        # suffix from that point is a valid prefix of "<custom_token_N>".
        _TAG_PREFIX = "<custom_token_"
        lt_pos = text.rfind("<")
        if lt_pos >= 0:
            tail = text[lt_pos:]
            if tail == _TAG_PREFIX[:len(tail)] or re.match(r"^<custom_token_\d*$", tail):
                return ids, tail

        return ids, ""

    def _decode_tokens(self, token_buffer: list[int]) -> np.ndarray | None:
        """
        Decode all SNAC frames in token_buffer in a single snac.decode call.

        Advancing by 1 on invalid frames handles the model's preamble tokens
        (values in layer-1 range that appear before the first real audio frame).
        Decoding the full clause at once avoids the per-frame boundary artifacts
        that a non-causal dilated ConvNet produces when called frame-by-frame.
        """
        layer1_codes: list[int] = []
        layer2_codes: list[int] = []
        layer3_codes: list[int] = []

        i = 0
        while i + TOKENS_PER_FRAME <= len(token_buffer):
            frame = token_buffer[i:i + TOKENS_PER_FRAME]
            if all(lo <= tok <= hi for tok, (lo, hi) in zip(frame, _SNAC_RANGES)):
                l1, l2, l3 = _frame_to_codebooks(frame)
                layer1_codes += l1
                layer2_codes += l2
                layer3_codes += l3
                i += TOKENS_PER_FRAME
            else:
                i += 1  # skip preamble / control token

        if not layer1_codes:
            return None

        try:
            layer1 = torch.tensor([layer1_codes], dtype=torch.long).clamp(0, 4095)
            layer2 = torch.tensor([layer2_codes], dtype=torch.long).clamp(0, 4095)
            layer3 = torch.tensor([layer3_codes], dtype=torch.long).clamp(0, 4095)

            if torch.cuda.is_available():
                layer1, layer2, layer3 = layer1.cuda(), layer2.cuda(), layer3.cuda()

            with torch.no_grad():
                audio_tensor = self.snac.decode([layer1, layer2, layer3])

            return audio_tensor.cpu().squeeze().numpy().astype(np.float32)

        except Exception as e:
            logger.error(f"SNAC decode error: {e}")
            return None