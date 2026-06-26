import asyncio 
import logging
import time
import numpy as np
from typing import AsyncGenerator


from models.voice_activity import VADClient
from models.text_to_speech_orpheus import TTSClient
from models.llm import LLMClient, EOC_SIGNALS
from models.speech_to_text import STTClient
from utils.audio import mulaw_to_float32, float32_to_mulaw, pcm16_to_float32, float32_to_pcm16le

logger = logging.getLogger("Agent")

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_CHUNKS = 30
MIC_OPEN_DELAY = 0.5
# Consecutive speech frames required before an utterance starts, so a single
# noisy/echo VAD frame cannot trigger a full turn (~96ms at 16kHz/512).
ONSET_FRAMES = 3
# Minimum RMS (float32 [-1,1]) for an utterance to reach STT. Rejects the
# echo/noise floor that Whisper hallucinates on (observed ~0.003 RMS).
ENERGY_RMS_THRESHOLD = 0.01
# Bounded-duration fallback for a stalled stream: if VAD has been "speaking"
# for longer than this without finalizing (no trailing silence and no explicit
# end-of-turn), force-finalize so the turn is not lost.
MAX_UTTERANCE_SECONDS = 15.0

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
        self._speaking: bool = False
        self._force_end_of_utterance: bool = False

        logger.info(f"New agent initialized for {customer_ctx.get('customer_name', 'TEST CUSTOMER')}")


    def enqueue_audio(self, raw_bytes: bytes):
        self._audio_queue.put_nowait(raw_bytes)

    def force_end_of_utterance(self) -> None:
        """Signal _utterance_stream() to yield the buffered utterance
        immediately.

        The browser sends an explicit end-of-turn control message on
        push-to-talk release because the frontend RMS gate suppresses the
        trailing silence VAD needs to finalize on its own.
        """
        self._force_end_of_utterance = True

    async def run(self, send_audio_callback):
        logger.info(
            f"New voice session started - {self.customer_ctx.get('customer_name', 'TEST CUSTOMER')}"
            f'[mode={self.mode}]'
        )

        await self._open_call(send_audio_callback)

        if not self._call_ended:
            async for utterance in self._utterance_stream():
                if self._call_ended:
                    break
                await self._handle_turn(utterance, send_audio_callback)
                if self._call_ended:
                    break
    
    async def _finish_speaking(self) -> None:
        await asyncio.sleep(MIC_OPEN_DELAY)
        flushed = 0
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                flushed += 1
            except asyncio.QueueEmpty:
                break
        
        self._speaking = False
        if hasattr(self.vad, "reset_states"):
            self.vad.reset_states()
        logger.info(f"Mic re-opened, flushed {flushed} echo frames")

    async def _open_call(self, send_audio):
        try: 
            logger.info('Opening the call')
            prompt = (
                "[CALL_START] You are initiating this outbound call. "
                "Greet the customer and state the reason for the call. "
                "Do not reveal account details until identity is verified."
            )
            collected: list[str] = []

            async def collect_stream():
                async for token in self.llm.stream(
                    prompt, self.history, self.customer_ctx
                ):
                    collected.append(token)
                    yield token
            
            self._speaking = True
            try:
                await self._stream_tts_audio(self.tts.stream(collect_stream()), send_audio)
            finally:
                await self._finish_speaking()

            response = ''.join(collected)

            for signal in EOC_SIGNALS:
                if signal in response:
                    self.call_status = signal.split(':')[-1]
                    self._call_ended = True
                    return
            
            clean = response
            for signal in EOC_SIGNALS:
                clean = clean.replace(signal, "").strip()
            
            self.history.append({'role': 'assistant', 'content': clean})
            logger.info(f'Opening line: {clean!r}')
        except Exception as e:
            self._speaking = False
            logger.error(f"Call opening failed with error: {e}", exc_info=True)

        

    def _decode_audio(self, raw_bytes: bytes) -> np.ndarray:
        if self.mode == 'twilio':
            return mulaw_to_float32(raw_bytes)
        else:
            return pcm16_to_float32(raw_bytes)

    async def _utterance_stream(self) -> AsyncGenerator[np.ndarray, None]:
        vad_buff: list[np.ndarray] = []
        onset_buff: list[np.ndarray] = []
        onset_count = 0
        silence_count = 0
        speaking = False
        utterance_start: float = 0.0
        rolling_buffer = np.array([], dtype=np.float32)

        while not self._call_ended:
            # Explicit end-of-turn from the browser (push-to-talk release):
            # finalize the buffered utterance immediately so STT runs without
            # waiting for VAD trailing silence the frontend RMS gate may have
            # suppressed.
            if self._force_end_of_utterance:
                self._force_end_of_utterance = False
                if speaking and vad_buff:
                    logger.info(f"End-of-turn signal, yielding {len(vad_buff)} frames")
                    yield np.concatenate(vad_buff)
                    vad_buff = []
                    silence_count = 0
                    speaking = False
                    utterance_start = 0.0
                    logger.info("VAD reset, listening for next utterance")
                continue

            # Bounded-duration fallback: if VAD has been "speaking" for longer
            # than MAX_UTTERANCE_SECONDS without finalizing (e.g. a stalled
            # stream with no trailing silence and no end-of-turn signal),
            # finalize so the turn is not lost.
            if speaking and utterance_start and (time.perf_counter() - utterance_start) >= MAX_UTTERANCE_SECONDS:
                logger.info(f"Utterance exceeded {MAX_UTTERANCE_SECONDS:.1f}s, force-finalizing {len(vad_buff)} frames")
                yield np.concatenate(vad_buff)
                vad_buff = []
                silence_count = 0
                speaking = False
                utterance_start = 0.0
                logger.info("VAD reset, listening for next utterance")
                continue

            try:
                raw_audio = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            audio_chunk = self._decode_audio(raw_audio)

            if self._speaking:
                continue

            rolling_buffer = np.concatenate([rolling_buffer, audio_chunk])
            logger.info(f"Audio received: chunk_len={len(audio_chunk)} buffer_len={len(rolling_buffer)}")

            while len(rolling_buffer) >= CHUNK_SIZE:
                frame = rolling_buffer[:CHUNK_SIZE]
                rolling_buffer = rolling_buffer[CHUNK_SIZE:]

                is_speech = self.vad.is_speech(frame)
                logger.info(f"VAD: is_speech={is_speech} speaking={speaking} silence={silence_count} onset={onset_count}")
                if is_speech:
                    if speaking:
                        silence_count = 0
                        vad_buff.append(frame)
                    else:
                        onset_count += 1
                        onset_buff.append(frame)
                        if onset_count >= ONSET_FRAMES:
                            speaking = True
                            vad_buff.extend(onset_buff)
                            onset_buff = []
                            onset_count = 0
                            silence_count = 0
                            utterance_start = time.perf_counter()
                            logger.info(f"VAD onset confirmed after {ONSET_FRAMES} frames, capturing utterance")
                elif speaking:
                    silence_count += 1
                    vad_buff.append(frame)
                    if silence_count >= SILENCE_CHUNKS:
                        logger.info(f"Utterance complete, yielding {len(vad_buff)} frames")
                        yield np.concatenate(vad_buff)
                        vad_buff = []
                        silence_count = 0
                        speaking = False
                        utterance_start = 0.0
                        logger.info("VAD reset, listening for next utterance")
                else:
                    onset_count = 0
                    onset_buff = []


        
    
    def _is_low_energy(self, utterance: np.ndarray) -> bool:
        """Reject the echo/noise floor before spending a Whisper call.

        Whisper hallucinates plausible text (e.g. "Thank you.") on near-silence,
        so callers that transcribe up front (the browser server wrapper) must
        gate on this before invoking STT.
        """
        rms = float(np.sqrt(np.mean(np.square(utterance))))
        if rms < ENERGY_RMS_THRESHOLD:
            logger.info(f"Low-energy utterance (rms={rms:.5f} < {ENERGY_RMS_THRESHOLD}), skipping STT")
            return True
        return False

    async def _handle_turn(self, utterance: np.ndarray, send_audio, transcript: str | None = None):
        logger.info(f"Handling turn for {self.customer_ctx.get('customer_name', 'Sarah Johnson')}")
        start = time.perf_counter()

        logger.info(f"Utterance stats: len={len(utterance)} min={utterance.min():.4f} max={utterance.max():.4f} mean={utterance.mean():.4f}")

        if self._is_low_energy(utterance):
            return

        # Allow the caller (e.g. the browser server, which transcribes up front to
        # emit a transcript event) to pass the transcript in so we don't run
        # Whisper a second time on the same utterance.
        if transcript is None:
            transcript = await asyncio.get_event_loop().run_in_executor(
                None, self.stt.transcribe, utterance
            )
        logger.info(f"STT trascribed as {transcript!r}")
        if not transcript.strip():
            logger.info("Empty transcript from STT, skipping")
            return
        
        logger.info(f"Customer {self.customer_ctx.get('customer_name', 'TEST CUSTOMER')} said: '{transcript}'")

        if self._should_escalate(transcript):
            await self._escalate(send_audio)
            self._call_ended = True
            self.call_status = "escalated"
            return
        
        tokens: list[str] = []


        logger.info(f"Calling LLM with transcript: {transcript!r}")
        
        async def token_stream_with_collection():
            async for tok in self.llm.stream(transcript, self.history, self.customer_ctx):
                tokens.append(tok)
                yield tok
        
        self._speaking = True

        try:
            await self._stream_tts_audio(self.tts.stream(token_stream_with_collection()), send_audio)
        finally:
            await self._finish_speaking()
        
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

    
    async def _stream_tts_audio(self, token_gen, send_audio):
        """Consume a TTS token stream, encode audio, and send each chunk immediately."""
        async for audio_chunk in token_gen:
            await send_audio(self._audio_encode(audio_chunk))

    def _audio_encode(self, audio: np.ndarray) -> bytes:
        if self.mode == 'twilio':
            return float32_to_mulaw(audio)
        else:
            return float32_to_pcm16le(audio)
    
    def _should_escalate(self, transcript: str) -> bool:
        phrases = [
            "speak to a human", "speak to someone", "real person",
            "agent please", "transfer me", "supervisor"
        ]

        return any(p in transcript.lower() for p in phrases)
    
    async def _escalate(self, send_audio):
        message = (
            "Of course, let me transfer you to one of our agents right now. "
            "Please hold for just a moment."
        )

        self._speaking = True
        try:
            await self._stream_tts_audio(self.tts.stream(self._as_token_stream(message)), send_audio)
        finally:
            await self._finish_speaking()

        logger.info("Escalation was detected")
    
    @staticmethod
    async def _as_token_stream(text: str) -> AsyncGenerator[str, None]:
        yield text
