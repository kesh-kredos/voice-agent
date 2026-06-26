import numpy as np
import logging
import torch
from typing import Generator

logger = logging.getLogger("VADClient")

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_CHUNKS = 22

class VADClient:
    def __init__(self, threshold: float = 0.5, silence_chunks: int = SILENCE_CHUNKS):
        self.threshold = threshold
        self.silence_chunks = silence_chunks
        self._load_model()

    def _load_model(self):
        logger.info("Loading Silero VAD")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        self.model.eval()
        logger.info("Silero VAD loaded")

    def _chunk_to_tensor(self, chunk: bytes) -> torch.Tensor:
        audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio_data)
    
    def iter_utterances(self, audio_stream: Generator[bytes, None, None]):
        buffer = []
        silence_count = 0
        speaking = False

        for chunk in audio_stream:
            audio_tensor = self._chunk_to_tensor(chunk)

            if audio_tensor.shape[0] != CHUNK_SIZE:
                continue

            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

            if speech_prob >= self.threshold:
                speaking = True
                silence_count = 0
                buffer.append(audio_tensor.numpy())

            elif speaking: 
                buffer.append(audio_tensor.numpy())
                silence_count += 1

                if silence_count >= self.silence_chunks:
                    noise = np.concatenate(buffer)
                    logger.debug(f"End of speech — {len(noise)/SAMPLE_RATE:.2f}s")
                    yield noise
                    buffer = []
                    silence_count = 0
                    speaking = False

    def is_speech(self, frame: np.ndarray) -> bool:
        tensor = torch.from_numpy(frame.astype(np.float32))
        return self.model(tensor, SAMPLE_RATE).item() >= self.threshold

    def reset_states(self) -> None:
        """Clear Silero internal LSTM state. Call after TTS so echo leakage
        does not prime the model into a spurious speech onset."""
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
