import numpy as np
import logging
import torch
from typing import Generator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_CHUNKS = 22

class VoiceActivityDetector:
    def __init__(self, threshold: float = 0.5, silence_chunks: int = SILENCE_CHUNKS):
        self.threshold = threshold
        self.silence_chunks = silence_chunks
        self._load_model()

    def _load_model(self):
        logger.info("LOADING SILERO VOICE ACTIVITY DETECTOR")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        self.model.eval()
        logger.info("SILERO VAD LOADED")

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

