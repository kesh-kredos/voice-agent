import numpy as np
import torch 
import logging
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger("STTClient")

MODEL_ID = 'openai/whisper-large-v3-turbo'

class STTClient:
    def __init__(self, device: str = None, torch_dtype = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

        logger.info(f"Loading {MODEL_ID} on {self.device} w/ {self.torch_dtype}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device
        )

        logger.info(f"{MODEL_ID} loaded")

    
    def transcribe(self, audio: np.ndarray, lang: str = "english") -> str:
        start = time.perf_counter()

        res = self.pipeline(
            {'array': audio, 'sampling_rate': 16000},
            generate_kwargs={
                'language': lang,
                'num_beams': 1,
                # 'condition_on_previous_tokens': False,
                # "compression_ratio_threshold": 1.35,
                # "no_speech_threshold": 0.6,
            }
        )

        txt = res['text'].strip()
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"Transcription took {elapsed:.2f}ms: {txt}")

        return txt
