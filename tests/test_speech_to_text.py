"""Regression tests for models/speech_to_text.py STT generate_kwargs.

The Whisper pipeline crashed at runtime with
``UnboundLocalError: cannot access local variable 'logprobs'`` because
``no_speech_threshold`` and ``compression_ratio_threshold`` were forwarded
in ``generate_kwargs`` without a matching ``logprob_threshold`` in the
installed transformers version. The browser end-of-turn path already has
an upstream energy guard, so these fallback thresholds are not needed to
prevent false turns. These tests pin ``transcribe()`` so the threshold
kwargs cannot regress, while confirming ``language`` is still forwarded.

The heavy ``STTClient.__init__`` (which loads Whisper) is bypassed by
constructing the instance via ``__new__`` and injecting a fake pipeline,
so no real models are loaded.
"""

import unittest

import numpy as np

from models.speech_to_text import STTClient


class _FakePipeline:
    """Records the inputs and generate_kwargs passed to the ASR pipeline."""

    def __init__(self, text: str = "hello world"):
        self._text = text
        self.last_inputs = None
        self.last_generate_kwargs = None

    def __call__(self, inputs, generate_kwargs=None):
        self.last_inputs = inputs
        self.last_generate_kwargs = dict(generate_kwargs or {})
        return {"text": self._text}


class TranscribeGenerateKwargsTests(unittest.TestCase):
    """transcribe() must not pass num_beams or the threshold guard kwargs."""

    def _make_client(self, text: str = "  hello world  "):
        # Bypass __init__ so no Whisper model is loaded.
        stt = STTClient.__new__(STTClient)
        pipeline = _FakePipeline(text=text)
        stt.pipeline = pipeline
        return stt, pipeline

    def test_num_beams_not_passed_to_pipeline(self) -> None:
        stt, pipeline = self._make_client()
        audio = np.zeros(16000, dtype=np.float32)

        stt.transcribe(audio)

        self.assertNotIn(
            "num_beams",
            pipeline.last_generate_kwargs,
            "num_beams must not be passed explicitly; it conflicts with the "
            "pipeline's internal generation config and crashes Whisper.",
        )

    def test_threshold_kwargs_not_passed_to_pipeline(self) -> None:
        stt, pipeline = self._make_client()
        audio = np.zeros(16000, dtype=np.float32)

        stt.transcribe(audio)

        kwargs = pipeline.last_generate_kwargs
        self.assertNotIn(
            "no_speech_threshold",
            kwargs,
            "no_speech_threshold must not be forwarded; without a matching "
            "logprob_threshold it triggers UnboundLocalError: logprobs in "
            "the installed transformers Whisper generation path.",
        )
        self.assertNotIn(
            "compression_ratio_threshold",
            kwargs,
            "compression_ratio_threshold must not be forwarded; it shares "
            "the same crash path as no_speech_threshold in this "
            "transformers version.",
        )

    def test_language_forwarded_and_result_stripped(self) -> None:
        stt, pipeline = self._make_client(text="  hello there  ")
        audio = np.zeros(16000, dtype=np.float32)

        result = stt.transcribe(audio, lang="french")

        self.assertEqual(result, "hello there")
        self.assertEqual(pipeline.last_generate_kwargs.get("language"), "french")
        self.assertIs(pipeline.last_inputs["array"], audio)
        self.assertEqual(pipeline.last_inputs["sampling_rate"], 16000)


if __name__ == "__main__":
    unittest.main()
