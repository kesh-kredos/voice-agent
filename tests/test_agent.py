"""Regression tests for agent.VoiceAgent audio-state handling."""

import asyncio
import unittest
from typing import AsyncGenerator

import numpy as np

import agent as agent_module
from agent import (
    VoiceAgent,
    CHUNK_SIZE,
    SILENCE_CHUNKS,
    ONSET_FRAMES,
    ENERGY_RMS_THRESHOLD,
)


class _StubVAD:
    def is_speech(self, frame):
        return False


class _StubSTT:
    def transcribe(self, utterance):
        return ""


class _RaisingLLM:
    """LLM stub whose stream always raises before yielding any token."""

    async def stream(
        self, prompt: str, history: list, ctx: dict
    ) -> AsyncGenerator[str, None]:
        raise RuntimeError("llm boom")
        yield  # pragma: no cover – makes this an async generator


class _PassThroughTTS:
    """TTS stub that iterates its token stream and yields dummy audio."""

    async def stream(
        self, token_gen: AsyncGenerator[str, None]
    ) -> AsyncGenerator[np.ndarray, None]:
        async for _ in token_gen:
            yield np.zeros(240, dtype=np.float32)


class _ResetRecordingVAD:
    """VAD stub that records reset_states() calls."""

    def __init__(self):
        self.reset_count = 0

    def is_speech(self, frame):
        return False

    def reset_states(self):
        self.reset_count += 1


class _ScriptedVAD:
    """VAD stub returning a scripted is_speech() sequence.

    Sets the bound agent's _call_ended flag after the last scripted value so
    _utterance_stream() terminates deterministically without hitting the queue
    timeout.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self._idx = 0
        self._agent = None

    def is_speech(self, frame):
        val = self._frames[self._idx]
        self._idx += 1
        if self._idx >= len(self._frames) and self._agent is not None:
            self._agent._call_ended = True
        return val


class _AlwaysSpeechVAD:
    """VAD stub that always reports speech."""

    def is_speech(self, frame):
        return True


class _OnsetSignalVAD:
    """VAD stub that always reports speech and sets an asyncio.Event once
    ONSET_FRAMES frames have been seen.

    Lets a test await the event to know _utterance_stream() has confirmed
    onset (speaking=True) before triggering a force-end.
    """

    def __init__(self):
        self._count = 0
        self.onset = asyncio.Event()

    def is_speech(self, frame):
        self._count += 1
        if self._count >= ONSET_FRAMES and not self.onset.is_set():
            self.onset.set()
        return True


class _RecordingSTT:
    """STT stub that records transcribe() calls."""

    def __init__(self):
        self.transcribe_calls = 0

    def transcribe(self, utterance):
        self.transcribe_calls += 1
        return ""


class _RecordingLLM:
    """LLM stub that records stream() calls."""

    def __init__(self):
        self.stream_calls = 0

    async def stream(self, prompt, history, ctx):
        self.stream_calls += 1
        yield "should not be reached"


class OpenCallSpeakingStateTests(unittest.TestCase):
    """_open_call must clear _speaking even when the TTS pipeline raises."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_agent(self) -> VoiceAgent:
        return VoiceAgent(
            vad=_StubVAD(),
            stt=_StubSTT(),
            llm=_RaisingLLM(),
            tts=_PassThroughTTS(),
            customer_ctx={"name": "TEST"},
            mode="browser",
        )

    def test_speaking_cleared_on_open_call_exception(self) -> None:
        agent = self._make_agent()
        self.assertFalse(agent._speaking)

        async def send_audio(_chunk):
            pass

        async def drive():
            await agent._open_call(send_audio)

        self._run(drive())
        self.assertFalse(
            agent._speaking,
            "_speaking must be False after _open_call raises",
        )


class FinishSpeakingTests(unittest.TestCase):
    """_finish_speaking() must reset VAD state and clear _speaking."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_finish_speaking_resets_vad_and_clears_flag(self) -> None:
        vad = _ResetRecordingVAD()
        agent = VoiceAgent(
            vad=vad,
            stt=_StubSTT(),
            llm=_RaisingLLM(),
            tts=_PassThroughTTS(),
            customer_ctx={"name": "TEST"},
            mode="browser",
        )
        agent._speaking = True

        async def drive():
            agent.enqueue_audio(b"\x00" * (CHUNK_SIZE * 2))
            agent.enqueue_audio(b"\x00" * (CHUNK_SIZE * 2))
            await agent._finish_speaking()

        self._run(drive())

        self.assertFalse(agent._speaking)
        self.assertEqual(vad.reset_count, 1)
        self.assertTrue(agent._audio_queue.empty())


class LowEnergyTurnTests(unittest.TestCase):
    """_handle_turn() must skip STT/LLM for utterances below the RMS threshold."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_agent(self):
        stt = _RecordingSTT()
        llm = _RecordingLLM()
        agent = VoiceAgent(
            vad=_StubVAD(),
            stt=stt,
            llm=llm,
            tts=_PassThroughTTS(),
            customer_ctx={"name": "TEST"},
            mode="browser",
        )
        return agent, stt, llm

    def test_low_energy_utterance_skips_stt_and_llm(self) -> None:
        agent, stt, llm = self._make_agent()
        # RMS == ENERGY_RMS_THRESHOLD / 10, comfortably below the cutoff.
        utterance = np.full(CHUNK_SIZE, ENERGY_RMS_THRESHOLD / 10, dtype=np.float32)

        async def send_audio(_chunk):
            pass

        async def drive():
            await agent._handle_turn(utterance, send_audio)

        self._run(drive())

        self.assertEqual(stt.transcribe_calls, 0)
        self.assertEqual(llm.stream_calls, 0)


class UtteranceStreamOnsetTests(unittest.TestCase):
    """_utterance_stream() must require ONSET_FRAMES consecutive speech frames
    before capturing an utterance, so a single noisy VAD frame cannot trigger a
    full turn."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_agent(self, frames):
        vad = _ScriptedVAD(frames)
        agent = VoiceAgent(
            vad=vad,
            stt=_StubSTT(),
            llm=_RaisingLLM(),
            tts=_PassThroughTTS(),
            customer_ctx={"name": "TEST"},
            mode="browser",
        )
        vad._agent = agent
        return agent

    def _collect(self, agent, num_chunks):
        # 512 int16 samples -> exactly one CHUNK_SIZE frame in browser mode.
        chunk = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

        async def drive():
            for _ in range(num_chunks):
                agent.enqueue_audio(chunk)
            results = []
            async for utt in agent._utterance_stream():
                results.append(utt)
            return results

        return self._run(drive())

    def test_single_true_frame_does_not_yield(self) -> None:
        # One speech frame, then enough silence to otherwise complete a turn.
        # A single frame cannot meet the ONSET_FRAMES requirement.
        frames = [True] + [False] * SILENCE_CHUNKS
        agent = self._make_agent(frames)
        results = self._collect(agent, len(frames))
        self.assertEqual(results, [])

    def test_three_true_frames_then_silence_yields(self) -> None:
        frames = [True] * ONSET_FRAMES + [False] * SILENCE_CHUNKS
        agent = self._make_agent(frames)
        results = self._collect(agent, len(frames))
        self.assertEqual(len(results), 1)
        # ONSET_FRAMES speech frames + SILENCE_CHUNKS trailing silence frames.
        expected_samples = (ONSET_FRAMES + SILENCE_CHUNKS) * CHUNK_SIZE
        self.assertEqual(len(results[0]), expected_samples)


class ForceEndOfUtteranceTests(unittest.TestCase):
    """force_end_of_utterance() must make _utterance_stream() yield the
    buffered utterance immediately, without waiting for SILENCE_CHUNKS
    trailing silence — the browser sends this on push-to-talk release
    because the frontend RMS gate suppresses that silence."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_agent(self, vad):
        return VoiceAgent(
            vad=vad,
            stt=_StubSTT(),
            llm=_RaisingLLM(),
            tts=_PassThroughTTS(),
            customer_ctx={"name": "TEST"},
            mode="browser",
        )

    def test_force_end_yields_buffered_utterance(self) -> None:
        vad = _OnsetSignalVAD()
        agent = self._make_agent(vad)
        chunk = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

        async def drive():
            for _ in range(ONSET_FRAMES):
                agent.enqueue_audio(chunk)

            results: list[np.ndarray] = []

            async def collect():
                async for utt in agent._utterance_stream():
                    results.append(utt)
                    agent._call_ended = True

            task = asyncio.create_task(collect())
            # Wait until onset is confirmed (speaking=True, frames buffered)
            # before signalling end-of-turn.
            await vad.onset.wait()
            agent.force_end_of_utterance()
            await asyncio.wait_for(task, timeout=5.0)
            return results

        results = self._run(drive())
        self.assertEqual(len(results), 1)
        # Only the ONSET_FRAMES speech frames — no trailing silence needed.
        self.assertEqual(len(results[0]), ONSET_FRAMES * CHUNK_SIZE)

    def test_force_end_with_no_active_utterance_is_noop(self) -> None:
        agent = self._make_agent(_StubVAD())

        async def drive():
            agent.force_end_of_utterance()
            results: list[np.ndarray] = []

            async def collect():
                async for utt in agent._utterance_stream():
                    results.append(utt)

            task = asyncio.create_task(collect())
            # Give the generator one loop iteration to consume the flag,
            # then terminate the stream.
            await asyncio.sleep(0.1)
            agent._call_ended = True
            await asyncio.wait_for(task, timeout=5.0)
            return results

        results = self._run(drive())
        self.assertEqual(results, [])


class MaxUtteranceDurationTests(unittest.TestCase):
    """_utterance_stream() must force-finalize a stalled utterance after
    MAX_UTTERANCE_SECONDS so a stream that never receives trailing silence
    (and no end-of-turn signal) is not lost."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_stalled_utterance_is_force_finalized(self) -> None:
        agent = VoiceAgent(
            vad=_AlwaysSpeechVAD(),
            stt=_StubSTT(),
            llm=_RaisingLLM(),
            tts=_PassThroughTTS(),
            customer_ctx={"name": "TEST"},
            mode="browser",
        )
        chunk = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

        async def drive():
            for _ in range(ONSET_FRAMES):
                agent.enqueue_audio(chunk)

            results: list[np.ndarray] = []

            async def collect():
                async for utt in agent._utterance_stream():
                    results.append(utt)
                    agent._call_ended = True

            task = asyncio.create_task(collect())
            await asyncio.wait_for(task, timeout=5.0)
            return results

        # Shrink the bound so the test doesn't wait the full 15s. The queue
        # timeout is 1s, so the force-finalize fires on the first timeout.
        original = agent_module.MAX_UTTERANCE_SECONDS
        agent_module.MAX_UTTERANCE_SECONDS = 0.05
        try:
            results = self._run(drive())
        finally:
            agent_module.MAX_UTTERANCE_SECONDS = original

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), ONSET_FRAMES * CHUNK_SIZE)


if __name__ == "__main__":
    unittest.main()
