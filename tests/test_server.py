"""Regression tests for the browser websocket wrapper in server.py.

The browser wrapper transcribes up front to emit a transcript event, which
previously ran Whisper before the agent's energy guard and so bypassed the
false-turn protection on the actual browser path. These tests pin the shared
guard so that bypass cannot regress.
"""

import asyncio
import unittest

import numpy as np

from agent import VoiceAgent, CHUNK_SIZE, ENERGY_RMS_THRESHOLD
from server import (
    _browser_handle_turn_with_events,
    _browser_open_call_with_events,
)


class _StubVAD:
    def is_speech(self, frame):
        return False


class _RecordingSTT:
    """STT stub that records transcribe() calls and returns a fixed transcript."""

    def __init__(self, transcript: str = "hello"):
        self.transcribe_calls = 0
        self._transcript = transcript

    def transcribe(self, utterance):
        self.transcribe_calls += 1
        return self._transcript


class _RecordingLLM:
    """LLM stub that records stream() calls."""

    def __init__(self):
        self.stream_calls = 0

    async def stream(self, prompt, history, ctx):
        self.stream_calls += 1
        yield "should not be reached"


class _PassThroughTTS:
    async def stream(self, token_gen):
        async for _ in token_gen:
            yield np.zeros(240, dtype=np.float32)


class _ScriptedLLM:
    """LLM stub that yields a fixed token sequence on every stream() call."""

    def __init__(self, tokens):
        self._tokens = list(tokens)
        self.stream_calls = 0

    async def stream(self, prompt, history, ctx):
        self.stream_calls += 1
        for tok in self._tokens:
            yield tok


class _RaisingLLM:
    """LLM stub whose stream always raises before yielding any token."""

    async def stream(self, prompt, history, ctx):
        raise RuntimeError("llm boom")
        yield  # pragma: no cover – makes this an async generator


class BrowserWrapperEnergyGuardTests(unittest.TestCase):
    """The browser wrapper must reject low-energy utterances before STT."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_agent(self, transcript: str = "hello"):
        stt = _RecordingSTT(transcript=transcript)
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

    def test_low_energy_utterance_skips_stt_in_wrapper(self) -> None:
        agent, stt, llm = self._make_agent()
        # RMS == ENERGY_RMS_THRESHOLD / 10, comfortably below the cutoff.
        utterance = np.full(CHUNK_SIZE, ENERGY_RMS_THRESHOLD / 10, dtype=np.float32)

        events: list = []
        delegated: list = []

        async def send_audio(_chunk):
            pass

        async def send_event(event_type, **kwargs):
            events.append((event_type, kwargs))

        async def original_handle_turn(utterance, send_audio_cb, transcript=None):
            delegated.append(transcript)

        async def drive():
            await _browser_handle_turn_with_events(
                agent, utterance, send_audio, send_event, original_handle_turn
            )

        self._run(drive())

        self.assertEqual(stt.transcribe_calls, 0)
        self.assertEqual(llm.stream_calls, 0)
        self.assertEqual(events, [])
        self.assertEqual(delegated, [])

    def test_high_energy_utterance_runs_stt_and_emits_transcript(self) -> None:
        agent, stt, llm = self._make_agent(transcript="hello there")
        # RMS == ENERGY_RMS_THRESHOLD * 10, comfortably above the cutoff.
        utterance = np.full(CHUNK_SIZE, ENERGY_RMS_THRESHOLD * 10, dtype=np.float32)

        events: list = []
        delegated: list = []

        async def send_audio(_chunk):
            pass

        async def send_event(event_type, **kwargs):
            events.append((event_type, kwargs))

        async def original_handle_turn(utterance, send_audio_cb, transcript=None):
            delegated.append(transcript)

        async def drive():
            await _browser_handle_turn_with_events(
                agent, utterance, send_audio, send_event, original_handle_turn
            )

        self._run(drive())

        self.assertEqual(stt.transcribe_calls, 1)
        self.assertEqual(events, [("transcript", {"text": "hello there"})])
        # The wrapper must forward the transcript it already computed so the
        # agent doesn't re-run Whisper on the same utterance.
        self.assertEqual(delegated, ["hello there"])


class AgentTextEmissionTests(unittest.TestCase):
    """The browser wrappers must emit agent_text for the opening line and for
    each new assistant reply, without duplicating the same text."""

    def _run(self, coro):
        return asyncio.run(coro)

    def _make_agent(self, tokens):
        llm = _ScriptedLLM(tokens)
        agent = VoiceAgent(
            vad=_StubVAD(),
            stt=_RecordingSTT(transcript="hello"),
            llm=llm,
            tts=_PassThroughTTS(),
            customer_ctx={"name": "TEST"},
            mode="browser",
        )
        return agent, llm

    def _make_recorders(self):
        events: list = []

        async def send_audio(_chunk):
            pass

        async def send_event(event_type, **kwargs):
            events.append((event_type, kwargs))

        return events, send_audio, send_event

    def test_open_call_emits_agent_text_for_opening_line(self) -> None:
        agent, llm = self._make_agent(["Hi there.", " How can I help?"])
        events, send_audio, send_event = self._make_recorders()
        last_emitted = [None]

        async def drive():
            await _browser_open_call_with_events(
                agent, send_audio, send_event, agent._open_call, last_emitted
            )

        self._run(drive())

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(
            events, [("agent_text", {"text": "Hi there. How can I help?"})]
        )
        self.assertEqual(last_emitted[0], "Hi there. How can I help?")

    def test_handle_turn_emits_agent_text_for_assistant_reply(self) -> None:
        agent, _ = self._make_agent(["Sure thing.", " Let me check."])
        events, send_audio, send_event = self._make_recorders()
        last_emitted = [None]
        # RMS comfortably above the cutoff so the wrapper runs STT.
        utterance = np.full(CHUNK_SIZE, ENERGY_RMS_THRESHOLD * 10, dtype=np.float32)

        async def drive():
            await _browser_handle_turn_with_events(
                agent, utterance, send_audio, send_event,
                agent._handle_turn, last_emitted,
            )

        self._run(drive())

        self.assertIn(("transcript", {"text": "hello"}), events)
        self.assertIn(
            ("agent_text", {"text": "Sure thing. Let me check."}), events
        )
        self.assertEqual(last_emitted[0], "Sure thing. Let me check.")

    def test_duplicate_assistant_text_not_emitted_twice(self) -> None:
        agent, _ = self._make_agent(["Same reply."])
        events, send_audio, send_event = self._make_recorders()
        last_emitted = [None]
        utterance = np.full(CHUNK_SIZE, ENERGY_RMS_THRESHOLD * 10, dtype=np.float32)

        async def drive():
            await _browser_handle_turn_with_events(
                agent, utterance, send_audio, send_event,
                agent._handle_turn, last_emitted,
            )
            await _browser_handle_turn_with_events(
                agent, utterance, send_audio, send_event,
                agent._handle_turn, last_emitted,
            )

        self._run(drive())

        agent_text_events = [e for e in events if e[0] == "agent_text"]
        self.assertEqual(len(agent_text_events), 1)
        self.assertEqual(agent_text_events[0], ("agent_text", {"text": "Same reply."}))

    def test_open_call_failure_emits_no_agent_text(self) -> None:
        # An LLM that raises before yielding any token: _open_call catches it
        # and leaves history empty, so no agent_text should be emitted.
        agent, _ = self._make_agent([])
        agent.llm = _RaisingLLM()
        events, send_audio, send_event = self._make_recorders()
        last_emitted = [None]

        async def drive():
            await _browser_open_call_with_events(
                agent, send_audio, send_event, agent._open_call, last_emitted
            )

        self._run(drive())

        self.assertEqual(events, [])
        self.assertIsNone(last_emitted[0])


if __name__ == "__main__":
    unittest.main()
