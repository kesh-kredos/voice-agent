"""Regression tests for utils/audio.py browser encoding helper."""

import struct
import unittest

import numpy as np

from utils.audio import float32_to_pcm16le


class Float32ToPcm16leTests(unittest.TestCase):
    """float32_to_pcm16le must emit little-endian signed 16-bit PCM."""

    def test_known_values_little_endian(self) -> None:
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        raw = float32_to_pcm16le(audio)

        # 5 samples * 2 bytes each
        self.assertEqual(len(raw), 10)

        samples = struct.unpack("<5h", raw)
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[1], int(0.5 * 32767))
        self.assertEqual(samples[2], int(-0.5 * 32767))
        self.assertEqual(samples[3], 32767)
        self.assertEqual(samples[4], -32767)

    def test_clipping_above_one(self) -> None:
        audio = np.array([1.5, 100.0], dtype=np.float32)
        raw = float32_to_pcm16le(audio)
        samples = struct.unpack("<2h", raw)
        self.assertEqual(samples[0], 32767)
        self.assertEqual(samples[1], 32767)

    def test_clipping_below_negative_one(self) -> None:
        audio = np.array([-1.5, -100.0], dtype=np.float32)
        raw = float32_to_pcm16le(audio)
        samples = struct.unpack("<2h", raw)
        self.assertEqual(samples[0], -32767)
        self.assertEqual(samples[1], -32767)

    def test_empty_input(self) -> None:
        audio = np.array([], dtype=np.float32)
        self.assertEqual(float32_to_pcm16le(audio), b"")


if __name__ == "__main__":
    unittest.main()
