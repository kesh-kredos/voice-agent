"""Regression tests for models/text_to_speech_orpheus.py targeted fixes.

These tests exercise the pure-logic helpers without instantiating the heavy
TTSClient (which loads SNAC and opens an AsyncOpenAI connection).  We call
``_parse_token_ids`` as an unbound method (it never touches ``self``) and
build a ``stream()`` harness by stubbing ``_synthesise``.
"""

import asyncio
import unittest
from typing import AsyncGenerator, List

import numpy as np

from models.text_to_speech_orpheus import (
    TTSClient,
    _frame_to_codebooks,
    _SNAC_RANGES,
)


# ---------------------------------------------------------------------------
# _parse_token_ids
# ---------------------------------------------------------------------------

class ParseTokenIdsTests(unittest.TestCase):
    """_parse_token_ids must carry a partial <custom_token_...> across chunks."""

    # Call as unbound — the method never reads self.
    def _parse(self, text: str):
        return TTSClient._parse_token_ids(None, text)

    def test_complete_tokens_no_carry(self) -> None:
        ids, carry = self._parse("<custom_token_5><custom_token_6>")
        self.assertEqual(ids, [5, 6])
        self.assertEqual(carry, "")

    def test_partial_prefix_carry(self) -> None:
        ids, carry = self._parse("<custom_token_5><custom_to")
        self.assertEqual(ids, [5])
        self.assertEqual(carry, "<custom_to")

    def test_partial_open_angle_carry(self) -> None:
        ids, carry = self._parse("<custom_token_5><")
        self.assertEqual(ids, [5])
        self.assertEqual(carry, "<")

    def test_partial_with_digits_carry(self) -> None:
        ids, carry = self._parse("<custom_token_5><custom_token_12")
        self.assertEqual(ids, [5])
        self.assertEqual(carry, "<custom_token_12")

    def test_full_tag_prefix_carry(self) -> None:
        # Exactly "<custom_token_" with no digits yet — still a valid prefix.
        ids, carry = self._parse("<custom_token_5><custom_token_")
        self.assertEqual(ids, [5])
        self.assertEqual(carry, "<custom_token_")

    def test_no_tokens_no_carry(self) -> None:
        ids, carry = self._parse("just some text")
        self.assertEqual(ids, [])
        self.assertEqual(carry, "")

    def test_carry_reassembly_across_chunks(self) -> None:
        """Simulate the caller's reassembly pattern end-to-end."""
        chunk1 = "<custom_token_10><custom_tok"
        chunk2 = "en_11><custom_token_12>"

        ids1, carry1 = self._parse(chunk1)
        self.assertEqual(ids1, [10])

        # Caller prepends carry to next chunk
        ids2, carry2 = self._parse(carry1 + chunk2)
        self.assertEqual(ids2, [11, 12])
        self.assertEqual(carry2, "")


# ---------------------------------------------------------------------------
# stream() flush on punctuation + trailing whitespace
# ---------------------------------------------------------------------------

class StreamFlushTests(unittest.TestCase):
    """stream() must flush when punctuation is followed by trailing spaces."""

    def _run(self, coro):
        return asyncio.run(coro)

    async def _collect(self, token_stream, clauses: list):
        """Build a TTSClient without __init__, stub _synthesise, run stream()."""
        client = TTSClient.__new__(TTSClient)
        client.voice = "tara"

        async def fake_synthesise(
            text: str,
        ) -> AsyncGenerator[np.ndarray, None]:
            clauses.append(text)
            yield np.zeros(240, dtype=np.float32)

        client._synthesise = fake_synthesise

        chunks: list = []
        async for chunk in client.stream(token_stream):
            chunks.append(chunk)
        return chunks

    def test_hard_flush_with_trailing_space(self) -> None:
        """'Hello. ' (punctuation + space in same token) must flush, not stall."""
        # The fix: check stripped[-1] instead of buffer[-1] so that
        # "Hello. " flushes on the period even though the buffer ends with space.
        tokens = ["Hello", ". "]

        async def token_gen() -> AsyncGenerator[str, None]:
            for t in tokens:
                yield t

        clauses: List[str] = []
        self._run(self._collect(token_gen(), clauses))
        # The period triggers the flush; the trailing space is included in the flushed text.
        self.assertEqual(clauses, ["Hello. "])

    def test_soft_flush_with_trailing_space(self) -> None:
        """A soft-flush punctuation + space in the same token must flush."""
        # Build a clause long enough to pass MIN_SOFT_FLUSH_CHARS (30).
        text = "Well, this is a fairly long clause"
        # Comma + trailing space in one token — should trigger soft flush.
        tokens = [text, ", "]

        async def token_gen() -> AsyncGenerator[str, None]:
            for t in tokens:
                yield t

        clauses: List[str] = []
        self._run(self._collect(token_gen(), clauses))
        self.assertEqual(clauses, [text + ", "])

    def test_no_flush_without_punctuation(self) -> None:
        """Plain text with no punctuation should only flush at end-of-stream."""
        tokens = ["just", " plain", " words"]

        async def token_gen() -> AsyncGenerator[str, None]:
            for t in tokens:
                yield t

        clauses: List[str] = []
        self._run(self._collect(token_gen(), clauses))
        # Only the end-of-stream remainder flush.
        self.assertEqual(clauses, ["just plain words"])


# ---------------------------------------------------------------------------
# _frame_to_codebooks + _SNAC_RANGES (Orpheus +10 per-position offset)
# ---------------------------------------------------------------------------

class FrameToCodebooksTests(unittest.TestCase):
    """Regression: Orpheus tokens carry a +10 per-position offset that must be
    stripped before the codebook-base subtraction, and _SNAC_RANGES must accept
    the resulting shifted boundaries.  These would have caught the original bug
    where the offset was missing and ranges started at 0/4096/8192/..."""

    # Lowest valid token at each position → every codebook id must be 0.
    LOWER = [10, 4106, 8202, 12298, 16394, 20490, 24586]
    # Highest valid token at each position → every codebook id must be 4095.
    UPPER = [4105, 8201, 12297, 16393, 20489, 24585, 28681]

    def test_lower_boundary_maps_to_zero(self) -> None:
        l1, l2, l3 = _frame_to_codebooks(self.LOWER)
        self.assertEqual(l1, [0])
        self.assertEqual(l2, [0, 0])
        self.assertEqual(l3, [0, 0, 0, 0])

    def test_upper_boundary_maps_to_4095(self) -> None:
        l1, l2, l3 = _frame_to_codebooks(self.UPPER)
        self.assertEqual(l1, [4095])
        self.assertEqual(l2, [4095, 4095])
        self.assertEqual(l3, [4095, 4095, 4095, 4095])

    def test_ranges_accept_lower_boundary(self) -> None:
        self.assertTrue(
            all(lo <= tok <= hi for tok, (lo, hi) in zip(self.LOWER, _SNAC_RANGES))
        )

    def test_ranges_accept_upper_boundary(self) -> None:
        self.assertTrue(
            all(lo <= tok <= hi for tok, (lo, hi) in zip(self.UPPER, _SNAC_RANGES))
        )

    def test_ranges_reject_pre_fix_lower_boundary(self) -> None:
        # The pre-fix lower boundary (0, 4096, 8192, ...) is now below the valid
        # range and must be rejected — this is exactly what the old code got
        # wrong by treating these as valid audio frames.
        old_lower = [0, 4096, 8192, 12288, 16384, 20480, 24576]
        self.assertFalse(
            all(lo <= tok <= hi for tok, (lo, hi) in zip(old_lower, _SNAC_RANGES))
        )


if __name__ == "__main__":
    unittest.main()
