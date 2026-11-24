#!/usr/bin/env python3
"""Tests for kld_compare.py's streaming file comparison.

    python3 tools/test_kld_compare.py
"""

import os
import tempfile
import unittest

from kld_compare import same_bytes


def blob(n):
    """Deterministic n bytes."""
    return (bytes(range(256)) * (n // 256 + 1))[:n]


class SameBytesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def write(self, name, data):
        p = self.path(name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_equal_across_chunks(self):
        # More than one 1 MiB chunk, so the loop reads past the first one.
        data = blob(2 * (1 << 20) + 12345)
        a = self.write("a.bin", data)
        b = self.write("b.bin", data)
        self.assertTrue(same_bytes(a, b))

    def test_differing_byte_in_second_chunk(self):
        data = blob(2 * (1 << 20) + 12345)
        off = (1 << 20) + 7
        changed = bytearray(data)
        changed[off] ^= 0xFF
        a = self.write("a.bin", data)
        b = self.write("b.bin", bytes(changed))
        self.assertFalse(same_bytes(a, b))

    def test_different_lengths(self):
        data = blob(2 * (1 << 20) + 12345)
        a = self.write("a.bin", data)
        # One extra trailing byte: equal up to the shorter file's end.
        b = self.write("b.bin", data + b"x")
        self.assertFalse(same_bytes(a, b))
        self.assertFalse(same_bytes(b, a))
        # A strict prefix, so the last read of the shorter file is non-empty but short.
        c = self.write("c.bin", data[:-1])
        self.assertFalse(same_bytes(a, c))
        self.assertFalse(same_bytes(c, a))


if __name__ == "__main__":
    unittest.main()
