"""ATen-compatible MT19937 + normal_fill_16, in numpy.

The nano decoder is noise-fed, so a rendering is only reproducible if the
noise stream is. PyTorch's is MT19937 seeded from the low 32 bits, a 24-bit
uniform, and Box-Muller in blocks of 16 -- reproduced here so that Python,
the C runtime and PyTorch all render a given seed the same way.

`numpy.random.RandomState` is also MT19937, but its seeding, its tempering
draw order and its Gaussian transform all differ from ATen's, so it cannot be
substituted. The arithmetic below is the ATen path exactly.

Exactness, measured rather than assumed: the integer half is bit-exact and
`mcu/test/fixtures/en_us_e12nano/e2e_uniform.bin` proves it. The Gaussian half
goes through log/cos/sin, which are not bit-identical across libm builds, so a
minority of draws differ by a few ulp -- the C runtime documents the same
limit and the fixture bounds it at 1e-5.
"""

from __future__ import annotations

import numpy as np

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF


_SEEDED: dict[int, np.ndarray] = {}


def _seeded_state(seed: int) -> np.ndarray:
    """The 624-word MT19937 state for `seed`, before the first twist.

    Seeding is a 623-step serial recurrence in Python ints (~0.4 ms), and the
    nano voices use one fixed seed, so each seed's state is built once and
    copied. Callers must copy it; it is never handed out to be mutated.
    """
    state = _SEEDED.get(seed)
    if state is None:
        words = [seed]
        for i in range(1, _N):
            prev = words[-1]
            words.append((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        state = np.array(words, dtype=np.uint32)
        state.flags.writeable = False
        if len(_SEEDED) >= 64:
            _SEEDED.clear()
        _SEEDED[seed] = state
    return state


class ATenMT19937:
    """MT19937 with ATen's seeding and 24-bit float uniform.

    The twist and the tempering run on whole numpy arrays rather than one
    Python int at a time. MT19937 is integer arithmetic, so this is the same
    stream bit for bit, not an approximation: `uniform` must equal what the
    scalar loop produced (and the fixture's torch.rand stream) exactly. It used
    to be the slowest thing in front of the nano decoder -- ~4 ms per
    utterance -- and every millisecond here is time before the first audio.
    """

    __slots__ = ("_state", "_next")

    def __init__(self, seed: int) -> None:
        self._state = _seeded_state(seed & 0xFFFFFFFF).copy()
        # The state is twisted before the first draw, as ATen does.
        self._next = _N

    def _next_state(self) -> None:
        """One MT19937 twist, in place.

        Element i reads s[i], s[i+1] and s[(i+397) % 624], where every index
        below i already holds its NEW value. The slices are cut so that each
        one only reads values its predecessors have finished writing: [0, 227)
        reads s[397:624] (all old), [227, 454) reads s[0:227] (new), [454, 623)
        reads s[227:396] (new), and 623 reads the new s[0] and s[396].
        """
        s = self._state
        upper = np.uint32(_UPPER)
        lower = np.uint32(_LOWER)
        matrix = np.uint32(_MATRIX_A)
        one = np.uint32(1)
        # (write range, the s[(i + 397) % 624] slice it reads)
        for (lo, hi), (rlo, rhi) in (((0, 227), (397, 624)),
                                     ((227, 454), (0, 227)),
                                     ((454, 623), (227, 396))):
            v = s[lo + 1:hi + 1]
            mixed = (s[lo:hi] & upper) | (v & lower)
            twist = (mixed >> one) ^ np.where((v & one) != 0, matrix, np.uint32(0))
            s[lo:hi] = s[rlo:rhi] ^ twist
        mixed = (int(s[_N - 1]) & _UPPER) | (int(s[0]) & _LOWER)
        twist = (mixed >> 1) ^ (_MATRIX_A if (int(s[0]) & 1) else 0)
        s[_N - 1] = np.uint32(int(s[_M - 1]) ^ twist)
        self._next = 0

    def next_uint32_array(self, n: int) -> np.ndarray:
        """The next n tempered outputs, as uint32."""
        if n < 0:
            raise ValueError(f"cannot draw {n} values")
        parts = []
        remaining = n
        while remaining > 0:
            if self._next >= _N:
                self._next_state()
            take = min(remaining, _N - self._next)
            y = self._state[self._next:self._next + take].copy()
            self._next += take
            remaining -= take
            y ^= y >> np.uint32(11)
            y ^= (y << np.uint32(7)) & np.uint32(0x9D2C5680)
            y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
            y ^= y >> np.uint32(18)
            parts.append(y)
        if not parts:
            return np.empty(0, dtype=np.uint32)
        return np.concatenate(parts)

    def next_uint32(self) -> int:
        return int(self.next_uint32_array(1)[0])

    def uniform(self, n: int) -> np.ndarray:
        """at::uniform_real_distribution<float>: (raw & (2**24 - 1)) * 2**-24.

        A 24-bit integer times 2**-24 is exactly representable in float32, so
        the cast loses nothing and the result matches the scalar form exactly.
        """
        raw = self.next_uint32_array(n) & np.uint32(0xFFFFFF)
        return (raw.astype(np.float64) * (1.0 / 16777216.0)).astype(np.float32)


def _normal_fill_16(block: np.ndarray) -> None:
    """ATen normal_fill_16, mean 0 std 1, float32 throughout. In place."""
    u1 = np.float32(1.0) - block[:8]
    u2 = block[8:]
    radius = np.sqrt(np.float32(-2.0) * np.log(u1, dtype=np.float32), dtype=np.float32)
    theta = np.float32(2.0 * np.pi) * u2
    block[:8] = radius * np.cos(theta, dtype=np.float32)
    block[8:] = radius * np.sin(theta, dtype=np.float32)


def _normal_fill_blocks(flat: np.ndarray) -> None:
    """`_normal_fill_16` over every whole block of 16, in one pass.

    The same float32 ufuncs on the same values: the first eight of each block
    feed the radius and the last eight the angle. The halves are copied into
    contiguous arrays first so log/cos/sin see the same contiguous float32
    input the per-block form gave them.
    """
    blocks = flat.reshape(-1, 16)
    u1 = np.ascontiguousarray(np.float32(1.0) - blocks[:, :8])
    u2 = np.ascontiguousarray(blocks[:, 8:])
    radius = np.sqrt(np.float32(-2.0) * np.log(u1, dtype=np.float32), dtype=np.float32)
    theta = np.float32(2.0 * np.pi) * u2
    blocks[:, :8] = radius * np.cos(theta, dtype=np.float32)
    blocks[:, 8:] = radius * np.sin(theta, dtype=np.float32)


def uniform_stream(seed: int, n: int) -> np.ndarray:
    return ATenMT19937(seed).uniform(n)


class _NoisePrefix:
    """One seed's uniform stream and its block-wise Gaussian transform, both
    as long as any request so far has needed.

    Every call of `seeded_noise` with the same seed restarts the same stream,
    so the draws for a short request are a prefix of those for a long one.
    Box-Muller works on independent blocks of 16 aligned to the start of the
    stream, so the transformed blocks are a prefix too. Only the tail block
    that torch redraws when the size is not a multiple of 16 depends on the
    size, and that one is computed per call. The nano voices use one seed for
    every utterance, so this replaces a twist-and-transform over the whole
    utterance with a slice.
    """

    __slots__ = ("uniform", "normal")

    def __init__(self, seed: int, n: int) -> None:
        # Whole blocks only, so that every block in `normal` is transformed.
        n = -(-n // 16) * 16
        self.uniform = ATenMT19937(seed).uniform(n)
        self.uniform.flags.writeable = False
        normal = self.uniform.copy()
        _normal_fill_blocks(normal)
        normal.flags.writeable = False
        self.normal = normal


_PREFIXES: dict[int, _NoisePrefix] = {}


def _noise_prefix(seed: int, n: int) -> _NoisePrefix:
    """A prefix of at least n draws for `seed`, grown geometrically."""
    seed &= 0xFFFFFFFF
    prefix = _PREFIXES.get(seed)
    if prefix is None or prefix.uniform.size < n:
        have = 0 if prefix is None else prefix.uniform.size
        prefix = _NoisePrefix(seed, max(n, 2 * have, 4096))
        if len(_PREFIXES) >= 8 and seed not in _PREFIXES:
            _PREFIXES.clear()
        _PREFIXES[seed] = prefix
    return prefix


def warm_noise(seed: int, channels: int, frames: int) -> None:
    """Build `seed`'s prefix for requests up to channels*frames now, so that
    cost is not paid inside the first request that needs it."""
    _noise_prefix(seed, channels * frames + 16)


def seeded_noise(seed: int, channels: int, frames: int) -> np.ndarray:
    """[channels, frames] of N(0,1), matching torch.randn under this seed.

    Torch sends sizes under 16 to a scalar path this does not implement; the
    decoder always asks for channels*frames well above that.

    The values are those of drawing `size` uniforms, transforming every whole
    block of 16, and (when size % 16) overwriting the last 16 with a fresh
    block drawn after them; they come from the cached prefix instead of a
    fresh generator. tests/test_nano_stream.py checks both against each other.
    """
    size = channels * frames
    if size < 16:
        raise ValueError(f"seeded_noise needs at least 16 values, got {size}")
    prefix = _noise_prefix(seed, size + 16)
    flat = prefix.normal[:size].copy()
    if size % 16:
        # Torch draws a FRESH block of 16 (continuing the same stream) and
        # overwrites the last 16 values with it. The remainder left by the
        # loop above is discarded, so the tail is not simply its leftover.
        tail = prefix.uniform[size:size + 16].copy()
        _normal_fill_16(tail)
        flat[size - 16 :] = tail
    return flat.reshape(channels, frames)


def seeded_noise_uncached(seed: int, channels: int, frames: int) -> np.ndarray:
    """`seeded_noise` from a fresh generator: the reference it is tested against."""
    size = channels * frames
    if size < 16:
        raise ValueError(f"seeded_noise needs at least 16 values, got {size}")
    gen = ATenMT19937(seed)
    flat = gen.uniform(size)
    whole = (size // 16) * 16
    _normal_fill_blocks(flat[:whole])
    if size % 16:
        tail = gen.uniform(16)
        _normal_fill_16(tail)
        flat[size - 16 :] = tail
    return flat.reshape(channels, frames)
