"""Numpy-only grapheme -> phoneme model for words the dictionary does not have.

The checkpoint is PeterReid/graphemes_to_phonemes_en_us (Apache-2.0), a
751,551-parameter BART with one encoder layer, one decoder layer, one attention
head and d_model 128, trained on misaki's own us_gold/us_silver dictionaries.
It was not distilled from espeak-ng, which is the whole point: it is the OOV
path for a front end that must not carry a GPL dependency.

`tools/build_nano_g2p_assets.py` converts the published safetensors to
`g2p_data/oov_bart_en_us.npz`; this module is the forward pass, transcribed
from transformers' `BartModel` so it can run under numpy alone:

  * post-norm blocks (residual add, then LayerNorm), which is BART, not the
    pre-norm arrangement most later decoders use;
  * learned positional embeddings with the offset of 2 that BART reserves for
    pad and bos, hence the 66-row table for 64 positions;
  * `scale_embedding=false`, so token embeddings are used unscaled;
  * exact erf gelu, not the tanh approximation;
  * greedy decoding from `decoder_start_token_id`, which is what the upstream
    reference script gets from its generation config (no beams, no sampling).

Decoding is greedy, per word, with the decoder's keys and values cached
(`_greedy`); `prefetch` decodes a text's unknown words as one batch, and
common words are not decoded at all but read from a table of this model's
own outputs (`_WordTable`, built by tools/build_oov_table.py).
"""

from __future__ import annotations

import bisect
import hashlib
import io
import json
import lzma
import math
from array import array
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "g2p_data"
MODEL_PATH = DATA_DIR / "oov_bart_en_us.npz"
# This model's own output for common words, precomputed by
# tools/build_oov_table.py (see _WordTable).
TABLE_PATH = DATA_DIR / "oov_table_en_us.npz.xz"

# The tokenizer reserves 0..3 for <pad>/<s>/</s>/<unk> and then indexes the
# character lists; the shipped config records both lists already padded with
# four underscores so the character at index i is simply chars[i].
SPECIAL_TOKEN_COUNT = 4
UNK_ID = 3
# Words kept by the memo in OOVPhonemizer.__call__ (oldest evicted first).
MEMO_SIZE = 4096


class OOVModelError(RuntimeError):
    """Raised when the fallback model is unusable; never swallowed by callers."""


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """x.mean / x.var over the last axis, done as numpy's own _mean/_var do
    them (add.reduce, then true_divide by an intp count with unsafe casting)
    but without their per-call wrapper overhead. Bit-identical to
    `(x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)
    * weight + bias`; tests/test_nano_g2p_oov_fast.py checks this."""
    # In-place division by the Python int count is the same IEEE division of
    # the same float32 values as numpy's true_divide by an intp count (the
    # count is exact in float32), without the keyword/casting dispatch.
    count = x.shape[-1]
    mean = np.add.reduce(x, axis=-1, keepdims=True)
    mean /= count
    dev = x - mean
    sq = np.multiply(dev, dev)
    var = np.add.reduce(sq, axis=-1, keepdims=True)
    var /= count
    var += eps
    np.sqrt(var, out=var)
    out = dev / var
    out *= weight
    out += bias
    return out


def _layer_norm_reference(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                          eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 erf, max abs error 1.5e-7.

    transformers maps the "gelu" activation to the erf form, not the tanh
    approximation. numpy has no erf and scipy is not a dependency of this
    package, so the approximation is inlined. 1.5e-7 is far below the margin between competing
    characters in a 63-way argmax; the k2 gate test compares this port's greedy
    output against transformers on the dictionary and finds no disagreement.
    """
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-ax * ax))


def _gelu_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


_SQRT2 = math.sqrt(2.0)


def _gelu_fast(x: np.ndarray) -> np.ndarray:
    """`_gelu_exact` with the same operations in the same order, in place
    where possible (IEEE + and * are commutative, so only the temporaries
    differ). Bit-identical; tests/test_nano_g2p_oov_fast.py checks this."""
    z = x / _SQRT2
    sign = np.sign(z)
    ax = np.abs(z)
    t = np.multiply(ax, 0.3275911)
    t += 1.0
    np.divide(1.0, t, out=t)
    p = np.multiply(t, 1.061405429)
    p += -1.453152027
    p *= t
    p += 1.421413741
    p *= t
    p += -0.284496736
    p *= t
    p += 0.254829592
    p *= t
    e = np.negative(ax)
    e *= ax
    np.exp(e, out=e)
    p *= e
    np.subtract(1.0, p, out=p)
    p *= sign
    p += 1.0
    h = np.multiply(x, 0.5)
    h *= p
    return h


def _softmax(x: np.ndarray) -> np.ndarray:
    """softmax over the last axis. maximum.reduce and add.reduce are what
    x.max() and x.sum() call, minus their Python wrappers; bit-identical to
    `_softmax_reference` (tests/test_nano_g2p_oov_fast.py)."""
    shifted = x - np.maximum.reduce(x, axis=-1, keepdims=True)
    np.exp(shifted, out=shifted)
    shifted /= np.add.reduce(shifted, axis=-1, keepdims=True)
    return shifted


def _softmax_reference(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class _Attention:
    """Single-head BartAttention. num_heads is 1 in this checkpoint, asserted
    by the asset builder, so the head reshape collapses to a plain matmul."""

    def __init__(self, weights: dict[str, np.ndarray], prefix: str, scaling: float) -> None:
        self.q_w = np.ascontiguousarray(weights[f"{prefix}.q_proj.weight"].T)
        self.q_b = weights[f"{prefix}.q_proj.bias"]
        self.k_w = np.ascontiguousarray(weights[f"{prefix}.k_proj.weight"].T)
        self.k_b = weights[f"{prefix}.k_proj.bias"]
        self.v_w = np.ascontiguousarray(weights[f"{prefix}.v_proj.weight"].T)
        self.v_b = weights[f"{prefix}.v_proj.bias"]
        self.o_w = np.ascontiguousarray(weights[f"{prefix}.out_proj.weight"].T)
        self.o_b = weights[f"{prefix}.out_proj.bias"]
        self.scaling = scaling

    def __call__(self, query: np.ndarray, memory: np.ndarray,
                 causal: bool = False) -> np.ndarray:
        q = (query @ self.q_w + self.q_b) * self.scaling
        k = memory @ self.k_w + self.k_b
        v = memory @ self.v_w + self.v_b
        scores = q @ k.T
        if causal:
            mask = np.triu(np.ones((scores.shape[0], scores.shape[1]), dtype=bool), k=1)
            scores = np.where(mask, np.float32(-np.inf), scores)
        return _softmax(scores) @ v @ self.o_w + self.o_b


class _Layer:
    """One BartEncoderLayer, or one BartDecoderLayer when `cross` is present."""

    def __init__(self, weights: dict[str, np.ndarray], prefix: str,
                 scaling: float, eps: float, cross: bool) -> None:
        self.eps = eps
        self.self_attn = _Attention(weights, f"{prefix}.self_attn", scaling)
        self.self_ln = (weights[f"{prefix}.self_attn_layer_norm.weight"],
                        weights[f"{prefix}.self_attn_layer_norm.bias"])
        if cross:
            self.cross_attn = _Attention(weights, f"{prefix}.encoder_attn", scaling)
            self.cross_ln = (weights[f"{prefix}.encoder_attn_layer_norm.weight"],
                             weights[f"{prefix}.encoder_attn_layer_norm.bias"])
        else:
            self.cross_attn = None
            self.cross_ln = None
        self.fc1_w = np.ascontiguousarray(weights[f"{prefix}.fc1.weight"].T)
        self.fc1_b = weights[f"{prefix}.fc1.bias"]
        self.fc2_w = np.ascontiguousarray(weights[f"{prefix}.fc2.weight"].T)
        self.fc2_b = weights[f"{prefix}.fc2.bias"]
        self.final_ln = (weights[f"{prefix}.final_layer_norm.weight"],
                         weights[f"{prefix}.final_layer_norm.bias"])

    def __call__(self, hidden: np.ndarray, memory: np.ndarray | None,
                 causal: bool) -> np.ndarray:
        hidden = _layer_norm(hidden + self.self_attn(hidden, hidden, causal=causal),
                             *self.self_ln, self.eps)
        if self.cross_attn is not None:
            if memory is None:
                raise OOVModelError("decoder layer reached with no encoder memory")
            hidden = _layer_norm(hidden + self.cross_attn(hidden, memory),
                                 *self.cross_ln, self.eps)
        ffn = _gelu_fast(hidden @ self.fc1_w + self.fc1_b) @ self.fc2_w + self.fc2_b
        return _layer_norm(hidden + ffn, *self.final_ln, self.eps)


class _SortedKeys:
    """The table's words as a sequence of bytes, for bisect."""

    def __init__(self, blob: bytes, off: np.ndarray) -> None:
        # array("q") indexes to plain ints at 8 bytes a row; a list would
        # hold ~36 bytes a row of int objects.
        self.blob = blob
        self.off = array("q")
        self.off.frombytes(np.ascontiguousarray(off, dtype=np.int64).tobytes())

    def __len__(self) -> int:
        return len(self.off) - 1

    def __getitem__(self, i: int) -> bytes:
        return self.blob[self.off[i]:self.off[i + 1]]


class _WordTable:
    """A read-only word -> phonemes table, the model's own greedy output.

    tools/build_oov_table.py runs OOVPhonemizer.__call__ over a fixed general
    word list and packs the results here; a hit returns what the model would
    return for that word on the machine the table was built on, without
    decoding it. Words are stored sorted (UTF-8 byte order) in one blob, the
    phonemes in the same order in another, with one length byte each, and a
    lookup is a binary search. The table records the sha256 of the model file
    it was built from and is refused for any other model.
    """

    def __init__(self, path: Path, model_sha256: str) -> None:
        try:
            # an uncompressed .npz inside xz: about half the size of
            # np.savez_compressed for this data, and lzma is in the stdlib
            raw = lzma.decompress(path.read_bytes())
            with np.load(io.BytesIO(raw), allow_pickle=False) as bundle:
                key_len = np.asarray(bundle["key_len"], dtype=np.int64)
                val_len = np.asarray(bundle["val_len"], dtype=np.int64)
                keys = bytes(bundle["keys"])
                values = bytes(bundle["values"])
                meta = json.loads(str(bundle["meta_json"]))
        except (OSError, ValueError, KeyError, lzma.LZMAError) as exc:
            raise OOVModelError(f"could not read {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OOVModelError(f"{path}: meta_json is not valid JSON: {exc}") from exc
        if meta.get("format") != 1:
            raise OOVModelError(f"{path}: unsupported table format {meta.get('format')!r}")
        if meta.get("model_sha256") != model_sha256:
            raise OOVModelError(
                f"{path} was built for model {meta.get('model_sha256')}, not {model_sha256}")
        if key_len.size != val_len.size or int(key_len.sum()) != len(keys) \
                or int(val_len.sum()) != len(values) or (key_len.size and int(key_len.min()) < 1):
            raise OOVModelError(f"{path}: inconsistent table arrays")
        self.keys = _SortedKeys(keys, np.concatenate([[0], np.cumsum(key_len)]))
        self.values = _SortedKeys(values, np.concatenate([[0], np.cumsum(val_len)]))
        self.size = len(self.keys)
        # Sortedness is not re-checked here (it would cost about as much as
        # the load): pack_table sorts, tests/test_oov_table.py checks the
        # shipped file, and get() compares the word it lands on, so a
        # mis-sorted table could only miss, never answer wrongly.
        self.meta = meta

    def get(self, word: str) -> str | None:
        raw = word.encode("utf-8")
        i = bisect.bisect_left(self.keys, raw)
        if i < self.size and self.keys[i] == raw:
            return self.values[i].decode("utf-8")
        return None

    def items(self, idx) -> list[tuple[str, str]]:
        return [(self.keys[i].decode("utf-8"), self.values[i].decode("utf-8")) for i in idx]


def write_table(path: Path | str, entries: dict[str, str], model_sha256: str, meta: dict) -> None:
    """Write the file _WordTable reads (build tool and tests only)."""
    buf = io.BytesIO()
    np.savez(buf, **pack_table(entries, model_sha256, meta))
    Path(path).write_bytes(lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME))


def pack_table(entries: dict[str, str], model_sha256: str, meta: dict) -> dict[str, np.ndarray]:
    """The arrays _WordTable reads."""
    rows = sorted((k.encode("utf-8"), v.encode("utf-8")) for k, v in entries.items())
    if any(not k or len(k) > 255 or len(v) > 255 for k, v in rows):
        raise OOVModelError("table rows need 1..255-byte words and at most 255-byte phonemes")
    full_meta = {"format": 1, "model_sha256": model_sha256, **meta}
    return {
        "key_len": np.array([len(k) for k, _ in rows], dtype=np.uint8),
        "val_len": np.array([len(v) for _, v in rows], dtype=np.uint8),
        "keys": np.frombuffer(b"".join(k for k, _ in rows), dtype=np.uint8),
        "values": np.frombuffer(b"".join(v for _, v in rows), dtype=np.uint8),
        "meta_json": np.array(json.dumps(full_meta, sort_keys=True)),
    }


class OOVPhonemizer:
    """Greedy grapheme -> phoneme decoding for one word at a time."""

    def __init__(self, path: Path | str = MODEL_PATH,
                 table_path: Path | str | None = TABLE_PATH) -> None:
        """`table_path=None` decodes every word (the build tool uses this)."""
        path = Path(path)
        if not path.is_file():
            raise OOVModelError(
                f"fallback model not found at {path}; run "
                "tools/build_nano_g2p_assets.py to fetch it"
            )
        try:
            with np.load(path, allow_pickle=False) as bundle:
                weights = {name: np.asarray(bundle[name], dtype=np.float32)
                           for name in bundle.files if name != "meta_json"}
                meta_raw = str(bundle["meta_json"])
        except (OSError, ValueError, KeyError) as exc:
            raise OOVModelError(f"could not read {path}: {exc}") from exc
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError as exc:
            raise OOVModelError(f"{path}: meta_json is not valid JSON: {exc}") from exc

        self.grapheme_chars: str = meta["grapheme_chars"]
        self.phoneme_chars: str = meta["phoneme_chars"]
        self.max_positions = int(meta["max_position_embeddings"])
        self.bos_id = int(meta["bos_token_id"])
        self.eos_id = int(meta["eos_token_id"])
        self.start_id = int(meta["decoder_start_token_id"])
        self.offset = int(meta["position_offset"])
        eps = float(meta["layer_norm_eps"])
        heads = int(meta["num_heads"])
        d_model = int(meta["d_model"])
        if heads != 1:
            raise OOVModelError(f"this port assumes one attention head, checkpoint has {heads}")
        scaling = (d_model // heads) ** -0.5

        self.grapheme_to_id = {
            ch: i for i, ch in enumerate(self.grapheme_chars) if i >= SPECIAL_TOKEN_COUNT
        }
        self.embed = weights["model.shared.weight"]
        self.embed_t = np.ascontiguousarray(self.embed.T)
        self.logit_bias = weights["final_logits_bias"][0]
        self.enc_pos = weights["model.encoder.embed_positions.weight"]
        self.dec_pos = weights["model.decoder.embed_positions.weight"]
        self.enc_ln = (weights["model.encoder.layernorm_embedding.weight"],
                       weights["model.encoder.layernorm_embedding.bias"])
        self.dec_ln = (weights["model.decoder.layernorm_embedding.weight"],
                       weights["model.decoder.layernorm_embedding.bias"])
        self.encoder = _Layer(weights, "model.encoder.layers.0", scaling, eps, cross=False)
        self.decoder = _Layer(weights, "model.decoder.layers.0", scaling, eps, cross=True)
        self.eps = eps
        self._memo: dict[str, str] = {}
        self._dec_inputs: tuple[np.ndarray, ...] | None = None
        self._fast: tuple | None = None
        self._enc_fast: tuple | None = None
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self._table_path = None if table_path is None else Path(table_path)
        self._table: _WordTable | None = None

    def table(self) -> _WordTable | None:
        """The precomputed table, loaded on first use; None if disabled."""
        if self._table is None and self._table_path is not None:
            if not self._table_path.is_file():
                raise OOVModelError(f"word table not found at {self._table_path}")
            self._table = _WordTable(self._table_path, self.model_sha256)
        return self._table

    def _known(self, word: str) -> str | None:
        """The memoised or tabled answer for `word`, without decoding."""
        cached = self._memo.get(word)
        if cached is not None:
            return cached
        table = self.table()
        return None if table is None else table.get(word)

    def encode_word(self, word: str) -> list[int]:
        """Character ids with bos/eos, unknown characters mapped to <unk>."""
        return [self.bos_id, *(self.grapheme_to_id.get(c, UNK_ID) for c in word), self.eos_id]

    def __call__(self, word: str) -> str:
        """Phonemes for `word`, or '' if the model emits nothing usable.

        The output is a pure function of `word`, so results are memoised
        (bounded); a text that is phonemized twice, as the long-text fallback
        does, pays for each unknown word once.
        """
        if not word:
            raise OOVModelError("cannot phonemize an empty word")
        cached = self._known(word)
        if cached is not None:
            return cached
        ids = self.encode_word(word)
        if len(ids) > self.max_positions:
            raise OOVModelError(
                f"word {word!r} needs {len(ids)} positions; the model has {self.max_positions}"
            )
        memory = self._encode_fast(ids)
        out = self._greedy_fast(memory)
        result = "".join(
            self.phoneme_chars[i] for i in out[1:]
            if SPECIAL_TOKEN_COUNT <= i < len(self.phoneme_chars)
        )
        if len(self._memo) >= MEMO_SIZE:
            self._memo.pop(next(iter(self._memo)))
        self._memo[word] = result
        return result

    def _greedy(self, memory: np.ndarray) -> list[int]:
        """Greedy decoding with the decoder's keys and values cached.

        The decoder is one layer, so row i of its self-attention keys and
        values depends only on token i; and the logits only need the last
        row. Each step therefore looks up one token's row, appends its
        key/value, and runs the rest of the layer on that row alone. This is
        the function `_decode_step` computes for the last row of the full
        prefix, with the projections regrouped (see `_cross_terms` and
        `_decoder_inputs`), so the float rounding differs slightly; what is
        guaranteed is the greedy output, which tests/test_nano_g2p_oov_fast.py
        checks word for word against `greedy_reference`.
        """
        dec = self.decoder
        sa, ca = dec.self_attn, dec.cross_attn
        if ca is None:
            raise OOVModelError("decoder layer has no cross-attention")
        cross_score_w, cross_score_b, cross_out = self._cross_terms(memory)
        rows, qs, ks, vos = self._decoder_inputs()
        vocab = self.embed.shape[0]
        steps = self.max_positions - 1
        keys = np.empty((steps, ks.shape[1]), dtype=np.float32)
        vals = np.empty((steps, vos.shape[1]), dtype=np.float32)
        out: list[int] = [self.start_id]
        for step in range(steps):
            at = step * vocab + out[-1]
            keys[step] = ks[at]
            vals[step] = vos[at]
            att = _softmax(qs[at : at + 1] @ keys[: step + 1].T) @ vals[: step + 1] + sa.o_b
            h = _layer_norm(rows[at : at + 1] + att, *dec.self_ln, dec.eps)
            att = _softmax(h @ cross_score_w + cross_score_b) @ cross_out + ca.o_b
            h = _layer_norm(h + att, *dec.cross_ln, dec.eps)
            ffn = _gelu_fast(h @ dec.fc1_w + dec.fc1_b) @ dec.fc2_w + dec.fc2_b
            h = _layer_norm(h + ffn, *dec.final_ln, dec.eps)
            nxt = int(np.argmax(h[0] @ self.embed_t + self.logit_bias))
            if nxt == self.eos_id:
                break
            out.append(nxt)
        return out

    def _enc_tables(self) -> tuple:
        """For every (position, character): the encoder's normed input row and
        its query, key and value (the value through the output projection),
        flattened to [position * vocab + id] -- the encoder-side twin of
        _decoder_inputs. Built once."""
        if self._enc_fast is None:
            a = self.encoder.self_attn
            vocab = self.embed.shape[0]
            pos = self.enc_pos[self.offset : self.offset + self.max_positions]
            rows = (pos[:, None, :] + self.embed[None, :, :]).reshape(-1, self.embed.shape[1])
            rows = np.ascontiguousarray(_layer_norm(rows, *self.enc_ln, self.eps), dtype=np.float32)
            qs = np.ascontiguousarray((rows @ a.q_w + a.q_b) * a.scaling, dtype=np.float32)
            ks = np.ascontiguousarray(rows @ a.k_w + a.k_b, dtype=np.float32)
            vos = np.ascontiguousarray((rows @ a.v_w + a.v_b) @ a.o_w + a.o_b, dtype=np.float32)
            self._enc_fast = (rows, qs, ks, vos, vocab)
        return self._enc_fast

    def _encode_fast(self, ids: list[int]) -> np.ndarray:
        """_encode on the tables of _enc_tables and the table GELU; agrees with
        it to float32 rounding (the output bias rides inside the softmax-
        weighted sum, whose weights sum to one)."""
        from .nano import gelu_fast  # noqa: PLC0415

        rows, qs, ks, vos, vocab = self._enc_tables()
        enc = self.encoder
        at = np.arange(len(ids)) * vocab + np.asarray(ids)
        h = rows[at]
        sc = qs[at] @ ks[at].T
        sc -= np.maximum.reduce(sc, axis=-1, keepdims=True)
        np.exp(sc, out=sc)
        sc /= np.add.reduce(sc, axis=-1, keepdims=True)
        x = sc @ vos[at]
        x += h
        x = _layer_norm(x, *enc.self_ln, enc.eps)
        f = x @ enc.fc1_w
        f += enc.fc1_b
        y = gelu_fast(f) @ enc.fc2_w
        y += enc.fc2_b
        y += x
        return _layer_norm(y, *enc.final_ln, enc.eps)

    def _fast_tables(self) -> tuple:
        """Word-independent constants of _greedy_fast, built once.

        - rows + the self-attention output bias, so a step adds one array;
        - the final norm's affine folded into the logits:
          (n * g + b) E + c = n (diag(g) E) + (b E + c);
        - step 0, which attends to the start token alone (a softmax over one
          score is exactly 1), so its self-attention block is a constant row.
        """
        if self._fast is None:
            dec = self.decoder
            sa = dec.self_attn
            rows, qs, ks, vos = self._decoder_inputs()
            rows_ob = np.ascontiguousarray(rows + sa.o_b, dtype=np.float32)
            g, b = dec.final_ln
            f64 = np.float64
            logit_w = np.ascontiguousarray((g.astype(f64)[:, None] * self.embed_t).astype(np.float32))
            logit_b = (b.astype(f64) @ self.embed_t + self.logit_bias).astype(np.float32)
            at0 = self.start_id
            h0 = _layer_norm(rows_ob[at0] + vos[at0], *dec.self_ln, dec.eps).astype(np.float32)
            self._fast = (rows_ob, logit_w, logit_b, h0)
        return self._fast

    def _greedy_fast(self, memory: np.ndarray) -> list[int]:
        """_greedy with fewer numpy calls per step (on one word every call
        costs more than its arithmetic): the constants of _fast_tables, the
        cross-attention output bias moved inside the softmax-weighted sum
        (the weights sum to one), 1-D norms with a Python-float scale, and
        the decoder's table GELU. The float rounding differs from _greedy;
        the greedy output is what is checked, against the shipped word table
        (this model's own answers) in tests/test_nano_g2p_oov_fast.py."""
        from .nano import gelu_fast  # noqa: PLC0415

        dec = self.decoder
        sa, ca = dec.self_attn, dec.cross_attn
        if ca is None:
            raise OOVModelError("decoder layer has no cross-attention")
        score_w, score_b, cross_out = self._cross_terms(memory)
        score_b = score_b.reshape(-1)
        cross_out = cross_out + ca.o_b
        rows, qs, ks, vos = self._decoder_inputs()
        rows_ob, logit_w, logit_b, h0 = self._fast_tables()
        (sg, sb), (cg, cb) = dec.self_ln, dec.cross_ln
        eps = dec.eps
        d = rows.shape[1]
        inv_d = 1.0 / d
        vocab = self.embed.shape[0]
        steps = self.max_positions - 1
        keys = np.empty((steps, ks.shape[1]), dtype=np.float32)
        vals = np.empty((steps, vos.shape[1]), dtype=np.float32)
        fc1_w, fc1_b, fc2_w, fc2_b = dec.fc1_w, dec.fc1_b, dec.fc2_w, dec.fc2_b
        add_reduce, max_reduce, exp = np.add.reduce, np.maximum.reduce, np.exp
        sqrt = math.sqrt
        eos = self.eos_id
        out: list[int] = [self.start_id]
        last = self.start_id
        for step in range(steps):
            at = step * vocab + last
            keys[step] = ks[at]
            vals[step] = vos[at]
            if step == 0:
                h = h0
            else:
                sc = keys[: step + 1] @ qs[at]
                sc -= max_reduce(sc)
                exp(sc, out=sc)
                sc /= add_reduce(sc)
                x = sc @ vals[: step + 1]
                x += rows_ob[at]
                x -= add_reduce(x) * inv_d
                x *= 1.0 / sqrt(float(x @ x) * inv_d + eps)
                x *= sg
                x += sb
                h = x
            c = h @ score_w
            c += score_b
            c -= max_reduce(c)
            exp(c, out=c)
            c /= add_reduce(c)
            x = c @ cross_out
            x += h
            x -= add_reduce(x) * inv_d
            x *= 1.0 / sqrt(float(x @ x) * inv_d + eps)
            x *= cg
            x += cb
            f = x @ fc1_w
            f += fc1_b
            y = gelu_fast(f) @ fc2_w
            y += fc2_b
            y += x
            y -= add_reduce(y) * inv_d
            y *= 1.0 / sqrt(float(y @ y) * inv_d + eps)
            logits = y @ logit_w
            logits += logit_b
            nxt = int(logits.argmax())
            if nxt == eos:
                break
            out.append(nxt)
            last = nxt
        return out

    def _cross_terms(self, memory: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cross-attention folded around the word's fixed memory.

        scores = ((h Wq + bq) s) (M Wk + bk)^T = h A + c with A = (Wq s) K^T
        and c = (bq s) K^T; output = softmax(.) (M Wv + bv) Wo + bo, so the
        values are taken through Wo once per word. Works on [W, D] memory or
        a [B, W, D] batch (then A is [B, D, W] and c is [B, 1, W]).
        """
        ca = self.decoder.cross_attn
        if ca is None:
            raise OOVModelError("decoder layer has no cross-attention")
        keys_t = np.swapaxes(memory @ ca.k_w + ca.k_b, -1, -2)           # [.., D, W]
        score_w = (ca.q_w * np.float32(ca.scaling)) @ keys_t              # [.., D, W]
        score_b = (ca.q_b * np.float32(ca.scaling)) @ keys_t              # [.., W]
        out = (memory @ ca.v_w + ca.v_b) @ ca.o_w                         # [.., W, D]
        if memory.ndim == 3:
            score_b = score_b[:, None, :]
        return score_w.astype(np.float32), score_b.astype(np.float32), out.astype(np.float32)

    def _decoder_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For every (position, token): the decoder's layer-normed input row
        and its self-attention query, key and value (the value already through
        the output projection: softmax(.) @ V @ Wo = softmax(.) @ (V Wo)), flattened to
        [position * vocab + token]. A token's row depends only on the token
        and its position, so the greedy loop looks them up instead of
        recomputing them (63 x 63 rows of 128, about 8 MB, built once)."""
        if self._dec_inputs is None:
            sa = self.decoder.self_attn
            vocab = self.embed.shape[0]
            steps = self.max_positions - 1
            pos = self.dec_pos[self.offset : self.offset + steps]
            rows = (pos[:, None, :] + self.embed[None, :, :]).reshape(steps * vocab, -1)
            rows = np.ascontiguousarray(_layer_norm(rows, *self.dec_ln, self.eps), dtype=np.float32)
            qs = (rows @ sa.q_w + sa.q_b) * sa.scaling
            ks = rows @ sa.k_w + sa.k_b
            vos = (rows @ sa.v_w + sa.v_b) @ sa.o_w
            self._dec_inputs = (rows, qs, ks, vos)
        return self._dec_inputs

    def warm(self) -> None:
        """Load the word table and build the decoder lookup tables now,
        rather than on the first word."""
        self.table()
        self._decoder_inputs()

    def memoized(self, word: str) -> bool:
        """True when `word` is answered without decoding (memo or table)."""
        return self._known(word) is not None

    def prefetch(self, words: list[str]) -> None:
        """Decode every word not yet memoised in one batch, into the memo.

        A text with k unknown words then needs one decoding loop instead of
        k: the step count is the longest word's, and each step runs the same
        operations on k rows instead of one (on the l40 bench, three words in
        about 2.4 ms against about 3.4 ms one at a time). Words are padded
        to a common length and padded positions are masked out of every
        attention, so each word sees exactly its own characters.
        tests/test_nano_g2p_oov_fast.py checks the batch against one-at-a-time
        decoding word for word.
        """
        todo = list(dict.fromkeys(w for w in words if w and self._known(w) is None))
        if not todo:
            return
        # A word too long for the model is left for __call__, which raises
        # OOVModelError for it on the caller's normal path.
        todo = [w for w in todo if len(self.encode_word(w)) <= self.max_positions]
        if not todo:
            return
        # One word at a time on the fused single-word path: since it halved
        # the per-step call count, it is as fast as or faster than the batch
        # below for the 1-4 unknown words a sentence brings (measured on the
        # leaderboard-like host, 2026-11-24). The batch is kept as
        # _prefetch_batch for larger groups and its test.
        if len(todo) <= 4:
            for w in todo:
                self(w)
            return
        self._prefetch_batch(todo)

    def _prefetch_batch(self, todo: list[str]) -> None:
        """prefetch's batched decoding of `todo` (unmemoised, in-range words)."""
        encoded = [self.encode_word(w) for w in todo]
        batch, width = len(todo), max(len(ids) for ids in encoded)
        tok = np.zeros((batch, width), dtype=np.int64)
        valid = np.zeros((batch, width), dtype=bool)
        for b, ids in enumerate(encoded):
            tok[b, : len(ids)] = ids
            valid[b, : len(ids)] = True
        key_mask = np.where(valid, np.float32(0.0), np.float32(-np.inf))[:, None, :]   # [B, 1, W]

        # encoder, batched with padded keys masked
        enc = self.encoder
        hidden = self.embed[tok] + self.enc_pos[self.offset : self.offset + width][None]
        hidden = _layer_norm(hidden, *self.enc_ln, self.eps)
        a = enc.self_attn
        q = (hidden @ a.q_w + a.q_b) * a.scaling
        k = hidden @ a.k_w + a.k_b
        v = hidden @ a.v_w + a.v_b
        att = _softmax(q @ k.transpose(0, 2, 1) + key_mask) @ v @ a.o_w + a.o_b
        hidden = _layer_norm(hidden + att, *enc.self_ln, enc.eps)
        ffn = _gelu_fast(hidden @ enc.fc1_w + enc.fc1_b) @ enc.fc2_w + enc.fc2_b
        memory = _layer_norm(hidden + ffn, *enc.final_ln, enc.eps)          # [B, W, D]

        dec = self.decoder
        sa, ca = dec.self_attn, dec.cross_attn
        if ca is None:
            raise OOVModelError("decoder layer has no cross-attention")
        score_w, score_b, cross_out = self._cross_terms(memory)
        score_b = score_b + key_mask                                        # [B, 1, W]
        rows, qs, ks, vos = self._decoder_inputs()
        vocab = self.embed.shape[0]
        steps = self.max_positions - 1
        keys = np.empty((batch, steps, ks.shape[1]), dtype=np.float32)
        vals = np.empty((batch, steps, vos.shape[1]), dtype=np.float32)
        last = np.full(batch, self.start_id, dtype=np.int64)
        outs: list[list[int]] = [[self.start_id] for _ in range(batch)]
        live = np.ones(batch, dtype=bool)
        for step in range(steps):
            at = step * vocab + last
            keys[:, step] = ks[at]
            vals[:, step] = vos[at]
            scores = qs[at][:, None, :] @ keys[:, : step + 1].transpose(0, 2, 1)   # [B, 1, s+1]
            att = (_softmax(scores) @ vals[:, : step + 1])[:, 0] + sa.o_b
            h = _layer_norm(rows[at] + att, *dec.self_ln, dec.eps)
            att = (_softmax(h[:, None, :] @ score_w + score_b) @ cross_out)[:, 0] + ca.o_b
            h = _layer_norm(h + att, *dec.cross_ln, dec.eps)
            ffn = _gelu_fast(h @ dec.fc1_w + dec.fc1_b) @ dec.fc2_w + dec.fc2_b
            h = _layer_norm(h + ffn, *dec.final_ln, dec.eps)
            nxt = np.argmax(h @ self.embed_t + self.logit_bias, axis=-1)
            for b in np.flatnonzero(live).tolist():
                t = int(nxt[b])
                if t == self.eos_id:
                    live[b] = False
                else:
                    outs[b].append(t)
            if not live.any():
                break
            last = np.where(live, nxt, self.start_id)
        for w, out in zip(todo, outs):
            result = "".join(
                self.phoneme_chars[i] for i in out[1:]
                if SPECIAL_TOKEN_COUNT <= i < len(self.phoneme_chars)
            )
            if len(self._memo) >= MEMO_SIZE:
                self._memo.pop(next(iter(self._memo)))
            self._memo[w] = result

    def greedy_reference(self, word: str) -> str:
        """The original full-prefix decoding algorithm, kept as the test
        reference. (The weights are now stored contiguously, so BLAS may round
        it differently from 0.5.1; the release was checked against 0.5.1
        itself on whole texts, phoneme id for phoneme id.)"""
        ids = self.encode_word(word)
        memory = self._encode(np.asarray(ids, dtype=np.int64))
        out: list[int] = [self.start_id]
        for _ in range(self.max_positions - 1):
            logits = self._decode_step(np.asarray(out, dtype=np.int64), memory)
            nxt = int(np.argmax(logits))
            if nxt == self.eos_id:
                break
            out.append(nxt)
        return "".join(
            self.phoneme_chars[i] for i in out[1:]
            if SPECIAL_TOKEN_COUNT <= i < len(self.phoneme_chars)
        )

    def _encode(self, ids: np.ndarray) -> np.ndarray:
        hidden = self.embed[ids] + self.enc_pos[self.offset : self.offset + len(ids)]
        hidden = _layer_norm(hidden, *self.enc_ln, self.eps)
        return self.encoder(hidden, None, causal=False)

    def _decode_step(self, ids: np.ndarray, memory: np.ndarray) -> np.ndarray:
        hidden = self.embed[ids] + self.dec_pos[self.offset : self.offset + len(ids)]
        hidden = _layer_norm(hidden, *self.dec_ln, self.eps)
        hidden = self.decoder(hidden, memory, causal=True)
        return hidden[-1] @ self.embed.T + self.logit_bias


_SHARED: OOVPhonemizer | None = None


def shared_phonemizer() -> OOVPhonemizer:
    """One process-wide instance; loading the npz costs ~3 MB and a few ms."""
    global _SHARED
    if _SHARED is None:
        _SHARED = OOVPhonemizer()
    return _SHARED
