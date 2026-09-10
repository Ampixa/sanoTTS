"""Stop Piper's Chinese front end spawning a DataLoader for every sentence.

    from piper_frontend_workers import disable_g2p_dataloader_workers
    voice = PiperVoice.load(model, config)
    disable_g2p_dataloader_workers(voice)

`piper.phonemize_chinese.ChinesePhonemizer` builds a g2pW `G2PWConverter` with
the worker count baked into the downloaded model's own config.py (num_workers =
2), and `G2PWConverter.__call__` constructs a fresh torch DataLoader for EVERY
sentence. On macOS each call therefore spawns two worker processes, and the
file descriptors they take are not all given back: measured on k2, open fds
climb by about 1.4 per sentence against a 256 soft limit, so a pack of more
than ~200 rows dies partway through with

    OSError: [Errno 24] Too many open files

which is exactly how the first zh_CN acoustic pack failed, 200 rows into 1781.

It is also where all the time goes. Same box, same rows: 9.2 s per FLORES
sentence with the workers, 0.17 s without -- 55x, for spawning two processes to
fetch a batch containing one sentence.

Setting num_workers to 0 iterates that batch in the calling process. Verified
on k2 over 12 FLORES rows that the phoneme lists and the 1775 phoneme ids that
come out are identical either way; the only thing that changes is how the batch
is fetched.

eSpeak teachers are unaffected -- that front end is a C bridge with no
dataloader and no worker processes -- so this is a no-op for them.
"""

from __future__ import annotations

from typing import Any

PROBE_TEXT = "今天天气很好。"


def disable_g2p_dataloader_workers(voice: Any) -> bool:
    """Make `voice`'s Chinese front end fetch its batches in-process.

    Returns True when the setting was applied, False for a front end that has
    no g2pW converter (every non-pinyin teacher).

    PiperVoice builds the phonemizer lazily on the first phonemize() call and
    caches it on a private attribute, so this builds the same object up front
    and caches it in the same place. That hook is private, so the result is
    checked rather than assumed: if a phonemize() call does not leave our
    instance in place, piper is no longer reading that attribute and the
    caller would silently get the leaking path back.
    """
    from piper.config import PhonemeType  # noqa: PLC0415

    if voice.config.phoneme_type != PhonemeType.PINYIN:
        return False

    from piper.phonemize_chinese import ChinesePhonemizer  # noqa: PLC0415

    phonemizer = getattr(voice, "_chinese_phonemizer", None)
    if phonemizer is None:
        phonemizer = ChinesePhonemizer(voice.download_dir / "g2pW")
        voice._chinese_phonemizer = phonemizer  # noqa: SLF001 - piper's own cache slot
    phonemizer.g2p.num_workers = 0

    voice.phonemize(PROBE_TEXT)
    if getattr(voice, "_chinese_phonemizer", None) is not phonemizer:
        raise RuntimeError(
            "piper no longer caches its Chinese phonemizer on "
            "PiperVoice._chinese_phonemizer, so the dataloader workers are "
            "still live. Rendering a pack this way exhausts the process's file "
            "descriptors after a few hundred rows and is 55x slower. Update "
            "tools/piper_frontend_workers.py against the installed piper.")
    return True
