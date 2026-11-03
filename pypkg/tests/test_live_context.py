"""Conservative verb/adjective selection for live; no model assets required.

Run: python -m pytest pypkg/tests/test_live_context.py
  or: python -m unittest discover -s pypkg/tests -p test_live_context.py
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sanotts import nano_g2p as G, piper_g2p  # noqa: E402

VERBS = (
    "I live here.",
    "You live nearby.",
    "We live together.",
    "They live in town.",
    "I still live here.",
    "I do not live there.",
    "I don't live there.",
    "They never live alone.",
    "They will live there.",
    "We want to live here.",
    "Do you live here?",
    "I LIVE here.",
    "I really live here.",
    "You do live here.",
    "They can live here.",
    "They did not live here.",
    "They don\u2019t live here.",
    "They currently live here.",
)
ADJECTIVES = (
    "A live broadcast.",
    "We are live.",
    "I am live.",
    "The stream is live.",
    "They were live on television.",
    "This has live music.",
    "We have live music.",
    "I have a live broadcast.",
    "The band is playing live.",
    "We will be live.",
    "We are not live.",
    "We're live on air.",
    "This is actually live.",
    "The live stream.",
    "Live music.",
    "We are still live.",
    "I am watching a live broadcast.",
    "It is LIVE.",
    "We\u2019re live.",
    "It isn't live.",
    "We have not live music but a recording.",
    "We do live broadcasts.",
    "I do live interviews.",
    "Do live music.",
)


class LiveContextTests(unittest.TestCase):
    def test_clear_verb_contexts(self):
        for text in VERBS:
            with self.subTest(text=text):
                phones = G.phonemize_to_string(text)
                self.assertIn("lˈɪv", phones)
                self.assertNotIn("lˈIv", phones)

    def test_clear_adjective_contexts(self):
        for text in ADJECTIVES:
            with self.subTest(text=text):
                phones = G.phonemize_to_string(text)
                self.assertIn("lˈIv", phones)
                self.assertNotIn("lˈɪv", phones)

    def test_piperlite_mapping_preserves_both_vowels(self):
        for text, expected in [("I live here.", "lˈɪv"), ("We are live.", "lˈaɪv")]:
            with self.subTest(text=text):
                self.assertIn(
                    expected,
                    piper_g2p.misaki_to_espeak_ipa(G.phonemize_to_string(text)),
                )

    def test_do_not_cross_punctuation_or_rewrite_other_words(self):
        for text in (
            "I. Live broadcast.",
            "I, live broadcast.",
            "I; live broadcast.",
            "I: live broadcast.",
            "I -- live broadcast.",
        ):
            with self.subTest(text=text):
                self.assertIn("lˈIv", G.phonemize_to_string(text))
        for text in (
            "I read here.",
            "We close the door.",
            "They use it.",
            "The close race.",
            "I lived here.",
            "They are alive.",
            "The liveable house.",
            "I watch them live.",
            "Live.",
        ):
            tokens = G.tokenize(text)
            before = deepcopy(tokens)
            G._retag_live(tokens)
            with self.subTest(text=text):
                self.assertEqual(
                    [(x.text, x.tag, x.whitespace) for x in before],
                    [(x.text, x.tag, x.whitespace) for x in tokens],
                )

    def test_input_order_and_word_framing_are_unchanged(self):
        for text in (*VERBS, *ADJECTIVES):
            tokens = G.tokenize(text)
            self.assertEqual(
                " ".join(text.split()), "".join(x.text + x.whitespace for x in tokens)
            )
            live = next(x for x in tokens if x.text.casefold() == "live")
            self.assertTrue(
                live.tag.startswith(("VB", "JJ")) or live.tag in ("NN", "NNP")
            )

    def test_other_tagged_dictionary_words_are_untouched(self):
        gold = json.loads((ROOT / "sanotts/g2p_data/us_gold.json").read_text("utf-8"))
        for word, entry in gold.items():
            if word.casefold() == "live" or not isinstance(entry, dict):
                continue
            for prefix in ("I ", "We are ", "We have ", "We will "):
                # Compare against tokenization without this new pass, not an
                # already-retagged result that could hide an idempotent error.
                with patch.object(G, "_retag_live"):
                    tokens = G.tokenize(prefix + word)
                before = deepcopy(tokens)
                G._retag_live(tokens)
                with self.subTest(word=word, prefix=prefix):
                    self.assertEqual(before, tokens)

    def test_context_scan_is_bounded(self):
        tokens = [G.Token(text="I", tag="PRP", whitespace=" ")]
        tokens.extend(G.Token(text="still", tag="RB", whitespace=" ") for _ in range(4))
        tokens.append(G.Token(text="live", tag="NN", whitespace=""))
        G._retag_live(tokens)
        self.assertEqual("NN", tokens[-1].tag)


if __name__ == "__main__":
    unittest.main()
