"""Feature extraction.

Two families:

*Rule features* count rubric hits, normalized per 100 words so that a long
document is not automatically suspicious.

*Structural features* describe cadence. The house standard's claim is that
structure is a stronger tell than vocabulary, because a writer can swap out
every flagged word and still produce text whose rhythm is flat. These features
exist to test that claim rather than assume it, which is why the evaluation
reports rule-only and structure-only models separately.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter

from . import rules

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\s\n]+")
_WORD = re.compile(r"[A-Za-z0-9']+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

STRUCTURAL_FEATURES = (
    "words",
    "mean_sentence_len",
    "sd_sentence_len",
    "cv_sentence_len",
    "pct_short_sentences",
    "pct_long_sentences",
    "mean_paragraph_sentences",
    "cv_paragraph_sentences",
    "list_marker_rate",
    "em_dash_rate",
    "semicolon_rate",
    "colon_rate",
    "contraction_rate",
    "type_token_ratio",
    "hapax_ratio",
    "comma_per_sentence",
    "triplet_rate",
    "sentence_start_repeat",
    "mean_word_len",
)

FEATURE_NAMES = tuple(list(rules.LEXICAL_FEATURES + rules.SYNTACTIC_FEATURES) + list(STRUCTURAL_FEATURES))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _cv(values: list[int | float]) -> float:
    """Coefficient of variation: spread scaled by mean.

    Used instead of raw standard deviation so the number is comparable across
    texts with different average sentence lengths.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / mean


def extract(text: str) -> dict[str, float]:
    """Return the full feature vector for one document."""
    sentences = _sentences(text)
    words = _words(text)
    n_words = len(words)
    paragraphs = [p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]

    feats: dict[str, float] = {}

    # Rule hits, per 100 words.
    scale = 100.0 / n_words if n_words else 0.0
    for group in rules.ALL_GROUPS:
        feats[f"rule_{group}"] = rules.count_group(text, group) * scale

    if not sentences or not n_words:
        for name in STRUCTURAL_FEATURES:
            feats.setdefault(name, 0.0)
        feats["words"] = float(n_words)
        return feats

    sent_lens = [len(_words(s)) for s in sentences]
    para_sent_counts = [max(1, len(_sentences(p))) for p in paragraphs] or [len(sentences)]

    feats["words"] = float(n_words)
    feats["mean_sentence_len"] = statistics.fmean(sent_lens)
    feats["sd_sentence_len"] = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    feats["cv_sentence_len"] = _cv(sent_lens)
    feats["pct_short_sentences"] = sum(1 for n in sent_lens if n <= 8) / len(sent_lens)
    feats["pct_long_sentences"] = sum(1 for n in sent_lens if n >= 30) / len(sent_lens)
    feats["mean_paragraph_sentences"] = statistics.fmean(para_sent_counts)
    feats["cv_paragraph_sentences"] = _cv(para_sent_counts)

    feats["list_marker_rate"] = len(re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s", text)) * scale
    feats["em_dash_rate"] = text.count("—") * scale
    feats["semicolon_rate"] = text.count(";") * scale
    feats["colon_rate"] = text.count(":") * scale
    feats["contraction_rate"] = len(re.findall(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", text, re.I)) * scale

    counts = Counter(words)
    feats["type_token_ratio"] = len(counts) / n_words
    feats["hapax_ratio"] = sum(1 for c in counts.values() if c == 1) / n_words
    feats["comma_per_sentence"] = text.count(",") / len(sentences)
    feats["mean_word_len"] = statistics.fmean(len(w) for w in words)

    # "x, y, and z" -- the reflexive triplet.
    feats["triplet_rate"] = len(re.findall(r"\b[\w-]+,\s+[\w-]+,\s+(?:and|or)\s+[\w-]+", text)) * scale

    # How often consecutive sentences open with the same word.
    starts = [(_words(s) or [""])[0] for s in sentences]
    repeats = sum(1 for a, b in zip(starts, starts[1:]) if a and a == b)
    feats["sentence_start_repeat"] = repeats / max(1, len(starts) - 1)

    return feats


def extract_many(texts: list[str]) -> list[dict[str, float]]:
    return [extract(t) for t in texts]


def to_matrix(rows: list[dict[str, float]], names: tuple[str, ...] = FEATURE_NAMES):
    """Dict rows to a dense matrix, with log scaling on the raw length feature."""
    import numpy as np

    out = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            v = row.get(name, 0.0)
            if name == "words":
                v = math.log1p(v)
            out[i, j] = v
    return out
