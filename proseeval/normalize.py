"""Text normalization.

The point of this module is narrow and important: several public
human-vs-machine corpora were assembled by detokenizing human text with a
whitespace tokenizer while leaving model output untouched. That leaves a
punctuation fingerprint that has nothing to do with writing style, and any
detector evaluated on the raw text will report inflated scores because it is
partly learning "was this string detokenized" instead of "was this written by
a machine".

`normalize` removes that fingerprint. `leakage_report` measures it, so the
inflation can be quantified rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# " n't" -> "n't", " 's" -> "'s", and friends.
_CLITIC = re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m|'S|'RE)\b")
# " ," -> ",", " ." -> "."
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]}])")
# "( x" -> "(x"
_SPACE_AFTER_OPEN = re.compile(r"([(\[{$])\s+")
# " - " used as a word joiner by tokenizers: "Kim Jong - Un" -> "Kim Jong-Un"
_SPACED_HYPHEN = re.compile(r"(?<=[A-Za-z0-9])\s+-\s+(?=[A-Za-z0-9])")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Strip detokenization artifacts without touching authored style.

    Deliberately does NOT alter word choice, sentence boundaries, casing,
    contractions-as-written, or em dashes, since all of those carry real
    stylistic signal that a detector is supposed to see.
    """
    if not text:
        return ""
    out = _CLITIC.sub(r"\1", text)
    out = _SPACED_HYPHEN.sub("-", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _SPACE_AFTER_OPEN.sub(r"\1", out)
    out = _MULTISPACE.sub(" ", out)
    out = _MULTINEWLINE.sub("\n\n", out)
    return out.strip()


@dataclass(frozen=True)
class LeakageReport:
    """Per-class rate at which a corpus artifact appears."""

    artifact: str
    positive_rate: float
    negative_rate: float

    @property
    def separation(self) -> float:
        """How cleanly this artifact alone splits the classes.

        1.0 means the artifact perfectly identifies the class, which means the
        corpus is unusable in raw form.
        """
        return abs(self.positive_rate - self.negative_rate)

    def __str__(self) -> str:
        return (
            f"{self.artifact:<28} machine={self.positive_rate:.3f} "
            f"human={self.negative_rate:.3f} separation={self.separation:.3f}"
        )


_ARTIFACTS = {
    "space_before_punct": _SPACE_BEFORE_PUNCT,
    "detached_clitic": _CLITIC,
    "spaced_hyphen": _SPACED_HYPHEN,
}


def leakage_report(machine: list[str], human: list[str]) -> list[LeakageReport]:
    """Measure how much each known artifact separates the two classes."""

    def rate(texts: list[str], pattern: re.Pattern[str]) -> float:
        if not texts:
            return 0.0
        return sum(1 for t in texts if pattern.search(t)) / len(texts)

    return [
        LeakageReport(name, rate(machine, pat), rate(human, pat))
        for name, pat in _ARTIFACTS.items()
    ]
