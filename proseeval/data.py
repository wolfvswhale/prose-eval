"""Corpus loading.

Two corpora, deliberately different:

HC3 (Hello-SimpleAI) pairs human answers with ChatGPT answers to the same
question across Reddit ELI5, finance, medicine, open QA, and Wikipedia CS/AI.
Pairing matters: it holds topic roughly constant, so a detector cannot succeed
by learning which subjects models get asked about.

Ateeqq/AI-and-Human-Generated-Text is scientific abstracts. It is used only as
a held-out out-of-domain set, never for fitting, to measure how much of the
in-domain score is domain memorization.

One caveat that applies to both and is stated plainly in the README: HC3 was
collected in 2022. A rubric that scores well here has been shown to detect
2022-era model prose, not 2026-era model prose.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .normalize import normalize

HC3_REPO = "Hello-SimpleAI/HC3"
OOD_REPO = "Ateeqq/AI-and-Human-Generated-Text"

MACHINE = 1
HUMAN = 0


@dataclass
class Corpus:
    texts: list[str]
    labels: list[int]
    sources: list[str]
    name: str

    def __len__(self) -> int:
        return len(self.texts)

    def balance(self) -> tuple[int, int]:
        m = sum(self.labels)
        return m, len(self.labels) - m


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset")


def load_hc3(
    limit_pairs: int | None = 4000,
    apply_normalize: bool = True,
    seed: int = 0,
    min_words: int = 40,
) -> Corpus:
    """Load HC3 as one human and one machine answer per question.

    Taking a single answer from each side per question keeps the classes
    balanced and prevents prolific answerers from dominating.
    """
    path = _download(HC3_REPO, "all.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)

    texts: list[str] = []
    labels: list[int] = []
    sources: list[str] = []

    for row in rows:
        humans = [a for a in row.get("human_answers", []) if a and a.strip()]
        machines = [a for a in row.get("chatgpt_answers", []) if a and a.strip()]
        if not humans or not machines:
            continue
        h, m = humans[0], machines[0]
        if apply_normalize:
            h, m = normalize(h), normalize(m)
        if len(h.split()) < min_words or len(m.split()) < min_words:
            continue
        src = row.get("source", "unknown")
        texts += [h, m]
        labels += [HUMAN, MACHINE]
        sources += [src, src]
        if limit_pairs and len(texts) >= limit_pairs * 2:
            break

    return Corpus(texts, labels, sources, name="HC3" + ("" if apply_normalize else " (raw)"))


def load_ood(limit: int | None = 2000, apply_normalize: bool = True, seed: int = 0) -> Corpus:
    """Scientific abstracts, human vs machine. Held out for out-of-domain testing."""
    import pandas as pd

    path = _download(OOD_REPO, "test.csv")
    df = pd.read_csv(path).dropna(subset=["abstract", "label"])
    df = df.sample(frac=1.0, random_state=seed)
    if limit:
        df = df.head(limit)

    texts = [normalize(t) if apply_normalize else t for t in df["abstract"].astype(str)]
    labels = [int(v) for v in df["label"]]
    return Corpus(texts, labels, ["abstract"] * len(texts), name="Scientific abstracts (OOD)")


def raw_class_texts(limit_pairs: int = 4000, seed: int = 0) -> tuple[list[str], list[str]]:
    """Un-normalized machine and human text, for the leakage audit only."""
    path = _download(HC3_REPO, "all.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    human, machine = [], []
    for row in rows:
        h = [a for a in row.get("human_answers", []) if a and a.strip()]
        m = [a for a in row.get("chatgpt_answers", []) if a and a.strip()]
        if h and m:
            human.append(h[0])
            machine.append(m[0])
        if len(human) >= limit_pairs:
            break
    return machine, human
