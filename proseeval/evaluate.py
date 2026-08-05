"""Evaluation harness.

Four scorers are compared on identical splits:

1. `threshold`   -- count rubric hits, flag above a cutoff. This is what a
                    plain regex linter does, and it is the baseline the rest
                    has to beat to justify existing.
2. `rules`       -- logistic regression over rule-hit rates only.
3. `structure`   -- logistic regression over cadence features only.
4. `full`        -- both feature families.

Reporting them separately is the point. It answers a question the rubric alone
cannot: how much of the signal is word choice, and how much is rhythm.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features, rules
from .data import Corpus

RULE_ONLY = rules.LEXICAL_FEATURES + rules.SYNTACTIC_FEATURES
STRUCTURE_ONLY = features.STRUCTURAL_FEATURES
ALL_FEATURES = features.FEATURE_NAMES


@dataclass
class Metrics:
    model: str
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def row(self) -> str:
        return (
            f"{self.model:<24} {self.accuracy:>7.3f} {self.precision:>9.3f} "
            f"{self.recall:>8.3f} {self.f1:>7.3f} {self.roc_auc:>8.3f}"
        )


@dataclass
class Report:
    dataset: str
    metrics: list[Metrics] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def table(self) -> str:
        head = f"{'model':<24} {'acc':>7} {'precision':>9} {'recall':>8} {'f1':>7} {'roc_auc':>8}"
        lines = [f"== {self.dataset} ==", head, "-" * len(head)]
        lines += [m.row() for m in self.metrics]
        return "\n".join(lines)


def _metrics(name: str, y_true, y_pred, y_score) -> Metrics:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        model=name,
        n=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        average_precision=float(average_precision_score(y_true, y_score)),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def _make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]
    )


def threshold_baseline(rows: list[dict[str, float]], cutoff: float = 0.5):
    """Total rubric hits per 100 words, thresholded. The naive linter."""
    scores = np.array([sum(r.get(f, 0.0) for f in RULE_ONLY) for r in rows])
    return scores, (scores >= cutoff).astype(int)


def _tfidf_model() -> Pipeline:
    """Word+punctuation TF-IDF over char n-grams.

    Included because it is the standard strong baseline for text
    classification, and because it is exactly the kind of model that will
    happily learn a corpus artifact. Comparing its raw-vs-normalized scores is
    the cleanest demonstration that the leakage is real.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                      min_df=3, max_features=50000, lowercase=False)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def evaluate(
    corpus: Corpus,
    test_size: float = 0.3,
    seed: int = 0,
    ood: Corpus | None = None,
) -> tuple[Report, Report | None, dict]:
    rows = features.extract_many(corpus.texts)
    y = np.array(corpus.labels)
    texts = np.array(corpus.texts, dtype=object)

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )

    report = Report(dataset=f"{corpus.name} (n={len(y)}, held-out={len(idx_test)})")

    # Baseline: no fitting, just count and threshold.
    scores, preds = threshold_baseline(rows)
    report.metrics.append(
        _metrics("threshold (rule count)", y[idx_test], preds[idx_test], scores[idx_test])
    )

    fitted: dict[str, Pipeline] = {}
    for name, feats in (
        ("logreg: rules", RULE_ONLY),
        ("logreg: structure", STRUCTURE_ONLY),
        ("logreg: rules+structure", ALL_FEATURES),
    ):
        X = features.to_matrix(rows, feats)
        model = _make_model()
        model.fit(X[idx_train], y[idx_train])
        proba = model.predict_proba(X[idx_test])[:, 1]
        report.metrics.append(
            _metrics(name, y[idx_test], (proba >= 0.5).astype(int), proba)
        )
        fitted[name] = model

    # Character TF-IDF: the artifact-hungry baseline.
    tfidf = _tfidf_model()
    tfidf.fit(texts[idx_train], y[idx_train])
    tf_proba = tfidf.predict_proba(texts[idx_test])[:, 1]
    report.metrics.append(
        _metrics("char tf-idf (reference)", y[idx_test], (tf_proba >= 0.5).astype(int), tf_proba)
    )
    fitted["char tf-idf (reference)"] = tfidf

    ood_report = None
    if ood is not None:
        ood_rows = features.extract_many(ood.texts)
        y_ood = np.array(ood.labels)
        ood_report = Report(dataset=f"{ood.name} (n={len(y_ood)}, zero-shot transfer)")
        s, p = threshold_baseline(ood_rows)
        ood_report.metrics.append(_metrics("threshold (rule count)", y_ood, p, s))
        for name, feats in (
            ("logreg: rules", RULE_ONLY),
            ("logreg: structure", STRUCTURE_ONLY),
            ("logreg: rules+structure", ALL_FEATURES),
        ):
            X = features.to_matrix(ood_rows, feats)
            proba = fitted[name].predict_proba(X)[:, 1]
            ood_report.metrics.append(
                _metrics(name, y_ood, (proba >= 0.5).astype(int), proba)
            )
        tf_ood = fitted["char tf-idf (reference)"].predict_proba(
            np.array(ood.texts, dtype=object)
        )[:, 1]
        ood_report.metrics.append(
            _metrics("char tf-idf (reference)", y_ood, (tf_ood >= 0.5).astype(int), tf_ood)
        )

    # Coefficients from the full model, for the writeup.
    full = fitted["logreg: rules+structure"]
    coefs = full.named_steps["clf"].coef_[0]
    weights = sorted(
        ({"feature": f, "weight": float(w)} for f, w in zip(ALL_FEATURES, coefs)),
        key=lambda d: -abs(d["weight"]),
    )

    # Confident mistakes from the full model, for the failure-mode writeup.
    X_all = features.to_matrix(rows, ALL_FEATURES)
    proba_test = full.predict_proba(X_all[idx_test])[:, 1]
    y_test = y[idx_test]
    mistakes = []
    for pos, (p, truth, gi) in enumerate(zip(proba_test, y_test, idx_test)):
        pred = int(p >= 0.5)
        if pred != truth:
            mistakes.append(
                {
                    "confidence": float(p if pred == 1 else 1 - p),
                    "true_label": "machine" if truth == 1 else "human",
                    "predicted": "machine" if pred == 1 else "human",
                    "source": corpus.sources[gi],
                    "words": int(rows[gi]["words"]),
                    "excerpt": corpus.texts[gi][:280],
                }
            )
    mistakes.sort(key=lambda d: -d["confidence"])

    diagnostics = {
        "feature_weights": weights,
        "class_balance": {"machine": int(y.sum()), "human": int(len(y) - y.sum())},
        "confident_mistakes": mistakes[:20],
        "n_mistakes": len(mistakes),
    }
    return report, ood_report, diagnostics


def save(path: str | Path, **payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def default(o):
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(type(o))
    p.write_text(json.dumps(payload, indent=2, default=default))
