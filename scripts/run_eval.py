"""Reproduce every number in the README.

    python scripts/run_eval.py --pairs 4000

Writes results/metrics.json and prints the tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proseeval import data, evaluate  # noqa: E402
from proseeval.normalize import leakage_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=4000, help="HC3 question pairs to sample")
    ap.add_argument("--ood", type=int, default=2000, help="out-of-domain documents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/metrics.json")
    args = ap.parse_args()

    print("[1/4] auditing the corpus for detokenization leakage")
    machine_raw, human_raw = data.raw_class_texts(limit_pairs=min(args.pairs, 4000), seed=args.seed)
    leaks = leakage_report(machine_raw, human_raw)
    for lk in leaks:
        print("   ", lk)
    worst = max(lk.separation for lk in leaks)
    print(f"    worst single-artifact separation: {worst:.3f}")

    print("\n[2/4] evaluating on RAW text (leakage intact) to quantify the inflation")
    raw = data.load_hc3(limit_pairs=args.pairs, apply_normalize=False, seed=args.seed)
    raw_report, _, _ = evaluate.evaluate(raw, seed=args.seed)
    print(raw_report.table())

    print("\n[3/4] evaluating on NORMALIZED text")
    clean = data.load_hc3(limit_pairs=args.pairs, apply_normalize=True, seed=args.seed)
    ood = data.load_ood(limit=args.ood, seed=args.seed)
    report, ood_report, diag = evaluate.evaluate(clean, seed=args.seed, ood=ood)
    print(report.table())

    print("\n[4/4] zero-shot transfer to a different domain")
    print(ood_report.table())

    print("\nTop features by absolute weight (full model):")
    for row in diag["feature_weights"][:12]:
        print(f"    {row['feature']:<30} {row['weight']:+.3f}")

    print(f"\nMost confident mistakes ({diag['n_mistakes']} total):")
    for m in diag["confident_mistakes"][:5]:
        print(f"    [{m['confidence']:.2f}] {m['source']:<12} true={m['true_label']:<8} "
              f"said={m['predicted']:<8} {m['words']}w")
        print(f"        {m['excerpt'][:150].strip()}...")

    evaluate.save(
        args.out,
        leakage=[{"artifact": l.artifact, "machine": l.positive_rate,
                  "human": l.negative_rate, "separation": l.separation} for l in leaks],
        raw=raw_report,
        normalized=report,
        out_of_domain=ood_report,
        feature_weights=diag["feature_weights"],
        class_balance=diag["class_balance"],
        confident_mistakes=diag["confident_mistakes"],
        config={"pairs": args.pairs, "ood": args.ood, "seed": args.seed},
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
