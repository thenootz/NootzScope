#!/usr/bin/env python3
"""
Train NOOTZSCOPE ML model from SSOSM dataset lots.

Expected lot structure:
  LOT_FOLDER/
    Clean/
    Spam/

Trains TF-IDF (word + char ngrams) + LogisticRegression, then saves model.joblib bundle.

Usage:
  python tools/train_model.py --lot data/lot1 --out models/model.joblib --tune-threshold --verbose
  python tools/train_model.py --lot data/lot1 --lot data/lot2 --out models/model.joblib --tune-threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure src/ is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from featurize import combine_lots
from model_io import ModelBundle, save_bundle


# -------------------------
# Pipeline
# -------------------------

def build_pipeline() -> Pipeline:
    feats = FeatureUnion(
        transformer_list=[
            ("word_tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("char_tfidf", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
            )),
        ],
        n_jobs=None,
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
    )

    return Pipeline([
        ("features", feats),
        ("clf", clf),
    ])


def tune_threshold(probas, y_true) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for i in range(20, 81):  # 0.20..0.80
        t = i / 100.0
        preds = (probas >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lot", action="append", required=True,
                    help="Path to lot folder (repeatable).")
    ap.add_argument("--out", required=True,
                    help="Output model path, e.g. models/model.joblib")
    ap.add_argument("--strip-html", action="store_true",
                    help="Strip HTML tags during normalization.")
    ap.add_argument("--tune-threshold", action="store_true",
                    help="Tune decision threshold on a validation split.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for split.")
    ap.add_argument("--verbose", action="store_true",
                    help="List every file used during training (from lots).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    lot_paths = [Path(p).resolve() for p in args.lot]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # This will list every file if verbose=True (via featurize.load_lot)
    X_all, y_all = combine_lots(
        lot_folders=lot_paths,
        strip_html=args.strip_html,
        verbose=args.verbose,
    )

    if len(X_all) < 10:
        print("[ERROR] Not enough samples to train.", file=sys.stderr)
        return 2

    pipeline = build_pipeline()

    threshold = 0.5
    if args.tune_threshold:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
        )
        pipeline.fit(X_tr, y_tr)
        probas = pipeline.predict_proba(X_va)[:, 1]
        threshold = tune_threshold(probas, y_va)

        va_preds = (probas >= threshold).astype(int)
        acc = accuracy_score(y_va, va_preds)
        f1 = f1_score(y_va, va_preds, zero_division=0)
        print(f"[OK] Validation: acc={acc:.4f} f1={f1:.4f} threshold={threshold:.2f}")

        # Retrain on full dataset after choosing threshold
        pipeline.fit(X_all, y_all)
    else:
        pipeline.fit(X_all, y_all)

    bundle = ModelBundle(
        pipeline=pipeline,
        threshold=float(threshold),
        strip_html=bool(args.strip_html),
        version="1.0",
        meta={
            "trainer": "tools/train_model.py",
            "lots": [str(p) for p in lot_paths],
        },
    )

    save_bundle(out_path, bundle)
    print(f"[OK] Saved model bundle to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
