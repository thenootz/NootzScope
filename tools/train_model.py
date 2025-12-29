#!/usr/bin/env python3
"""
Train NOOTZSCOPE model using HashingVectorizer + SGDClassifier (log_loss).
This keeps model.joblib very small (<10MB typically) because no vocabulary is stored.

Usage:
  python tools/train_model.py --lot data/Lot1 --out models/model.joblib --tune-threshold --verbose
  python tools/train_model.py --lot data/Lot1 --lot data/Lot2 --out models/model.joblib --tune-threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# ---- bootstrap imports so tools/ can import src/ modules ----
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
# ------------------------------------------------------------

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from featurize import combine_lots
from model_io import ModelBundle, save_bundle


def build_pipeline(n_features: int) -> Pipeline:
    feats = FeatureUnion(
        transformer_list=[
            ("word_hash", HashingVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                n_features=n_features,
                alternate_sign=False,
                norm="l2",
                lowercase=False,   # featurize already lowercases
            )),
            ("char_hash", HashingVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                n_features=n_features,
                alternate_sign=False,
                norm="l2",
                lowercase=False,
            )),
        ],
        n_jobs=None,
    )

    clf = SGDClassifier(
        loss="log_loss",          # enables predict_proba
        alpha=1e-5,
        max_iter=50,
        tol=1e-3,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline([
        ("features", feats),
        ("clf", clf),
    ])


def tune_threshold(probas: np.ndarray, y_true: List[int]) -> float:
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lot", action="append", required=True, help="Path to lot folder (repeatable).")
    ap.add_argument("--out", required=True, help="Output model path, e.g. models/model.joblib")
    ap.add_argument("--strip-html", action="store_true", help="Strip HTML tags during normalization.")
    ap.add_argument("--tune-threshold", action="store_true", help="Tune threshold on a validation split.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    ap.add_argument("--verbose", action="store_true", help="List every file used during training.")
    ap.add_argument(
        "--n-features",
        type=int,
        default=2**18,  # 262,144
        help="HashingVectorizer n_features (default: 2**18). Try 2**17 if you want even smaller/faster.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    lot_paths = [Path(p).resolve() for p in args.lot]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_all, y_all = combine_lots(
        lot_folders=lot_paths,
        strip_html=args.strip_html,
        verbose=args.verbose,
    )

    if len(X_all) < 10:
        print("[ERROR] Not enough samples to train.", file=sys.stderr)
        return 2

    pipeline = build_pipeline(n_features=int(args.n_features))

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

        pipeline.fit(X_all, y_all)
    else:
        pipeline.fit(X_all, y_all)

    bundle = ModelBundle(
        pipeline=pipeline,
        threshold=float(threshold),
        strip_html=bool(args.strip_html),
        version="1.1-hashing-sgd",
        meta={
            "trainer": "tools/train_model.py",
            "lots": [str(p) for p in lot_paths],
            "n_features": int(args.n_features),
            "model": "HashingVectorizer+SGDClassifier(log_loss)",
        },
    )

    save_bundle(out_path, bundle)
    print(f"[OK] Saved model bundle to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
