#!/usr/bin/env python3
"""
Train NOOTZSCOPE ML model from SSOSM dataset lots.

Expected lot structure:
  LOT_FOLDER/
    Clean/
      *.txt / *.html ...
    Spam/
      *.txt / *.html ...

Trains TF-IDF (word + char ngrams) + LogisticRegression, then saves model.joblib
containing:
  - preprocessor config
  - vectorizers
  - classifier
  - recommended threshold (optional tuning)

Usage:
  python tools/train_model.py --lot data/lot1 --out models/model.joblib
  python tools/train_model.py --lot data/lot2 --out models/model.joblib
  python tools/train_model.py --lot data/lot1 --lot data/lot2 --out models/model.joblib

Optional threshold tuning (on validation split):
  python tools/train_model.py --lot data/lot1 --out models/model.joblib --tune-threshold
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline


# -------------------------
# Robust email file reading
# -------------------------

def read_email_file(file_path: Path) -> Tuple[str, str]:
    """
    First line = subject, rest = body.
    Robust decoding: utf-8-sig then latin-1 fallback.
    """
    data: str
    try:
        data = file_path.read_text(encoding="utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        data = file_path.read_text(encoding="latin-1", errors="replace")
    except Exception:
        b = file_path.read_bytes()
        data = b.decode("utf-8", errors="replace")

    data = data.replace("\r\n", "\n").replace("\r", "\n")
    if not data:
        return ("", "")
    lines = data.split("\n")
    subject = lines[0].strip() if lines else ""
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return subject, body


HTML_TAG_RE = re.compile(r"<[^>]+>")


def basic_normalize(subject: str, body: str, strip_html: bool = False) -> str:
    text = (subject + "\n" + body).strip()
    text = text.replace("\u00a0", " ")  # nbsp
    if strip_html:
        text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# -------------------------
# Dataset loading
# -------------------------

def load_lot(lot_folder: Path, strip_html: bool) -> Tuple[List[str], List[int]]:
    clean_dir = lot_folder / "Clean"
    spam_dir = lot_folder / "Spam"
    if not clean_dir.is_dir() or not spam_dir.is_dir():
        raise ValueError(f"Lot folder must contain Clean/ and Spam/: {lot_folder}")

    X: List[str] = []
    y: List[int] = []

    def add_dir(d: Path, label: int) -> None:
        for p in sorted([x for x in d.iterdir() if x.is_file()], key=lambda z: z.name.lower()):
            subj, body = read_email_file(p)
            X.append(basic_normalize(subj, body, strip_html=strip_html))
            y.append(label)

    add_dir(clean_dir, 0)
    add_dir(spam_dir, 1)
    return X, y


# -------------------------
# Training + threshold tune
# -------------------------

@dataclass
class TrainedBundle:
    pipeline: Pipeline
    threshold: float
    strip_html: bool


def build_pipeline() -> Pipeline:
    # Word + char TF-IDF union
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
        n_jobs=None,
        solver="lbfgs",
    )

    return Pipeline([
        ("features", feats),
        ("clf", clf),
    ])


def tune_threshold(probas, y_true) -> float:
    # Sweep thresholds and pick best F1 (common in spam tasks).
    best_t = 0.5
    best_f1 = -1.0
    for t in [i / 100 for i in range(20, 81)]:  # 0.20..0.80
        preds = (probas >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lot", action="append", required=True, help="Path to lot folder (can be repeated).")
    ap.add_argument("--out", required=True, help="Output model path, e.g. models/model.joblib")
    ap.add_argument("--strip-html", action="store_true", help="Strip HTML tags during normalization.")
    ap.add_argument("--tune-threshold", action="store_true", help="Tune decision threshold on validation split.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    lot_paths = [Path(p).resolve() for p in args.lot]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_all: List[str] = []
    y_all: List[int] = []

    for lp in lot_paths:
        X, y = load_lot(lp, strip_html=args.strip_html)
        X_all.extend(X)
        y_all.extend(y)

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
        # Retrain on full data
        pipeline.fit(X_all, y_all)
    else:
        pipeline.fit(X_all, y_all)

    bundle = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "strip_html": bool(args.strip_html),
        "version": "1.0",
    }
    joblib.dump(bundle, out_path)
    print(f"[OK] Saved model bundle to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
