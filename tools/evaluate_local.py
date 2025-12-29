#!/usr/bin/env python3
"""
Local evaluator for NOOTZSCOPE (SSOSM).

Evaluates a trained model (model.joblib) on one or more lots that have:
  LOT/
    Clean/
    Spam/

Metrics:
- Accuracy
- Precision / Recall / F1 (spam as positive class)
- Confusion matrix (TN FP / FN TP)

Also supports:
- Threshold sweep (to pick best threshold on the given evaluation set)
- Listing misclassified files
- Optional per-lot breakdown

Usage examples:
  python tools/evaluate_local.py --lot data/lot1 --model models/model.joblib
  python tools/evaluate_local.py --lot data/lot1 --lot data/lot2 --model models/model.joblib --per-lot
  python tools/evaluate_local.py --lot data/lot1 --model models/model.joblib --sweep
  python tools/evaluate_local.py --lot data/lot1 --model models/model.joblib --list-mis --max-mis 30
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# -------------------------
# Robust email file reading
# -------------------------

def read_email_file(file_path: Path) -> Tuple[str, str]:
    """
    First line = subject, remaining = body. Robust decoding.
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


def normalize_text(subject: str, body: str, strip_html: bool) -> str:
    text = (subject + "\n" + body).strip()
    text = text.replace("\u00a0", " ")
    if strip_html:
        text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# -------------------------
# Data loading
# -------------------------

@dataclass
class Sample:
    lot_name: str
    rel_path: str  # e.g. "Clean/abc.txt"
    filename: str  # base name only
    y_true: int    # 0 clean, 1 spam
    text: str


def load_lot_samples(lot_folder: Path, strip_html: bool) -> List[Sample]:
    clean_dir = lot_folder / "Clean"
    spam_dir = lot_folder / "Spam"
    if not clean_dir.is_dir() or not spam_dir.is_dir():
        raise ValueError(f"Lot folder must contain Clean/ and Spam/: {lot_folder}")

    lot_name = lot_folder.name
    samples: List[Sample] = []

    def add_dir(d: Path, label: int, subdir_name: str) -> None:
        files = sorted([p for p in d.iterdir() if p.is_file()], key=lambda x: x.name.lower())
        for p in files:
            subj, body = read_email_file(p)
            text = normalize_text(subj, body, strip_html=strip_html)
            samples.append(
                Sample(
                    lot_name=lot_name,
                    rel_path=f"{subdir_name}/{p.name}",
                    filename=p.name,
                    y_true=label,
                    text=text,
                )
            )

    add_dir(clean_dir, 0, "Clean")
    add_dir(spam_dir, 1, "Spam")
    return samples


# -------------------------
# Model loading + scoring
# -------------------------

def load_bundle(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Invalid model bundle. Expected dict with key 'pipeline'.")
    return bundle


def get_spam_scores(pipeline, texts: List[str]) -> List[float]:
    """
    Returns a spam score per sample.
    Prefer predict_proba; fallback to decision_function; last resort uses predict.
    """
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(texts)
        return [float(p[1]) for p in probs]
    if hasattr(pipeline, "decision_function"):
        margins = pipeline.decision_function(texts)
        # margins can be np array; cast per element
        return [float(m) for m in margins]
    preds = pipeline.predict(texts)
    return [float(v) for v in preds]  # 0/1


def apply_threshold(scores: List[float], threshold: float, scores_are_prob: bool) -> List[int]:
    """
    If scores are probabilities: spam if score >= threshold
    If scores are margins: spam if score >= 0.0 unless user provides threshold explicitly
    """
    if scores_are_prob:
        return [1 if s >= threshold else 0 for s in scores]
    # margin case: typical boundary is 0
    return [1 if s >= threshold else 0 for s in scores]


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def print_confusion(y_true: List[int], y_pred: List[int]) -> None:
    # confusion_matrix returns [[TN, FP],[FN, TP]] for labels [0,1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print("Confusion matrix (labels: 0=clean, 1=spam):")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")


def sweep_threshold(scores: List[float], y_true: List[int], scores_are_prob: bool) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5 if scores_are_prob else 0.0
    best = {"f1": -1.0}

    if scores_are_prob:
        candidates = [i / 100 for i in range(10, 91)]  # 0.10..0.90
    else:
        # margin thresholds around 0
        candidates = [i / 10 for i in range(-20, 21)]  # -2.0 .. +2.0

    for t in candidates:
        y_pred = apply_threshold(scores, float(t), scores_are_prob=scores_are_prob)
        m = compute_metrics(y_true, y_pred)
        if m["f1"] > best["f1"]:
            best_t = float(t)
            best = m

    return best_t, best


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local evaluator for SSOSM lots")
    ap.add_argument("--lot", action="append", required=True, help="Path to lot folder (repeatable).")
    ap.add_argument("--model", required=True, help="Path to model.joblib bundle.")
    ap.add_argument("--threshold", type=float, default=None, help="Override threshold (prob or margin).")
    ap.add_argument("--sweep", action="store_true", help="Sweep threshold and print best F1.")
    ap.add_argument("--per-lot", action="store_true", help="Print metrics per lot as well as overall.")
    ap.add_argument("--list-mis", action="store_true", help="List misclassified files.")
    ap.add_argument("--max-mis", type=int, default=50, help="Max misclassifications to print.")
    return ap.parse_args()


def evaluate_samples(samples: List[Sample], bundle: dict, threshold_override: float | None, sweep: bool,
                     list_mis: bool, max_mis: int, title: str) -> None:
    pipeline = bundle["pipeline"]
    strip_html = bool(bundle.get("strip_html", False))

    # NOTE: samples already normalized with strip_html passed in load; ensure consistency:
    # We re-normalize if mismatch is likely (simple and safe).
    texts = []
    for s in samples:
        # s.text is normalized already, but might have been normalized with different strip_html.
        # We'll trust it if caller used bundle strip_html; evaluate_local passes that.
        texts.append(s.text)

    scores = get_spam_scores(pipeline, texts)
    scores_are_prob = hasattr(pipeline, "predict_proba")

    # Choose threshold
    if threshold_override is not None:
        threshold = float(threshold_override)
    else:
        threshold = float(bundle.get("threshold", 0.5 if scores_are_prob else 0.0))

    y_true = [s.y_true for s in samples]

    if sweep:
        best_t, best_m = sweep_threshold(scores, y_true, scores_are_prob=scores_are_prob)
        print(f"\n[{title}] Best threshold by F1: {best_t:.2f} (scores={'proba' if scores_are_prob else 'margin'})")
        print(f"  acc={best_m['accuracy']:.4f} prec={best_m['precision']:.4f} rec={best_m['recall']:.4f} f1={best_m['f1']:.4f}")

    y_pred = apply_threshold(scores, threshold, scores_are_prob=scores_are_prob)
    m = compute_metrics(y_true, y_pred)

    print(f"\n[{title}] Using threshold={threshold:.2f} (scores={'proba' if scores_are_prob else 'margin'})")
    print(f"  acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}")
    print_confusion(y_true, y_pred)

    if list_mis:
        mis = []
        for s, yp, sc in zip(samples, y_pred, scores):
            if yp != s.y_true:
                mis.append((s.lot_name, s.rel_path, s.y_true, yp, float(sc)))
        if not mis:
            print("No misclassifications.")
            return

        print(f"\nMisclassifications (showing up to {max_mis}):")
        for i, (lot_name, rel_path, yt, yp, sc) in enumerate(mis[:max_mis], start=1):
            yt_s = "SPAM" if yt == 1 else "CLEAN"
            yp_s = "SPAM" if yp == 1 else "CLEAN"
            print(f"  {i:>3}. [{lot_name}] {rel_path}  true={yt_s} pred={yp_s} score={sc:.4f}")
        if len(mis) > max_mis:
            print(f"  ... and {len(mis) - max_mis} more.")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 2

    bundle = load_bundle(model_path)
    strip_html = bool(bundle.get("strip_html", False))

    all_samples: List[Sample] = []
    per_lot_samples: Dict[str, List[Sample]] = {}

    for lot in args.lot:
        lot_path = Path(lot).resolve()
        samples = load_lot_samples(lot_path, strip_html=strip_html)
        all_samples.extend(samples)
        per_lot_samples[lot_path.name] = samples

    if not all_samples:
        print("[ERROR] No samples found.", file=sys.stderr)
        return 2

    # Overall
    evaluate_samples(
        samples=all_samples,
        bundle=bundle,
        threshold_override=args.threshold,
        sweep=args.sweep,
        list_mis=args.list_mis,
        max_mis=args.max_mis,
        title="OVERALL",
    )

    # Per-lot
    if args.per_lot and len(per_lot_samples) > 1:
        for lot_name, samples in per_lot_samples.items():
            evaluate_samples(
                samples=samples,
                bundle=bundle,
                threshold_override=args.threshold,
                sweep=args.sweep,
                list_mis=args.list_mis,
                max_mis=args.max_mis,
                title=f"LOT: {lot_name}",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
