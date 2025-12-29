#!/usr/bin/env python3
#  _   _             _        _____                      
# | \ | |           | |      / ____|                     
# |  \| | ___   ___ | |_ ___| (___   ___ ___  _ __   ___ 
# | . ` |/ _ \ / _ \| __|_  /\___ \ / __/ _ \| '_ \ / _ \
# | |\  | (_) | (_) | |_ / / ____) | (_| (_) | |_) |  __/
# |_| \_|\___/ \___/ \__/___|_____/ \___\___/| .__/ \___|
#                                            | |         
#                                            |_|         
#  NootzScope — ML Spam Detection (SSOSM) · author: thenootz

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from featurize import iter_files_non_recursive, read_email_file, normalize_text
from fallback_rules import fallback_is_spam
from model_io import default_model_path_near, load_bundle_safe


PROJECT_INFO = {
    "student_name": "Danut IURASCU",
    "project_name": "NootzScope",
    "student_alias": "thenootz",
    "project_version": "1.0.0",
}


# -------------------------
# Strict IO helpers
# -------------------------

def safe_write_text(path: Path, content: str) -> None:
    """
    Atomically writes UTF-8 text. Avoids partially written outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", errors="strict")
    tmp.replace(path)


# -------------------------
# Prediction
# -------------------------

def predict_is_spam(subject: str, body: str, bundle) -> bool:
    """
    Uses ML bundle if available, otherwise heuristic fallback.
    """
    if bundle is None:
        return fallback_is_spam(subject, body)

    pipeline = bundle.pipeline
    threshold = float(bundle.threshold)
    strip_html = bool(bundle.strip_html)

    x = [normalize_text(subject, body, strip_html=strip_html)]

    # Prefer probas
    if hasattr(pipeline, "predict_proba"):
        p_spam = float(pipeline.predict_proba(x)[0][1])
        return p_spam >= threshold

    # Margin models (e.g. LinearSVC)
    if hasattr(pipeline, "decision_function"):
        margin = float(pipeline.decision_function(x)[0])
        # If your model uses margins, train_model should store a margin threshold in bundle.threshold
        return margin >= threshold

    # Last resort: hard predict
    pred = int(pipeline.predict(x)[0])
    return pred == 1


# -------------------------
# CLI commands
# -------------------------

def cmd_info(output_file: Path) -> int:
    safe_write_text(output_file, json.dumps(PROJECT_INFO, indent=4) + "\n")
    return 0


def cmd_scan(folder: Path, output_file: Path) -> int:
    # Load model next to this main.py in the submission folder
    model_path = default_model_path_near(Path(__file__), "model.joblib")
    bundle = load_bundle_safe(model_path)

    lines = []
    for file_path in iter_files_non_recursive(folder):
        try:
            subject, body = read_email_file(file_path)
            verdict = "inf" if predict_is_spam(subject, body, bundle) else "cln"
        except Exception:
            # Bug-proof: never crash; conservative default
            verdict = "inf"

        # STRICT format: base filename only, no spaces
        lines.append(f"{file_path.name}|{verdict}")

    safe_write_text(output_file, "\n".join(lines) + ("\n" if lines else ""))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("-info", nargs=1, metavar=("output_file"),
                   help="Write project info JSON to output_file")
    p.add_argument("-scan", nargs=2, metavar=("folder", "output_file"),
                   help="Scan folder and write verdicts (non-recursive)")
    return p


def main(argv: list[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    info_arg = args.info
    scan_arg = args.scan

    # Enforce exactly one mode
    if (info_arg is None and scan_arg is None) or (info_arg is not None and scan_arg is not None):
        parser.print_usage(sys.stderr)
        return 2

    if info_arg is not None:
        return cmd_info(Path(info_arg[0]))

    return cmd_scan(Path(scan_arg[0]), Path(scan_arg[1]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
