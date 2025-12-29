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
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

import joblib


PROJECT_INFO = {
    "student_name": "Danut IURASCU",   # edit if needed
    "project_name": "NootzScope",
    "student_alias": "thenootz",
    "project_version": "1.0.0",
}


# -------------------------
# Strict IO helpers
# -------------------------

def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", errors="strict")
    tmp.replace(path)


def iter_files_non_recursive(folder: Path) -> Iterable[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.name.lower())
    return files


def read_email_file(file_path: Path) -> Tuple[str, str]:
    """
    First line = subject; remaining = body. Robust decoding.
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
# ML bundle loading + predict
# -------------------------

def load_bundle() -> dict | None:
    """
    Loads model.joblib from same directory as this main.py.
    Returns None if missing/corrupt (so we can fallback without crashing).
    """
    here = Path(__file__).resolve().parent
    model_path = here / "model.joblib"
    if not model_path.exists():
        return None
    try:
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict):
            return None
        if "pipeline" not in bundle:
            return None
        return bundle
    except Exception:
        return None


def heuristic_fallback(subject: str, body: str) -> bool:
    """
    Safe fallback if model can't be loaded.
    Returns True if spam.
    """
    text = (subject + "\n" + body).lower()
    spam_words = ["viagra", "winner", "prize", "click", "act now", "limited time", "unsubscribe", "casino", "lottery"]
    score = sum(1 for w in spam_words if w in text)
    score += 1 if ("http://" in text or "https://" in text or "www." in text) else 0
    return score >= 2


def is_spam(subject: str, body: str, bundle: dict | None) -> bool:
    if bundle is None:
        return heuristic_fallback(subject, body)

    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", 0.5))
    strip_html = bool(bundle.get("strip_html", False))

    x = [normalize_text(subject, body, strip_html=strip_html)]
    # LogisticRegression supports predict_proba; if changed later, handle decision_function too.
    if hasattr(pipeline, "predict_proba"):
        p_spam = float(pipeline.predict_proba(x)[0][1])
        return p_spam >= threshold
    if hasattr(pipeline, "decision_function"):
        # Map margin to boolean using 0.0 (common for linear models)
        margin = float(pipeline.decision_function(x)[0])
        return margin >= 0.0
    # Worst-case fallback
    pred = int(pipeline.predict(x)[0])
    return pred == 1


# -------------------------
# CLI
# -------------------------

def cmd_info(output_file: Path) -> int:
    safe_write_text(output_file, json.dumps(PROJECT_INFO, indent=4) + "\n")
    return 0


def cmd_scan(folder: Path, output_file: Path) -> int:
    bundle = load_bundle()

    lines = []
    for file_path in iter_files_non_recursive(folder):
        try:
            subject, body = read_email_file(file_path)
            verdict = "inf" if is_spam(subject, body, bundle) else "cln"
        except Exception:
            verdict = "inf"  # never crash scanning

        # strict: base filename only, no spaces
        lines.append(f"{file_path.name}|{verdict}")

    safe_write_text(output_file, "\n".join(lines) + ("\n" if lines else ""))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("-info", nargs=1, metavar=("output_file"), help="Write project info JSON to output_file")
    p.add_argument("-scan", nargs=2, metavar=("folder", "output_file"), help="Scan folder and write verdicts")
    return p


def main(argv: list[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    info_arg = args.info
    scan_arg = args.scan

    if (info_arg is None and scan_arg is None) or (info_arg is not None and scan_arg is not None):
        parser.print_usage(sys.stderr)
        return 2

    if info_arg is not None:
        return cmd_info(Path(info_arg[0]))

    return cmd_scan(Path(scan_arg[0]), Path(scan_arg[1]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
