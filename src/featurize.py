"""
featurize.py — NOOTZSCOPE

Centralized, bug-proof text extraction + normalization for email files.

Design goals:
- Robust decoding (utf-8-sig -> latin-1 -> bytes fallback)
- Deterministic normalization (whitespace, casing)
- Optional HTML stripping (cheap regex-based)
- Safe handling of empty/short files

Used by:
- tools/train_model.py
- tools/evaluate_local.py
- src/main.py (optional, if you want to share code)

NOTE: Keep this module "generic text processing" (not an anti-spam library).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


# -------------------------
# Email reading
# -------------------------

def read_email_file(file_path: Path) -> Tuple[str, str]:
    """
    Reads an email file where:
      - first line is subject
      - rest is body

    Supports text/html, unknown encodings.
    Returns (subject, body), both strings (possibly empty).
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


# -------------------------
# Normalization
# -------------------------

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html_tags(text: str) -> str:
    """
    Removes basic HTML tags using a regex.
    Not perfect parsing, but robust and dependency-free.
    """
    return HTML_TAG_RE.sub(" ", text)


def normalize_text(
    subject: str,
    body: str,
    *,
    lowercase: bool = True,
    strip_html: bool = False,
    keep_newlines: bool = False,
) -> str:
    """
    Normalizes subject+body into a single text string.

    - Replaces NBSP with space
    - Optionally strips HTML tags
    - Normalizes whitespace to single spaces (or keeps newlines if requested)
    - Optionally lowercases

    Returns normalized text (possibly empty).
    """
    text = (subject + "\n" + body).replace("\u00a0", " ").strip()

    if strip_html:
        text = strip_html_tags(text)

    if keep_newlines:
        # Normalize spaces/tabs etc, but preserve newlines as separators
        parts = []
        for line in text.split("\n"):
            line = WHITESPACE_RE.sub(" ", line).strip()
            if line:
                parts.append(line)
        text = "\n".join(parts)
    else:
        text = WHITESPACE_RE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


# -------------------------
# Dataset helpers
# -------------------------

@dataclass(frozen=True)
class LabeledEmail:
    """
    Represents a single training/eval item from the lots.
    label: 0 = clean, 1 = spam
    """
    path: Path
    label: int
    text: str


def iter_files_non_recursive(folder: Path) -> Iterable[Path]:
    """
    Deterministic listing of files in a folder (non-recursive).
    """
    if not folder.exists() or not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.name.lower())
    return files


def load_lot(
    lot_folder: Path,
    *,
    strip_html: bool = False,
    verbose: bool = False,
) -> List[LabeledEmail]:
    """
    Loads a SSOSM lot folder:
      lot_folder/
        Clean/
        Spam/

    Returns a list of LabeledEmail, deterministic order:
    - All Clean (sorted by filename), then all Spam (sorted).
    """
    clean_dir = lot_folder / "Clean"
    spam_dir = lot_folder / "Spam"
    if not clean_dir.is_dir() or not spam_dir.is_dir():
        raise ValueError(f"Lot folder must contain Clean/ and Spam/: {lot_folder}")

    items: List[LabeledEmail] = []

    def add_dir(d: Path, label: int, label_name: str) -> None:
        for p in iter_files_non_recursive(d):
            if verbose:
                print(f"[DATA] {label_name:<5} {p.name}")
            subj, body = read_email_file(p)
            text = normalize_text(subj, body, strip_html=strip_html)
            items.append(LabeledEmail(path=p, label=label, text=text))

    if verbose:
        print(f"\n[LOT] Loading data from: {lot_folder}")

    add_dir(clean_dir, 0, "CLEAN")
    add_dir(spam_dir, 1, "SPAM")

    if verbose:
        print(f"[LOT] Loaded {len(items)} samples from {lot_folder}\n")

    return items


def combine_lots(
    lot_folders: List[Path],
    *,
    strip_html: bool = False,
    verbose: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Convenience: loads multiple lots and returns (X_texts, y_labels).
    """
    X: List[str] = []
    y: List[int] = []
    for lf in lot_folders:
        items = load_lot(lf, strip_html=strip_html, verbose=verbose)
        for it in items:
            X.append(it.text)
            y.append(it.label)
    return X, y
