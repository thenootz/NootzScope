#!/usr/bin/env python3
"""
Submission ZIP builder for SSOSM.

Creates:
  project.zip/<LASTNAME_FIRSTNAME>/main.py
Optional:
  project.zip/<LASTNAME_FIRSTNAME>/model.joblib
  ... other assets

Examples:
  python tools/make_submission_zip.py --student-folder Gavrilut_Dragos \
    --main src/main.py --model models/model.joblib --out submissions/project.zip

  python tools/make_submission_zip.py --student-folder IURASCU_Danut \
    --main src/main.py --out project.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable


FOLDER_NAME_RE = re.compile(r"^[A-Za-z]+_[A-Za-z]+$")  # LASTNAME_FIRSTNAME


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[OK] {msg}")


def add_file_to_zip(zf: zipfile.ZipFile, src: Path, arc_path: str) -> None:
    if not src.exists() or not src.is_file():
        die(f"Missing file: {src}")
    # Ensure forward slashes inside zip
    arc_path = arc_path.replace("\\", "/")
    zf.write(src, arcname=arc_path)
    info(f"Added: {src} -> {arc_path}")


def build_zip(
    out_zip: Path,
    student_folder: str,
    main_path: Path,
    model_path: Path | None,
    extra_files: Iterable[Path],
) -> None:
    if not FOLDER_NAME_RE.match(student_folder):
        die(
            f'Invalid --student-folder "{student_folder}". Expected format LASTNAME_FIRSTNAME (letters only).'
        )

    if out_zip.suffix.lower() != ".zip":
        die(f'Output must be a .zip file, got: "{out_zip}"')

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite safely
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_file_to_zip(zf, main_path, f"{student_folder}/main.py")

        if model_path is not None:
            if model_path.exists():
                add_file_to_zip(zf, model_path, f"{student_folder}/{model_path.name}")
            else:
                warn(f"Model file not found, skipping: {model_path}")

        for ef in extra_files:
            if ef.exists() and ef.is_file():
                add_file_to_zip(zf, ef, f"{student_folder}/{ef.name}")
            else:
                warn(f"Extra file not found, skipping: {ef}")

    # Sanity-check zip contents
    with zipfile.ZipFile(out_zip, mode="r") as zf:
        names = zf.namelist()
        required = f"{student_folder}/main.py"
        if required not in names:
            die(f"ZIP sanity check failed: missing {required}")
        # Ensure nothing accidentally included outside the student folder
        bad = [n for n in names if not n.startswith(f"{student_folder}/")]
        if bad:
            die(f"ZIP sanity check failed: unexpected paths: {bad}")

    info(f"Created submission zip: {out_zip}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SSOSM submission ZIP")
    p.add_argument(
        "--student-folder",
        required=True,
        help='Folder name inside ZIP, e.g. "Gavrilut_Dragos" (LASTNAME_FIRSTNAME).',
    )
    p.add_argument(
        "--main",
        required=True,
        help="Path to main.py to include in the ZIP (will be renamed to main.py inside ZIP).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Optional path to model file (e.g., models/model.joblib). If missing, it will be skipped.",
    )
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Optional extra files to include next to main.py inside the ZIP (e.g., config.json).",
    )
    p.add_argument(
        "--out",
        default="project.zip",
        help='Output zip path (default: "project.zip").',
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    student_folder = args.student_folder
    main_path = Path(args.main).resolve()
    model_path = Path(args.model).resolve() if args.model else None
    extra_files = [Path(x).resolve() for x in args.extra]
    out_zip = Path(args.out).resolve()

    build_zip(
        out_zip=out_zip,
        student_folder=student_folder,
        main_path=main_path,
        model_path=model_path,
        extra_files=extra_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
