#!/usr/bin/env python3
"""
Submission ZIP builder for SSOSM.

Creates:
  project.zip/<LASTNAME_FIRSTNAME>/main.py

Optionally includes:
  - model file (e.g. model.joblib)
  - helper python files from src/
  - requirements.txt

Examples:
  python tools/make_submission_zip.py --student-folder IURASCU_Danut \
    --main src/main.py --model models/model.joblib \
    --include-src-helpers --include-requirements \
    --out submissions/project.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List


FOLDER_NAME_RE = re.compile(r"^[A-Za-z]+_[A-Za-z]+$")
DEFAULT_HELPERS = ["featurize.py", "model_io.py", "fallback_rules.py"]


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def add_file_to_zip(zf: zipfile.ZipFile, src: Path, arc_path: str) -> None:
    if not src.exists() or not src.is_file():
        die(f"Missing file: {src}")
    arc_path = arc_path.replace("\\", "/")
    zf.write(src, arcname=arc_path)
    ok(f"Added: {src} -> {arc_path}")


def collect_src_helpers(src_dir: Path, include_all: bool) -> List[Path]:
    if not src_dir.exists() or not src_dir.is_dir():
        die(f"src directory not found: {src_dir}")

    if include_all:
        return sorted(
            [p for p in src_dir.iterdir() if p.is_file() and p.suffix == ".py"],
            key=lambda p: p.name.lower(),
        )

    helpers: List[Path] = []
    for name in DEFAULT_HELPERS:
        p = src_dir / name
        if not p.exists():
            die(f"Expected helper not found in src/: {name}")
        helpers.append(p)
    return helpers


def build_zip(
    out_zip: Path,
    student_folder: str,
    main_path: Path,
    model_path: Path | None,
    extra_files: Iterable[Path],
    src_dir: Path | None,
    include_src_helpers: bool,
    include_src_all: bool,
    include_requirements: bool,
    requirements_path: Path | None,
) -> None:
    if not FOLDER_NAME_RE.match(student_folder):
        die(f'Invalid --student-folder "{student_folder}". Expected LASTNAME_FIRSTNAME.')

    if out_zip.suffix.lower() != ".zip":
        die("Output file must be .zip")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        add_file_to_zip(zf, main_path, f"{student_folder}/main.py")

        if model_path:
            if model_path.exists():
                add_file_to_zip(zf, model_path, f"{student_folder}/{model_path.name}")
            else:
                warn(f"Model file not found, skipping: {model_path}")

        if include_src_helpers:
            if src_dir is None:
                die("--include-src-helpers requires --src-dir")
            helpers = collect_src_helpers(src_dir, include_all=include_src_all)
            for hp in helpers:
                if hp.resolve() == main_path.resolve():
                    continue
                add_file_to_zip(zf, hp, f"{student_folder}/{hp.name}")

        if include_requirements:
            if not requirements_path or not requirements_path.exists():
                die("requirements.txt not found but --include-requirements was specified")
            add_file_to_zip(zf, requirements_path, f"{student_folder}/requirements.txt")

        for ef in extra_files:
            if ef.exists():
                add_file_to_zip(zf, ef, f"{student_folder}/{ef.name}")
            else:
                warn(f"Extra file not found, skipping: {ef}")

    ok(f"Created submission zip: {out_zip}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SSOSM submission ZIP")

    p.add_argument("--student-folder", required=True)
    p.add_argument("--main", required=True)
    p.add_argument("--model")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--out", default="project.zip")

    p.add_argument("--include-src-helpers", action="store_true")
    p.add_argument("--include-src-all", action="store_true")
    p.add_argument("--src-dir", default="src")

    p.add_argument("--include-requirements", action="store_true")
    p.add_argument("--requirements", default="requirements.txt")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    def resolve(p: str) -> Path:
        return (repo_root / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()

    build_zip(
        out_zip=resolve(args.out),
        student_folder=args.student_folder,
        main_path=resolve(args.main),
        model_path=resolve(args.model) if args.model else None,
        extra_files=[resolve(x) for x in args.extra],
        src_dir=resolve(args.src_dir),
        include_src_helpers=args.include_src_helpers,
        include_src_all=args.include_src_all,
        include_requirements=args.include_requirements,
        requirements_path=resolve(args.requirements),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
