#!/usr/bin/env python3
"""
Prefix files in a SSOSM lot with their class name.

Example:
  Clean/email01.txt  -> Clean_email01.txt
  Spam/offer.html    -> Spam_offer.html

Usage:
  python tools/prefix_lot_files.py --lot data/Lot1
  python tools/prefix_lot_files.py --lot data/Lot1 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CLASSES = ["Clean", "Spam"]


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefix SSOSM lot filenames with class name")
    p.add_argument("--lot", required=True, help="Path to lot folder (contains Clean/ and Spam/)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be renamed without changing files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lot_path = Path(args.lot).resolve()

    if not lot_path.exists() or not lot_path.is_dir():
        die(f"Lot folder not found: {lot_path}")

    for cls in CLASSES:
        cls_dir = lot_path / cls
        if not cls_dir.exists() or not cls_dir.is_dir():
            die(f"Missing expected folder: {cls_dir}")

        print(f"\n[{cls}] Processing {cls_dir}")

        files = sorted([p for p in cls_dir.iterdir() if p.is_file()],
                       key=lambda p: p.name.lower())

        for p in files:
            if p.name.startswith(f"{cls}_"):
                print(f"  SKIP  {p.name} (already prefixed)")
                continue

            new_name = f"{cls}_{p.name}"
            new_path = p.with_name(new_name)

            if new_path.exists():
                print(f"  SKIP  {p.name} -> {new_name} (target exists)")
                continue

            if args.dry_run:
                print(f"  DRY   {p.name} -> {new_name}")
            else:
                p.rename(new_path)
                print(f"  OK    {p.name} -> {new_name}")

    print("\n[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

