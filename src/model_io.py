"""
model_io.py — NOOTZSCOPE

Safe loading/saving for the ML model bundle used by main.py.

Bundle format (dict):
{
  "pipeline": <sklearn Pipeline>,
  "threshold": float,      # probability threshold for spam
  "strip_html": bool,      # whether text normalization strips HTML tags
  "version": "1.0",
  ... optional metadata ...
}

Design goals:
- Never crash the scanner if the model is missing/corrupt
- Provide clear validation and helpful errors for training tools
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Any
    threshold: float
    strip_html: bool
    version: str = "1.0"
    meta: Dict[str, Any] | None = None


def validate_bundle_dict(d: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "bundle is not a dict"
    if "pipeline" not in d:
        return False, "missing key: pipeline"
    thr = d.get("threshold", 0.5)
    try:
        thr_f = float(thr)
    except Exception:
        return False, "threshold is not a float"
    if not (0.0 <= thr_f <= 1.0):
        # If using margins instead of probabilities, threshold might not be in [0,1],
        # but our project primarily uses predict_proba; keep strict for safety.
        # You can relax this if you switch to decision_function models.
        return False, "threshold out of range [0,1]"
    strip_html = d.get("strip_html", False)
    if not isinstance(strip_html, bool):
        return False, "strip_html is not bool"
    ver = d.get("version", "1.0")
    if not isinstance(ver, str):
        return False, "version is not str"
    return True, "ok"


def dict_to_bundle(d: Dict[str, Any]) -> ModelBundle:
    ok, msg = validate_bundle_dict(d)
    if not ok:
        raise ValueError(f"Invalid model bundle: {msg}")
    return ModelBundle(
        pipeline=d["pipeline"],
        threshold=float(d.get("threshold", 0.5)),
        strip_html=bool(d.get("strip_html", False)),
        version=str(d.get("version", "1.0")),
        meta={k: v for k, v in d.items() if k not in {"pipeline", "threshold", "strip_html", "version"}},
    )


def bundle_to_dict(b: ModelBundle) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "pipeline": b.pipeline,
        "threshold": float(b.threshold),
        "strip_html": bool(b.strip_html),
        "version": str(b.version),
    }
    if b.meta:
        d.update(b.meta)
    return d


def save_bundle(path: Path, bundle: ModelBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle_to_dict(bundle), path, compress=3)


def load_bundle_strict(path: Path) -> ModelBundle:
    """
    Loads and validates a model bundle. Raises on error.
    Useful for training/evaluation tools.
    """
    d = joblib.load(path)
    if not isinstance(d, dict):
        raise ValueError("Loaded object is not a dict")
    return dict_to_bundle(d)


def load_bundle_safe(path: Path) -> Optional[ModelBundle]:
    """
    Loads a model bundle safely. Returns None if missing/corrupt.
    Useful for main.py to avoid crashing the scan command.
    """
    try:
        if not path.exists() or not path.is_file():
            return None
        d = joblib.load(path)
        if not isinstance(d, dict):
            return None
        ok, _ = validate_bundle_dict(d)
        if not ok:
            return None
        return dict_to_bundle(d)
    except Exception:
        return None


def default_model_path_near(file_path: Path, model_filename: str = "model.joblib") -> Path:
    """
    Given a reference file (e.g., __file__ from main.py),
    returns the model path in the same directory.
    """
    return file_path.resolve().parent / model_filename
