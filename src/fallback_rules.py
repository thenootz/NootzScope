"""
fallback_rules.py — NOOTZSCOPE

A conservative heuristic fallback used only if the ML model cannot be loaded.
Goal: never crash, keep reasonable behavior.

Usage:
  from fallback_rules import fallback_is_spam
  verdict = "inf" if fallback_is_spam(subject, body) else "cln"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


# -------------------------
# Lightweight normalization
# -------------------------

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
MONEY_RE = re.compile(r"\b(?:€|\$|£)\s?\d+|\b\d+\s?(?:eur|usd|gbp)\b", re.IGNORECASE)
SHOUT_RE = re.compile(r"\b[A-Z]{8,}\b")


def normalize(subject: str, body: str) -> str:
    txt = (subject + "\n" + body).replace("\u00a0", " ")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    # Keep some structure but avoid huge whitespace variance
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def strip_html(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


# -------------------------
# Heuristic features
# -------------------------

SPAM_KEYWORDS = {
    # classic
    "free", "winner", "won", "win", "prize", "urgent", "act now", "limited time",
    "congratulations", "guarantee", "risk free", "exclusive deal",
    # finance / scam-ish
    "investment", "bitcoin", "crypto", "wire transfer", "bank transfer", "loan", "credit",
    # pharma
    "viagra", "cialis", "levitra",
    # engagement
    "click", "click here", "open now", "unsubscribe",
    # gambling
    "casino", "lottery",
}

# Obfuscations / leetspeak-like patterns
OBFUSCATED_RE = re.compile(
    r"\b(v[i1!|]agra|c[i1!|]al[i1!|]s|fr[e3]{2}|cl[i1!|]ck|w[i1!|]n|pr[i1!|]ze)\b",
    re.IGNORECASE,
)

# A few "clean-ish" hints to reduce false positives
CLEAN_HINTS = {
    "meeting", "minutes", "invoice", "schedule", "report", "project", "deadline",
    "regards", "thank you", "attached", "attachment", "please find",
}


@dataclass
class RuleScore:
    score: float
    reasons: Tuple[str, ...]


def score_email(subject: str, body: str) -> RuleScore:
    """
    Returns a score and a list of short reasons (useful for debugging).
    """
    reasons = []
    text_raw = normalize(subject, body)
    text = text_raw.lower()

    score = 0.0

    # Keyword matches (strong indicator)
    hits = 0
    for kw in SPAM_KEYWORDS:
        if kw in text:
            hits += 1
    if hits:
        score += min(4.0, 1.2 * hits)
        reasons.append(f"kw:{hits}")

    # Obfuscations
    if OBFUSCATED_RE.search(text):
        score += 2.0
        reasons.append("obf")

    # URLs
    url_hits = len(URL_RE.findall(text_raw))
    if url_hits:
        score += min(3.0, 0.9 * url_hits)
        reasons.append(f"url:{url_hits}")

    # Money
    if MONEY_RE.search(text_raw):
        score += 1.3
        reasons.append("money")

    # Shouting
    if SHOUT_RE.search(text_raw):
        score += 0.8
        reasons.append("caps")

    # Many HTML tags suggests marketing/spammy templates
    tag_count = len(HTML_TAG_RE.findall(text_raw))
    if tag_count >= 25:
        score += 1.2
        reasons.append("html:high")
    elif tag_count >= 8:
        score += 0.6
        reasons.append("html:mid")

    # Excess punctuation
    if text_raw.count("!") >= 5:
        score += 0.7
        reasons.append("excl")

    # Reduce score if it strongly resembles normal work mail
    clean_hits = 0
    for ck in CLEAN_HINTS:
        if ck in text:
            clean_hits += 1
    if clean_hits:
        score -= min(2.4, 0.6 * clean_hits)
        reasons.append(f"clean:{clean_hits}")

    # Email addresses are common in both, but many can indicate phishing lists
    email_hits = len(EMAIL_RE.findall(text_raw))
    if email_hits >= 5:
        score += 0.6
        reasons.append(f"emails:{email_hits}")

    return RuleScore(score=score, reasons=tuple(reasons))


def fallback_is_spam(subject: str, body: str) -> bool:
    """
    Conservative default threshold.
    Adjust only if you see many false positives/negatives in local tests.
    """
    rs = score_email(subject, body)
    return rs.score >= 2.2


def fallback_debug(subject: str, body: str) -> str:
    """
    Optional helper: returns a one-line explanation.
    Not used in submission output (keep scan output strict).
    """
    rs = score_email(subject, body)
    return f"score={rs.score:.2f} reasons={','.join(rs.reasons) if rs.reasons else 'none'}"
