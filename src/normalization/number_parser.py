"""Parse so kieu Viet Nam trong BCTC.

Cac dang phai xu ly:
    "1.234.567"      -> 1234567.0      (dau . la phan cach nghin)
    "1.234.567,89"   -> 1234567.89     (dau , la thap phan)
    "(1.234)"        -> -1234.0        (ngoac = so am, quy uoc ke toan)
    "-"  / ""        -> None           (o trong)
    "12,5%"          -> 0.125          (neu as_ratio=True)
    "1,234,567.89"   -> 1234567.89     (dang Anh — mot so bao cao dung)

Nham dau phan cach la nguon sai so lon nhat -> dung heuristic ro rang
va co test bao phu (tests/test_number_parser.py).
"""

from __future__ import annotations

import re

from ..extraction.ocr_fixer import fix_numeric_token, is_numeric_like

# He so quy doi ve VND
UNIT_SCALES: dict[str, float] = {
    "vnd": 1.0,
    "dong": 1.0,
    "nghin": 1e3,
    "nghin dong": 1e3,
    "ngan dong": 1e3,
    "trieu": 1e6,
    "trieu dong": 1e6,
    "ty": 1e9,
    "ty dong": 1e9,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
}

_UNIT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bty\s*(?:dong|vnd)\b"), "ty dong"),
    (re.compile(r"\btrieu\s*(?:dong|vnd)\b"), "trieu dong"),
    (re.compile(r"\b(?:nghin|ngan)\s*(?:dong|vnd)\b"), "nghin dong"),
    (re.compile(r"\bbillion\b"), "billion"),
    (re.compile(r"\bmillion\b"), "million"),
    (re.compile(r"\bthousand\b"), "thousand"),
    (re.compile(r"\b(?:vnd|dong)\b"), "vnd"),
)

_NULL_TOKENS = frozenset({"", "-", "--", "n/a", "na", "none", "null", "..", "..."})
_STRIP_CHARS = " \t '\"*"
_NEG_PAREN_RE = re.compile(r"^\(\s*(.+?)\s*\)$")
_PERCENT_RE = re.compile(r"%\s*$")
_ALLOWED_RE = re.compile(r"^[+-]?[\d.,\s]+$")


def parse_vn_number(
    raw: object,
    *,
    fix_ocr: bool = True,
    as_ratio: bool = False,
    scale: float = 1.0,
) -> float | None:
    """Chuyen mot o bang thanh float, hoac None neu khong phai so.

    Args:
        raw: gia tri o (thuong la str tu OCR).
        fix_ocr: cho phep sua O->0, l->1 khi token co hinh dang so.
        as_ratio: "12,5%" -> 0.125 thay vi 12.5.
        scale: he so nhan (vd 1e6 khi bang ghi don vi "trieu dong").
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw) * scale

    text = str(raw).strip(_STRIP_CHARS)
    if text.lower() in _NULL_TOKENS:
        return None

    if fix_ocr and is_numeric_like(text):
        text = fix_numeric_token(text)

    negative = False
    m = _NEG_PAREN_RE.match(text)
    if m:
        negative = True
        text = m.group(1).strip()

    is_percent = bool(_PERCENT_RE.search(text))
    if is_percent:
        text = _PERCENT_RE.sub("", text).strip()

    if text.startswith("-"):
        negative = not negative
        text = text[1:].strip()
    elif text.startswith("+"):
        text = text[1:].strip()

    text = text.replace(" ", "").replace(" ", "")
    if not text or not _ALLOWED_RE.match(text):
        return None

    numeric = _to_float(text)
    if numeric is None:
        return None

    if negative:
        numeric = -numeric
    if is_percent and as_ratio:
        numeric /= 100.0

    return numeric * scale


def _to_float(text: str) -> float | None:
    """Quyet dinh dau nao la thap phan, dau nao la phan cach nghin."""
    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        # Dau xuat hien SAU cung la dau thap phan
        dec_sep = "." if text.rfind(".") > text.rfind(",") else ","
        thou_sep = "," if dec_sep == "." else "."
        text = text.replace(thou_sep, "").replace(dec_sep, ".")
    elif has_comma:
        text = _single_sep_to_float(text, ",")
    elif has_dot:
        text = _single_sep_to_float(text, ".")

    try:
        return float(text)
    except ValueError:
        return None


def _single_sep_to_float(text: str, sep: str) -> str:
    """Chi co 1 loai dau — phan biet thap phan vs phan cach nghin.

    Quy tac: nhieu lan xuat hien, HOAC dung 3 chu so sau dau -> phan cach
    nghin. Nguoc lai -> thap phan. "1.234" trong BCTC gan nhu chac chan la
    1234 (nghin), khong phai 1.234.
    """
    parts = text.split(sep)
    if len(parts) > 2:
        return "".join(parts)
    tail = parts[-1]
    if len(tail) == 3 and parts[0]:
        return "".join(parts)          # phan cach nghin
    return ".".join(parts)             # thap phan


def detect_unit(context: str) -> tuple[str | None, float]:
    """Tim don vi tien te trong caption/context bang.

    Returns: (ten don vi da chuan hoa, he so ve VND). Khong thay -> (None, 1.0).
    """
    from ..utils.spell_check import normalize_text

    flat = normalize_text(context)
    for pattern, label in _UNIT_PATTERNS:
        if pattern.search(flat):
            return label, UNIT_SCALES[label]
    return None, 1.0
