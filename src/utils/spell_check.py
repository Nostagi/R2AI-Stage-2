from __future__ import annotations

import re
from typing import List


# ──────────────────────────────────────────────────────────
# Pyvi - Word Segmentation
# ──────────────────────────────────────────────────────────

from pyvi import ViTokenizer
import unicodedata

_WS_RE = re.compile(r"\s+")

def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalize_text(text: str, *, lower: bool = True) -> str:
    """Chuan hoa co ban: NFC -> lowercase -> gom whitespace."""
    text = unicodedata.normalize("NFC", text)
    if lower:
        text = text.lower()
    return normalize_ws(text)

_NON_WORD_RE = re.compile(r"[^0-9a-z_]+")

def tokenize(text: str) -> list[str]:
    """Tokenizer tinh gon cho BM25 su dung Pyvi.
    Khong cat dau, khong dung stopwords de BM25 tu dong can bang trong so bang IDF.
    """
    # 1. Tokenize bang pyvi (giu nguyen text goc de nhan dien tu ghep)
    tokenized_text = ViTokenizer.tokenize(text)
    
    # 2. Chuan hoa (ha chu thuong, gom khoang trang) 
    flat = normalize_text(tokenized_text)
    
    # 3. Tach token (giu lai dau '_' cua pyvi)
    return [t for t in _NON_WORD_RE.split(flat) if t]


# ──────────────────────────────────────────────────────────
# SymSpell Utility Functions
# ──────────────────────────────────────────────────────────

_VN_LOWER = "a-zàáảãạâấầẩẫậăắằẳẵặèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
_VN_UPPER = "A-ZÀÁẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ"
_VN_ALL = _VN_LOWER + _VN_UPPER

# Symspell Patterns
_LOWER_UPPER_RE = re.compile(rf"([{_VN_LOWER}])([{_VN_UPPER}])")
_LETTER_NUM_RE = re.compile(rf"([{_VN_ALL}])(\d)")
_NUM_LETTER_RE = re.compile(rf"(\d)([{_VN_ALL}])")

# Numeric Patterns
_VN_NUMBER_RE = re.compile(r"\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?|-")
_NUMERIC_CELL_RE = re.compile(r"^[\d\s.,()\-]+$")
_NUMERIC_SHAPE_RE = re.compile(r"^[\(\-]?[\d\s.,OoIlSsBZzQgD]+[\)%]?$")
_HAS_DIGIT_RE = re.compile(r"\d")

_LOOKALIKE_DIGITS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "i": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9",
})

def symspell(text: str) -> str:
    """Xử lý ranh giới dính chữ phi tự nhiên: thường-HOA, chữ-số, và số-chữ."""
    text = _LOWER_UPPER_RE.sub(r"\1 \2", text)
    text = _LETTER_NUM_RE.sub(r"\1 \2", text)
    text = _NUM_LETTER_RE.sub(r"\1 \2", text)
    return text

def is_numeric_like(token: str) -> bool:
    token = token.strip()
    return bool(token and _NUMERIC_SHAPE_RE.match(token) and _HAS_DIGIT_RE.search(token))

def numeric_translate(token: str) -> str:
    """Tích hợp kiểm tra is_numeric_like và tự động sửa ký tự quang học nếu thỏa mãn."""

    if is_numeric_like(token):
        return token.translate(_LOOKALIKE_DIGITS)
    else:
        return token


def split_glued_numbers(text: str) -> List[str]:
    """Tách các chuỗi số bị dính liền dọc do lỗi rowspan."""
    stripped = text.strip()
    if not stripped or not _NUMERIC_CELL_RE.match(stripped):
        return [stripped] if stripped else []
    
    parts = []
    pos = 0
    for m in _VN_NUMBER_RE.finditer(stripped):
        if m.start() != pos:
            return [stripped]
        parts.append(m.group())
        pos = m.end()
        
    if pos != len(stripped) or len(parts) < 2:
        return [stripped]
    return parts