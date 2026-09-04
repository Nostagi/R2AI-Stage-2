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
_VN_MONEY_RE = re.compile(r"\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?|-")
_NUMERIC_CELL_RE = re.compile(r"^[\d\s.,()\-]+$")
_NUMERIC_SHAPE_RE = re.compile(r"^[\(\-]?[\d\s.,OoIlSsBZzQgD]+[\)%]?$")
_HAS_DIGIT_RE = re.compile(r"\d")

_LOOKALIKE_DIGITS = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "i": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "g": "9",
})

def parse_string_to_number(num_str: str):
    # Remove everything except digits, dots, and commas
    num_str = re.sub(r'[^\d.,\-]', '', num_str.strip())
    if num_str == "-":
        return num_str
    
    # Rule 1: Find the rightmost separator and check trailing digits
    last_sep_match = re.search(r'([.,])(\d+)$', num_str)
    
    if last_sep_match:
        sep, trailing_digits = last_sep_match.groups()
        
        # If followed by less than 3 digits, it's definitely the decimal separator
        if len(trailing_digits) < 3:
            thousands_sep = ',' if sep == '.' else '.'
            num_str = num_str.replace(thousands_sep, '').replace(sep, '.')
        else:
            # Rule 2 Fallback: If both exist, the first one from left is thousands
            has_comma = ',' in num_str
            has_dot = '.' in num_str
            if has_comma and has_dot:
                first_sep = ',' if num_str.find(',') < num_str.find('.') else '.'
                decimal_sep = '.' if first_sep == ',' else ','
                num_str = num_str.replace(first_sep, '').replace(decimal_sep, '.')
            else:
                # We assume it is a thousands separator for a whole number
                num_str = num_str.replace(',', '').replace('.', '')

    # Convert to numeric type (int or float)
    return float(num_str) if num_str != "" else "-"

def _split_glued_money_numbers(text: str) -> List:
    stripped = text.strip()

    if not stripped or not _NUMERIC_CELL_RE.fullmatch(stripped):
        return [stripped] if stripped else []

    parts = []
    pos = 0

    for m in _VN_MONEY_RE.finditer(stripped):
        # Allow whitespace between numbers.
        if stripped[pos:m.start()].strip():
            return [stripped]

        parts.append(m.group().strip())
        pos = m.end()

    if stripped[pos:].strip():
        return [stripped]

    return [parse_string_to_number(p) for p in parts]

def symspell(text: str) -> str:
    """Xử lý ranh giới dính chữ phi tự nhiên: thường-HOA, chữ-số, và số-chữ."""
    text = _LOWER_UPPER_RE.sub(r"\1 \2", text)
    text = _LETTER_NUM_RE.sub(r"\1 \2", text)
    text = _NUM_LETTER_RE.sub(r"\1 \2", text)
    return text

def is_empty(value: str) -> bool:
    if bool(re.search(rf"[{_VN_ALL}]", value)) or bool(re.search(_HAS_DIGIT_RE, value)):
        return False
    return True

def is_numeric_like(token: str) -> bool:
    token = token.strip()
    return bool(token and _NUMERIC_SHAPE_RE.match(token) and _HAS_DIGIT_RE.search(token))

def numeric_translate(token: str) -> List[str]:
    """
    Tích hợp kiểm tra is_numeric_like và tự động sửa ký tự quang học nếu thỏa mãn.
    Nếu là định dạng về tiền tệ, sẽ tách các số dính liền ra thành nhiều token.
    """

    if is_numeric_like(token):
        token = token.translate(_LOOKALIKE_DIGITS).strip()
        token = re.sub(r'^[([{]|[)]}]$', '', token).strip()

        numbers = _split_glued_money_numbers(token)
        return numbers
            
    return [token]