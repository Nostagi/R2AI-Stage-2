"""Don vi cua DAP AN — suy tu cau hoi, ap o cuoi pipeline.

TAI SAO CAN (do tren 1012 cau hoi that):
    391 cau hoi "ty dong", 213 cau "trieu dong", 16 cau "nghin dong"
    -> 620/1012 = 61.3% cau hoi can chia lai.

Pipeline chuan hoa MOI gia tri ve VND o `schema_std.py` (input-side, giu
nguyen — do la representation noi bo dung). Nhung cau hoi lai hoi bang
don vi khac. Khong co buoc chuyen nguoc thi dap an lech 1e3..1e9 lan du
retrieval va pandas deu dung.

THIET KE: deterministic, o BOUNDARY, khong nho LLM.
    raw_result (VND) -> AnswerNormalizer.to_asked_unit() -> final answer

KHONG nhung he so chia vao code pandas sinh ra: LLM hay quen, hay chia
hai lan, va khi sua loi (self-repair) thi he so bien mat. Chia mot lan o
`SubmissionBuilder` la diem duy nhat khong the truot.

PHAN BIET `%` VA `lan` — hai semantic KHAC NHAU:
    "ty suat loi nhuan ... bao nhieu %"   -> ratio 0.153 -> 15.3
    "he so thanh toan ... bao nhieu lan"  -> ratio 1.53  -> 1.53  (GIU NGUYEN)
Gop chung se lam sai ca 115 cau `%` lan 51 cau `lan`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..utils.spell_check import normalize_text


class AskedUnit(str, Enum):
    """Don vi ma cau hoi yeu cau o dap an."""

    DONG = "dong"                # VND — khong doi
    NGHIN_DONG = "nghin_dong"    # /1e3
    TRIEU_DONG = "trieu_dong"    # /1e6
    TY_DONG = "ty_dong"          # /1e9
    PERCENT = "percent"          # ratio -> phan tram
    TIMES = "times"              # "lan" — ty so, GIU NGUYEN
    NONE = "none"                # cau hoi khong neu don vi


# Chia cho he so nay de doi tu VND sang don vi duoc hoi.
# PERCENT/TIMES khong phai don vi tien te -> khong nam o day.
MONETARY_DIVISORS: dict[AskedUnit, float] = {
    AskedUnit.DONG: 1.0,
    AskedUnit.NGHIN_DONG: 1e3,
    AskedUnit.TRIEU_DONG: 1e6,
    AskedUnit.TY_DONG: 1e9,
}

# THU TU QUAN TRONG. Do tren bo cau hoi that:
#  - 7 cau chua CA "%" lan "lan" ("ty le ... bao nhieu lan") -> "lan" phai
#    duoc xet TRUOC "%" vi no la don vi cua CAU TRA LOI, con "%" thuong
#    nam trong menh de mo ta ("ty le so huu 51%").
#  - "ty dong" phai truoc "dong", "trieu dong" truoc "dong": neu khong,
#    "dong" khop truoc va moi cau tien te deu thanh DONG.
# Match tren text da bo dau (normalize_text) de chiu duoc cach go khac dau.
_UNIT_PATTERNS: tuple[tuple[re.Pattern[str], AskedUnit], ...] = (
    (re.compile(r"\bbao nhieu lan\b"), AskedUnit.TIMES),
    (re.compile(r"\b(?:la|bang|dat)\s+bao nhieu\s+lan\b"), AskedUnit.TIMES),
    (re.compile(r"\bty\s*dong\b"), AskedUnit.TY_DONG),
    (re.compile(r"\btrieu\s*dong\b"), AskedUnit.TRIEU_DONG),
    (re.compile(r"\b(?:nghin|ngan)\s*dong\b"), AskedUnit.NGHIN_DONG),
    (re.compile(r"\bbao nhieu\s*%"), AskedUnit.PERCENT),
    (re.compile(r"\bbao nhieu phan tram\b"), AskedUnit.PERCENT),
    (re.compile(r"\bty le phan tram\b"), AskedUnit.PERCENT),
    (re.compile(r"\bdong\b"), AskedUnit.DONG),
    (re.compile(r"%"), AskedUnit.PERCENT),
    (re.compile(r"\bphan tram\b"), AskedUnit.PERCENT),
    (re.compile(r"\blan\b"), AskedUnit.TIMES),
)


def detect_asked_unit(question: str) -> AskedUnit:
    """Don vi cau hoi yeu cau. Khong chac -> NONE (khong doi gi ca).

    Nguyen tac an toan: NONE giu nguyen gia tri. Doan bua mot don vi se
    lam hong cau dang dung, con NONE chi giu nguyen hien trang.
    """
    flat = normalize_text(question)
    for pattern, unit in _UNIT_PATTERNS:
        if pattern.search(flat):
            return unit
    return AskedUnit.NONE


@dataclass(frozen=True)
class NormalizedAnswer:
    """Ket qua sau khi doi don vi — giu ca gia tri goc de debug/log."""

    value: float
    raw_value: float
    unit: AskedUnit
    divisor: float = 1.0
    converted: bool = False


class AnswerNormalizer:
    """Doi ket qua thuc thi (VND / ratio) sang don vi cau hoi yeu cau.

    Dat o BOUNDARY cuoi cung (SubmissionBuilder), sau execution, truoc khi
    lam tron. Deterministic hoan toan — khong goi LLM.
    """

    def __init__(self, round_to: int = 2):
        self.round_to = round_to

    def normalize(
        self, value: float, question: str, *, unit: AskedUnit | None = None
    ) -> NormalizedAnswer:
        asked = unit if unit is not None else detect_asked_unit(question)
        return self.apply(value, asked)

    def apply(self, value: float, asked: AskedUnit) -> NormalizedAnswer:
        raw = float(value)

        if asked in MONETARY_DIVISORS:
            div = MONETARY_DIVISORS[asked]
            out = raw / div
            return NormalizedAnswer(
                value=round(out, self.round_to), raw_value=raw,
                unit=asked, divisor=div, converted=div != 1.0,
            )

        if asked is AskedUnit.PERCENT:
            # Code sinh ra da duoc yeu cau tra dang phan tram (pandas_gen
            # rule 6: 15.3 chu khong phai 0.153). Khong nhan 100 lan nua o
            # day — se thanh 1530. Chi lam tron.
            return NormalizedAnswer(
                value=round(raw, self.round_to), raw_value=raw,
                unit=asked, divisor=1.0, converted=False,
            )

        # TIMES va NONE: ty so / khong ro don vi -> giu nguyen.
        return NormalizedAnswer(
            value=round(raw, self.round_to), raw_value=raw,
            unit=asked, divisor=1.0, converted=False,
        )


def divisor_for(question: str) -> float:
    """He so chia ung voi cau hoi — tien cho prompt/log."""
    return MONETARY_DIVISORS.get(detect_asked_unit(question), 1.0)
