"""Giai ma header cot BCTC thanh (year, period).

VAN DE GOC: `_find_year_cols` cu lay `_YEAR_RE.findall(col)[-1]` — chi lay
con so nam, vut bo ngay/thang. Hai cot rat pho bien trong BCTC:

    "31/12/2015"  va  "01/01/2015"

deu tra ve 2015. Nhung chung la HAI THOI DIEM KHAC NHAU voi hai gia tri
khac nhau -> long dataframe co hai dong cung (item, year), khac `value`.
Prompt lai bao LLM lay `.iloc[0]` -> chon bua mot trong hai.

QUY UOC KE TOAN (da kiem chung tren corpus, khong phai gia dinh):
    AAA 2015, cot 31/12/2015 -> Tai san ngan han = 1.071.561.008.455
    AAA 2016, cot 01/01/2016 -> Tai san ngan han = 1.071.561.008.455  (KHOP)
So du dau ky 01/01/N chinh la so du cuoi ky 31/12/(N-1). Vi vay:

    31/12/N -> year=N,   period=closing
    01/01/N -> year=N-1, period=opening      <- semantic year lui mot nam
    dd/mm/N -> year=N,   period=point_in_time   (KHONG lui nam)
    YYYY    -> year=N,   period=annual

Chi ap rule lui nam khi ngay DUNG BANG 01/01. Corpus co 57 cot ngay le
(8/8/2019, 15/10/2019...) — lui nam cho chung se sai hoan toan.

CHONG NHAN NHAM: 1.417/3.000 cot chua nam la VAN XUOI, khong phai cot du
lieu ("BCTC tong hop nam 2025 (Da kiem toan)", "bo nhiem ngay 6 thang 6
nam 2025"). Parser cu nhan het chung thanh cot nam. Xem `_looks_like_prose`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..utils.spell_check import normalize_text


class Period(str, Enum):
    """Thoi diem ma cot du lieu bieu dien."""

    CLOSING = "closing"              # 31/12/N — so du cuoi ky
    OPENING = "opening"              # 01/01/N — so du dau ky (= cuoi ky N-1)
    POINT_IN_TIME = "point_in_time"  # ngay cu the khac
    ANNUAL = "annual"                # chi co nam, khong co ngay
    UNKNOWN = "unknown"


# Ngay thang: cho phep '/', '-', '.' lam dau phan cach (corpus co ca ba;
# rieng dang '.' xuat hien 192 lan, vd "31.12.2025Triệu đồng").
# Cho phep space thua — OCR hay chen.
_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*((?:19|20)\d{2})(?!\d)"
)
# "ngay 31 thang 12 nam 2015" — dang viet chu.
_DATE_WORDS_RE = re.compile(
    r"(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*((?:19|20)\d{2})", re.IGNORECASE
)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# Cot thoi diem viet bang CHU, KHONG chua nam. Rat pho bien — do tren 50
# bao cao ngau nhien: 1.890 cot dang nay, NHIEU HON ca 1.573 cot ghi ngay.
# Parser chi bat ngay/nam se bo sot het, day toan bo bang do vao nhanh
# fallback (`period=unknown`, `year=fallback_year`) -> moi cot deu ra cung
# mot (item, year, period) voi gia tri khac nhau. Day moi la nguyen nhan
# chinh cua collision, khong phai cap 31/12 vs 01/01.
#
# Nam duoc suy tu `report_year` cua tai lieu (tham so `context_year`):
#   "So cuoi nam"/"Nam nay"   -> year = report_year,     closing/annual
#   "So dau nam"/"Nam truoc"  -> year = report_year - 1, opening/annual
_WORD_PERIODS: tuple[tuple[re.Pattern[str], int, Period], ...] = (
    # Thu tu quan trong: "so dau nam va cuoi nam" chua ca hai cum — cum
    # "dau nam" dung truoc nen phai xet trước để không nhận nhầm.
    (re.compile(r"so\s*du\s*dau\s*(?:nam|ky)|so\s*dau\s*(?:nam|ky)"), -1, Period.OPENING),
    (re.compile(r"\bdau\s*(?:nam|ky)\b"), -1, Period.OPENING),
    (re.compile(r"so\s*du\s*cuoi\s*(?:nam|ky)|so\s*cuoi\s*(?:nam|ky)"), 0, Period.CLOSING),
    (re.compile(r"\bcuoi\s*(?:nam|ky)\b"), 0, Period.CLOSING),
    (re.compile(r"\bnam\s*truoc\b|\bky\s*truoc\b|\bnam\s*ngoai\b"), -1, Period.ANNUAL),
    (re.compile(r"\bnam\s*nay\b|\bky\s*nay\b|\bnam\s*hien\s*tai\b"), 0, Period.ANNUAL),
)

# Tu khoa cho thay o la VAN XUOI/ghi chu, khong phai nhan cot du lieu.
# Do tren corpus that: 1.417 cot dang nay bi parser cu nhan nham.
_PROSE_HINTS = (
    "bctc", "bao cao", "báo cáo", "kiem toan", "kiểm toán", "bo nhiem",
    "bổ nhiệm", "nghi quyet", "nghị quyết", "thong tu", "thông tư",
    "quyet dinh", "quyết định", "giay chung nhan", "giấy chứng nhận",
    "hop dong", "hợp đồng", "nhiem ky", "nhiệm kỳ", "dai hoi", "đại hội",
    "mau so", "mẫu số", "tu ngay", "từ ngày", "den ngay", "đến ngày",
    "duoc bau", "được bầu", "cap ngay", "cấp ngày", "so huu", "sở hữu",
)
# Nhan cot du lieu that thuong rat ngan: "31/12/2015", "2015",
# "31/12/2015 VND", "Tại ngày 31.12.2025 Triệu đồng". Van xuoi thi dai.
_MAX_LABEL_CHARS = 60


@dataclass(frozen=True)
class ColumnPeriod:
    """Ket qua giai ma mot ten cot.

    `year` la NAM TAI CHINH ngu nghia (da lui cho opening balance).
    `source_date`/`raw_header` giu de truy nguoc vi sao map nhu vay.
    """

    year: int
    period: Period
    raw_header: str = ""
    source_date: str | None = None      # "31/12/2015" — None neu chi co nam

    @property
    def label(self) -> str:
        """Nhan gon dung trong card/prompt: '2015 (closing)'."""
        return f"{self.year} ({self.period.value})"


def _looks_like_prose(text: str) -> bool:
    """O nay la cau van/ghi chu chu khong phai nhan cot du lieu."""
    if len(text) > _MAX_LABEL_CHARS:
        return True
    low = text.lower()
    if any(h in low for h in _PROSE_HINTS):
        return True
    # Nhieu hon mot nam trong cung mot o -> khoang thoi gian, khong phai cot.
    return len(_YEAR_RE.findall(text)) > 1


def parse_column_period(
    header: object, context_year: int | None = None
) -> ColumnPeriod | None:
    """Giai ma ten cot -> (year, period). None neu khong phai cot thoi gian.

    `context_year`: nam bao cao cua tai lieu. Bat buoc de giai ma cac cot
    viet bang chu ("So cuoi nam", "Nam truoc") vi ban than chung khong chua
    nam nao. Khong co no thi cac cot do tra ve None.

    >>> parse_column_period("31/12/2015").year
    2015
    >>> parse_column_period("01/01/2015").year          # so du dau ky
    2014
    >>> parse_column_period("Số đầu năm", context_year=2022).year
    2021
    """
    text = str(header or "").strip()
    if not text or _looks_like_prose(text):
        return None

    m = _DATE_RE.search(text) or _DATE_WORDS_RE.search(text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        raw_date = m.group(0).strip()

        # Ngay khong hop le (OCR loi) -> ha xuong muc chi biet nam.
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return ColumnPeriod(year, Period.UNKNOWN, text, raw_date)

        if (day, month) == (31, 12):
            return ColumnPeriod(year, Period.CLOSING, text, raw_date)
        if (day, month) == (1, 1):
            # So du dau ky N == so du cuoi ky N-1. Lui nam de
            # `df[df.year == 2015]` tra ve dung so lieu cua nam 2015.
            return ColumnPeriod(year - 1, Period.OPENING, text, raw_date)
        return ColumnPeriod(year, Period.POINT_IN_TIME, text, raw_date)

    years = _YEAR_RE.findall(text)
    if len(years) == 1:
        return ColumnPeriod(int(years[0]), Period.ANNUAL, text, None)

    # Cot viet bang chu: "So cuoi nam", "Nam truoc"... — can nam bao cao.
    if context_year is not None:
        flat = normalize_text(text)
        for pattern, offset, period in _WORD_PERIODS:
            if pattern.search(flat):
                return ColumnPeriod(context_year + offset, period, text, None)

    return None


def find_period_cols(
    columns: list[object],
    exclude: set[object] | None = None,
    context_year: int | None = None,
) -> dict[object, ColumnPeriod]:
    """Map cot -> ColumnPeriod cho moi cot bieu dien mot thoi diem.

    Neu hai cot cung cho ra (year, period) giong het nhau (vd bang co hai
    cot '31/12/2015' do OCR nhan doi), cot DAU duoc giu — cot sau bi bo de
    khong sinh dong trung.
    """
    exclude = exclude or set()
    out: dict[object, ColumnPeriod] = {}
    seen: set[tuple[int, Period]] = set()

    for col in columns:
        if col in exclude:
            continue
        parsed = parse_column_period(col, context_year)
        if parsed is None:
            continue
        key = (parsed.year, parsed.period)
        if key in seen:
            continue
        seen.add(key)
        out[col] = parsed
    return out


# ── phia CAU HOI ──────────────────────────────────────────

# Cau hoi noi ro thoi diem nao. Do tren 1012 cau hoi that:
#   326 cau "cuoi nam/cuoi ky", 91 cau "ngay 31/12", 118 cau "trong nam",
#    14 cau "dau nam", 2 cau "ngay 01/01", 437 cau khong noi gi.
# 24 cau OPENING la it nhung deu la cau se SAI HOAN TOAN neu lay closing.
_Q_OPENING_RE = re.compile(
    r"dau\s*(?:nam|ky)|(?:tai|den)?\s*ngay\s*0?1\s*[/\-.]\s*0?1", re.IGNORECASE
)
_Q_CLOSING_RE = re.compile(
    r"cuoi\s*(?:nam|ky)|(?:tai|den)?\s*ngay\s*31\s*[/\-.]\s*12", re.IGNORECASE
)
_Q_ANNUAL_RE = re.compile(r"trong\s*nam|ca\s*nam|trong\s*ky", re.IGNORECASE)


def detect_requested_period(question: str) -> Period | None:
    """Thoi diem cau hoi yeu cau; None = khong noi ro (de pipeline tu chon).

    Thu tu uu tien: OPENING truoc CLOSING. Cau "muc tang ... tu dau nam den
    cuoi nam" chua ca hai, nhung tin hieu OPENING la tin hieu hiem va co
    chu dich — bo qua no gay sai hoan toan, con lay them opening thi chi la
    du mot dong trong prompt.
    """
    from ..utils.spell_check import normalize_text

    flat = normalize_text(question)
    if _Q_OPENING_RE.search(flat):
        return Period.OPENING
    if _Q_CLOSING_RE.search(flat):
        return Period.CLOSING
    if _Q_ANNUAL_RE.search(flat):
        return Period.ANNUAL
    return None
