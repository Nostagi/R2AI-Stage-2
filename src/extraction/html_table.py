import re
from dataclasses import dataclass, replace
from pathlib import Path
import pandas as pd
from typing import List, Optional

from ..contracts.schemas import Table
from src.config import get_settings
from ..utils.io import write_csv

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

def _extract_cells(row_html: str) -> List[str]:
    """Bóc tách text thô từ các thẻ td/th trong một dòng tr."""
    cells = _CELL_RE.findall(row_html)
    return [_TAG_RE.sub(" ", c).replace("\xa0", " ").strip() for c in cells]

@dataclass
class TableRow:
    cells: List[str]
    is_header: bool = False
    is_corrupted: bool = False

@dataclass
class HtmlTable:
    rows: List[TableRow]
    table: Optional[Table] = None

    @classmethod
    @classmethod
    def from_schema(cls, table: Table) -> "HtmlTable":
        """Khởi tạo HtmlTable từ schema Table và giữ reference đến Table gốc."""
        if not table.html_table:
            return cls(rows=[], table=table)

        def _extract_cells(row_html: str) -> List[str]:
            cells = _CELL_RE.findall(row_html)
            return [_TAG_RE.sub(" ", c).replace("\xa0", " ").strip() for c in cells]

        parsed_parts: List[List[List[str]]] = []
        for html_chunk in table.html_table:
            raw_rows = _ROW_RE.findall(html_chunk)
            parsed_parts.append([_extract_cells(r) for r in raw_rows])

        if not parsed_parts:
            return cls(rows=[], table=table)

        header_count = 0
        if len(parsed_parts) > 1:
            part1 = parsed_parts[0]
            for i, row in enumerate(part1):
                is_common = all(
                    i < len(p) and p[i] == row 
                    for p in parsed_parts[1:]
                )
                if is_common:
                    header_count += 1
                else:
                    break

        final_rows: List[TableRow] = []
        for i, row_cells in enumerate(parsed_parts[0]):
            final_rows.append(TableRow(cells=row_cells, is_header=(i < header_count)))

        for part in parsed_parts[1:]:
            for row_cells in part[header_count:]:
                final_rows.append(TableRow(cells=row_cells, is_header=False))

        return cls(rows=final_rows, table=table)

class TableUpdater:
    """
    Chuyển đổi HtmlTable thành pandas.DataFrame, xuất file CSV bằng `write_csv` (utf-8-sig),
    và trả về một instance Table mới đã cập nhật csv_path, xóa html_table và giữ lại toàn bộ metadata.
    """

    def __init__(self, processed_dir: Optional[Path] = None):
        settings = get_settings()
        self.processed_dir = processed_dir or settings.paths.processed
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def update(self, html_table: HtmlTable) -> Table:
        orig_table = html_table.table
        if orig_table is None:
            raise ValueError("HtmlTable không chứa tham chiếu đến đối tượng Table gốc (.table is None).")

        # 1. Phân tách Header (1D) và Data Rows để tạo DataFrame
        header_rows = [r for r in html_table.rows if r.is_header]
        data_rows = [r for r in html_table.rows if not r.is_header]

        columns = header_rows[0].cells if header_rows else None
        data = [r.cells for r in data_rows]

        df = pd.DataFrame(data, columns=columns)

        # 2. Tạo tên file CSV dựa trên doc_id (từ Document) và dòng trong file OCR
        doc_id = orig_table.docs.doc_id if orig_table.docs else "doc"
        lines_str = orig_table.line[0]
        
        filename = f"at_{lines_str}.csv"
        csv_file_path = self.processed_dir / doc_id / filename

        # 3. Ghi file CSV bằng io utils (write_csv)
        write_csv(df, csv_file_path)

        # 4. Tạo đối tượng Table mới kế thừa toàn bộ core data & metadata
        # Sử dụng dataclasses.replace để copy đầy đủ các trường (docs, line, statement, title,...)
        updated_table = replace(
            orig_table,
            csv_path=Path(csv_file_path),  # Chuẩn Path theo schema
            html_table=None                # Xóa trắng trường html_table
        )

        return updated_table