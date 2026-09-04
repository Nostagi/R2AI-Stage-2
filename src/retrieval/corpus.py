from typing import Optional, List, Dict, Any
from pathlib import Path

from ..contracts.retriever import Corpus
from ..contracts.schemas import Table
from ..utils.io import read_csv


class StringCorpus(Corpus):
    """Corpus dành cho văn bản thuần (danh sách chuỗi văn bản thô)."""

    def __init__(self, items: Optional[List[str]] = None):
        self._items: List[str] = items or []

    def add(self, item: str) -> int:
        self._items.append(item)
        return len(self._items) - 1

    def get(self, index: int) -> str:
        return self._items[index]

    def get_batch(self, indices: List[int]) -> List[str]:
        return [self._items[i] for i in indices]

    def get_info(self, index: int) -> Dict[str, Any]:
        return {"content": self._items[index]}

    def to_text(self, index: int) -> str:
        return self._items[index]

    def __len__(self) -> int:
        return len(self._items)


class TableMetadataCorpus(Corpus):
    """Corpus dành cho đối tượng Table, chỉ trích xuất thông tin Metadata cơ bản."""

    def __init__(self, tables: Optional[List[Table]] = None):
        self._tables: List[Table] = tables or []

    def add(self, item: Table) -> int:
        self._tables.append(item)
        return len(self._tables) - 1

    def get(self, index: int) -> Table:
        return self._tables[index]

    def get_batch(self, indices: List[int]) -> List[Table]:
        return [self._tables[i] for i in indices]

    def get_info(self, index: int) -> Dict[str, Any]:
        table = self._tables[index]
        doc_id = table.docs.doc_id if table.docs else ""
        return {
            "doc_id": doc_id,
            "line": table.line,
            "title": table.title or "",
            "company": table.company or "",
            "year": table.year or "",
            "report_type": table.report_type or "",
            "statement": table.statement or ""
        }

    def to_text(self, index: int) -> str:
        table = self._tables[index]
        parts = [
            f"Công ty: {table.company}" if table.company else "",
            f"Năm: {table.year}" if table.year else "",
            f"Loại báo cáo: {table.report_type}" if table.report_type else "",
            f"Báo cáo: {table.statement}" if table.statement else "",
            f"Tiêu đề: {table.title}" if table.title else "",
            f"Mô tả: {table.description}" if table.description else ""
        ]
        return "\n\t".join([p for p in parts if p])

    def __len__(self) -> int:
        return len(self._tables)


class TableFewShotCorpus(TableMetadataCorpus):
    """
    Corpus nâng cấp cho Table: Bao gồm toàn bộ Metadata + Danh sách tên các Cột (Header)
    và vài dòng dữ liệu mẫu (Few-shot) được đọc từ file CSV.
    """

    def __init__(self, tables: Optional[List[Table]] = None):
        super().__init__(tables)
        self._headers_cache: Dict[int, List[str]] = {}
        self._fewshot_cache: Dict[int, str] = {}

    def _get_table_content(self, index: int) -> tuple[List[str], str]:
        if index in self._headers_cache and index in self._fewshot_cache:
            return self._headers_cache[index], self._fewshot_cache[index]

        table = self._tables[index]
        headers = []
        fewshot = ""
        if table.csv_path and Path(table.csv_path).exists():
            try:
                df = read_csv(table.csv_path)
                headers = [str(c) for c in df.columns.tolist()]
                # Chuyển đổi vài dòng thành text. Lấy tối đa 2 dòng.
                if not df.empty:
                    df_head = df.head(2)
                    transpose_dict = {col: df_head[col].dropna().astype(str).tolist() for col in df_head.columns}
                    import json
                    fewshot = json.dumps(transpose_dict, ensure_ascii=False)
            except Exception:
                pass

        self._headers_cache[index] = headers
        self._fewshot_cache[index] = fewshot
        return headers, fewshot

    def get_info(self, index: int) -> Dict[str, Any]:
        info = super().get_info(index)
        headers, fewshot = self._get_table_content(index)
        info["headers"] = ", ".join(headers)
        info["fewshot"] = fewshot
        return info

    def to_text(self, index: int) -> str:
        metadata_text = super().to_text(index)
        headers, fewshot = self._get_table_content(index)
        
        parts = [metadata_text]
        if headers:
            parts.append(f"Danh sách cột: {', '.join(headers)}")
        if fewshot:
            parts.append(f"Dữ liệu mẫu: {fewshot}")
            
        return " | ".join(filter(None, parts))