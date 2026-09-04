from abc import ABC, abstractmethod
from typing import List

from .html_handler import GridTable
from ..contracts.schemas import Document, Table
from ..utils.logging import get_logger
from ..config import get_settings

LOGGER = get_logger("ocr_pipeline")
LLM_PROVIDER = get_settings().llm

class TableFixer(ABC):
    @abstractmethod
    def fix(self, table: GridTable) -> GridTable:
        pass

class OCRTableFixerPipeline:
    """
    Pipeline tích hợp 3 TableFixer chạy tuần tự để làm sạch và chuẩn hóa HtmlTable,
    sau đó chuyển đổi thành Dataframe thông qua TableUpdater.
    """
    def __init__(self):
        self.fixers = []

    def process_table(self, table: Table) -> List[Table]:
        LOGGER.detail(f"Bắt đầu sửa chữa bảng tại dòng: {table.line}")
        grids = GridTable.build(table)
        
        for fixer in self.fixers:
            for i in len(grids):
                grids[i] = fixer.fix(grids[i])
        
        # Chuyển đổi và lưu CSV
        updated_tables = GridTable.finalize(grids, table)
        LOGGER.detail(f"Hoàn tất lưu CSV cho bảng tại dòng: {table.line}")
        return updated_tables
        
    def process_document(self, doc: Document) -> Document:
        LOGGER.progress(f"Bắt đầu chuẩn hóa {len(doc.tables)} bảng cho Document: {doc.doc_id}")

        leng = len(doc.tables)
        for idx in range(leng):
            updated_tables = self.process_table(doc.tables[idx])
            doc.tables[idx] = updated_tables[0]
            doc.tables.extend(updated_tables[1:])
            
        return doc