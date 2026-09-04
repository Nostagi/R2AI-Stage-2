import json
import pandas as pd
from typing import List
import concurrent.futures

from ..contracts.schemas import Table, Document
from ..config import get_settings
from ..utils.logging import get_logger

LOGGER = get_logger("table_describe")

class TableDescribePipeline:
    def __init__(self):
        self.settings = get_settings()
        self.llm = self.settings.llm
        self.llm_alias = "general purpose"
        self.prompt_name = "table describe"

    def _extract_dataframe_head(self, table: Table, num_rows: int = 3) -> tuple[str, str]:
        """Reads CSV and extracts columns + transposed head rows for prompt context."""
        if not table.csv_path:
            return "", ""
        
        try:
            df = pd.read_csv(table.csv_path)
            columns = ", ".join(df.columns.astype(str).tolist())
            
            # Chuyển đổi thành dict mapping: cột -> [danh sách 2-3 giá trị]
            df_head = df.head(num_rows)
            transpose_dict = {col: df_head[col].dropna().astype(str).tolist() for col in df_head.columns}
            
            # Ghi ra chuỗi JSON dễ đọc
            head_data = json.dumps(transpose_dict, ensure_ascii=False, indent=2)
            
            return columns, head_data
        except Exception as e:
            LOGGER.progress(f"Failed to read CSV for table {table.docs.get_id()} at {table.line}: {e}")
            return "", ""

    def describe(self, table: Table) -> Table:
        """Calls LLM to generate title and description for a single table."""
        columns, head_data = self._extract_dataframe_head(table)
        
        template_kwargs = {
            "pre_text": table.pre_text or "None",
            "post_text": table.post_text or "None",
            "columns": columns or "None",
            "head_data": head_data or "None"
        }
        
        LOGGER.detail(f"Đang describe table {table.docs.get_id()} at {table.line} bằng LLM...")
        try:
            # Lấy phản hồi từ LLM, giới hạn token đầu ra
            response = self.llm.generate(
                alias=self.llm_alias,
                prompt_name=self.prompt_name,
                template_kwargs=template_kwargs,
                max_tokens=512
            )
            
            # Xử lý kết quả JSON
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            result = json.loads(clean_response)
            table.title = result.get("title", "")
            table.description = result.get("description", "")
            LOGGER.detail(f"Describe thành công table dòng {table.docs.get_id()} at {table.line}: {table.title}")
            
        except Exception as e:
            LOGGER.detail(f"Describe table dòng {table.line} thất bại: {e}")
            
        return table

    def process_document(self, doc: Document) -> Document:
        """Xử lý tuần tự describe cho tất cả các bảng trong document."""
        LOGGER.progress(f"Bắt đầu Table Describe Pipeline cho {len(doc.tables)} tables trong Document: {doc.get_id()}")
        
        for table in doc.tables:
            self.describe(table)

        return doc
