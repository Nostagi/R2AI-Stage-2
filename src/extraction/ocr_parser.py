import re
from typing import List, Optional
from collections import deque

from ..contracts.schemas import Document, Table
from ..utils.logging import get_logger
from ..utils.re import truncate_text

LOGGER = get_logger("ocr_pipeline")

class OCRParser:

    def __init__(self):
        self._pre_text = deque(maxlen=5)
        self._post_text: List = None
        self._last_table: Optional[Table] = None
        self._last_header: Optional[str] = None

    def process_single_ocr_file(self, doc: Document) -> Document:
        """
        Đọc và phân tích luồng dữ liệu từng dòng của tài liệu OCR.
        Kết quả trả về là một Document đã được định dạng đầy đủ cáu trúc.
        Tuy nhiên, các Table mới chỉ được set HTML raw content.
        """

        # Reset state
        self._pre_text = deque(maxlen=5)
        self._post_text: List = None
        self._last_table: Optional[Table] = None
        self._last_header: Optional[str] = None

        try:
            with doc.doc_path.open("r", encoding="utf-8", errors="replace") as f:
                for line_idx, line_content in enumerate(f, start=1):
                    line_clean = line_content.strip()
                    
                    if not line_clean:
                        continue

                    if self._detect_new_page(line_clean):
                        self._pre_text.clear()
                        continue


                    html_raw = self._extract_html_table(line_clean)
                    if html_raw:
                        self._handle_table_line(doc, line_idx, html_raw)
                    else:
                        self._handle_normal_line(line_clean)

            # Xử lý các bảng chưa thu thập đủ 5 dòng post_text khi đã đọc hết file
            self._finalize_post_text()

            unique_tables = len(set(id(t) for t in doc.tables.values()))
            LOGGER.progress(f"Đã xử lý xong {doc.doc_id}. Tìm thấy {unique_tables} bảng độc lập.")   

        except Exception as e:
            LOGGER.detail(f"Lỗi khi parse file {doc.doc_path}: {str(e)}")
            
        return doc

    def _handle_table_line(self, doc: Document, line_idx: int, html_raw: str) -> None:
        current_header = self._extract_table_header(html_raw)
        if not current_header :
            LOGGER.progress(f"Lỗi định dạng table tại dòng {line_idx} của document {doc.doc_id}.")
            LOGGER.detail(f"Lỗi định dạng table tại dòng {line_idx} của document {doc.doc_id}.")
        
        # Xử lý gộp bảng
        if self._last_header and (current_header == self._last_header):
            self._last_table.line.append(line_idx)
            self._last_table.html_table.append(html_raw)
            doc.tables[line_idx] = self._last_table
            LOGGER.detail(f"\tĐã gộp bảng tại dòng {line_idx} vào bảng trước đó tại {self._last_table.line[0]}.")
                
            # Tiếp tục merge nên ta reset tiến trình gom post_text
            self._pre_text.clear()
            self._post_text = []
            return

        # Bắt đầu table kế tiếp mà không merge -> Chốt post_text của bảng trước
        self._finalize_post_text()

        # Tạo bảng mới
        pre_text_str = truncate_text(list(self._pre_text), max_words=200, keep_last=True)
        new_table = Table(
            docs=doc,
            line=[line_idx],
            year=doc.year,
            report_type=doc.report_type,
            html_table=[html_raw],
            pre_text=pre_text_str if pre_text_str else None
        )
        
        doc.tables[line_idx] = new_table
        LOGGER.detail(f"\tĐã trích xuất bảng mới tại dòng: {line_idx}")
        
        # Cập nhật trạng thái để bắt đầu track post_text cho bảng mới
        self._last_table = new_table
        self._last_header = current_header
        self._pre_text.clear()
        self._post_text = []

    def _handle_normal_line(self, line: str) -> None:
        self._pre_text.append(line)
        
        if self._post_text is not None:
            self._post_text.append(line)

            if len(self._post_text) >= 5:
                self._finalize_post_text()

    def _detect_new_page(self, line: str) -> bool:
            """
            Kiểm tra dấu hiệu bắt đầu một trang mới trong tệp OCR.
            """
            return line.startswith("===== PAGE ") and line.endswith(" =====")
    
    def _extract_html_table(self, line:str) -> str:
        if "<table>" in line and "</table>" in line:
            # Trích xuất chính xác đoạn HTML
            start_idx = line.find("<table>")
            end_idx = line.find("</table>") + len("</table>")
            html_raw = line[start_idx:end_idx]
                                
            return html_raw

        else:
            return None

    def _extract_table_header(self, html_raw: str) -> str:
        """
        Trích xuất dòng đầu tiên (thẻ <tr> đầu tiên) của bảng HTML để phục vụ đối chiếu gộp bảng.
        """
        match = re.search(r'(<tr[^>]*>.*?</tr>)', html_raw, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else ""

    def _finalize_post_text(self) -> None:
        """
        Đóng gói post_text cho bảng lưu trong self.last_table và dừng tracking.
        """
        if self._post_text and self._last_table:
            post_str = truncate_text(self._post_text, max_words=200, keep_last=False)
            self._last_table.post_text = post_str if post_str else None

        self._post_text = None
