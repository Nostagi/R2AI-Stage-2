from abc import ABC, abstractmethod
import json
from typing import List, Optional

from html_table import HtmlTable, TableRow
from ..utils.spell_check import is_numeric_like, numeric_translate, split_glued_numbers, symspell
from ..contracts.llm import LLM


class TableFixer(ABC):
    @abstractmethod
    def fix(self, table: HtmlTable) -> HtmlTable:
        pass


class RuleBasedFixer(TableFixer):
    """
    Xử lý làm sạch ở mức độ Cell: sửa số liệu, tách số dính, và tách chữ-số.
    """

    def fix(self, table: HtmlTable) -> HtmlTable:
        for row in table.rows:
            fixed_cells = []
            
            for cell in row.cells:
                # 1. Tách các khoảng dính đột ngột (vd: "Tiền mặt1500" -> "Tiền mặt 1500")
                cell = symspell(cell)
                
                # 2. Xử lý ký tự quang học nếu ô có hình dáng số liệu
                cell = numeric_translate(cell)
                
                # 3. Phân rã chuỗi số dính (nếu trả về list > 1 phần tử, ta cần làm phẳng nó ra)
                # Lưu ý: Việc này có thể làm thay đổi độ dài cột (is_corrupted)
                split_cells = split_glued_numbers(cell)
                fixed_cells.extend(split_cells)
                
            row.cells = fixed_cells
            
        return table

class HeaderFixer(TableFixer):
    """
    Xử lý Heading: Dò tìm biên giới, Flatten Multi-index, và Đánh dấu Data Row lỗi.
    Hỗ trợ LLM để tinh chỉnh tên cột nếu được cung cấp.
    """
    def __init__(self, model_fixer:LLM=None):
        # Nạp LLM từ provider nếu có, nếu không sẽ chạy thuần rule-based
        self.llm = model_fixer

    def fix(self, table: HtmlTable) -> HtmlTable:
        if not table.rows:
            return table

        # 1. Xác nhận / Dò tìm ranh giới Header
        header_limit = self._detect_header_boundary(table.rows)
        
        # Cập nhật lại cờ is_header một cách chính xác
        for i, row in enumerate(table.rows):
            row.is_header = (i < header_limit)
            
        header_rows = [r for r in table.rows if r.is_header]
        data_rows = [r for r in table.rows if not r.is_header]

        if not header_rows:
            return table # Trường hợp bảng không có header
        
        # 2. Xử lý Multi-index: Forward-fill và Flatten
        flat_header = self._flatten_headers(header_rows)

        # 3. Tích hợp LLM để tinh chỉnh ngữ nghĩa (Optional)
        if self.llm:
            flat_header = self._refine_header_with_llm(flat_header)

        # Ghi đè lại bảng: Thay thế toàn bộ header cũ bằng một dòng Header 1D chuẩn mực
        table.rows = [TableRow(cells=flat_header, is_header=True)] + data_rows

        # 4. Đánh dấu các dòng dữ liệu bị lỗi (is_corrupted)
        target_cols = len(flat_header)
        for row in data_rows:
            # Nếu số ô không khớp số cột header, chắc chắn dòng này đang dính/thiếu cột
            if len(row.cells) != target_cols:
                row.is_corrupted = True

        return table

    def _detect_header_boundary(self, rows: List[TableRow]) -> int:
        """
        Dùng Heuristic dựa trên mật độ số liệu để chốt số dòng header.
        Nếu dòng có chứa token mang hình dáng số liệu, đó là dòng bắt đầu Data.
        """
        # Nếu factory đã gán is_header (qua việc so khớp n trang), lấy đó làm mốc tham chiếu
        factory_limit = sum(1 for r in rows if r.is_header)
        
        for i, row in enumerate(rows):
            # Đếm số lượng ô chứa dạng số liệu (numeric) trên mỗi dòng
            num_count = sum(1 for cell in row.cells if is_numeric_like(cell))
            
            # Một dòng dữ liệu tài chính thường có ít nhất 1-2 ô chứa số
            if num_count > 0:
                # Nếu heuristic tìm thấy data sớm hơn/muộn hơn factory, ưu tiên heuristic
                return i 
                
        return factory_limit if factory_limit > 0 else 0 # Fallback mặc định 1 dòng

    def _flatten_headers(self, header_rows: List[TableRow]) -> List[str]:
        """Lấp ô trống (ffill) ngang/dọc và nén đa tầng thành 1D array."""
        n_cols = max((len(r.cells) for r in header_rows), default=0)
        grid = []
        
        # Đệm ô rỗng cho các dòng bị thiếu cột
        for r in header_rows:
            grid.append(r.cells + [""] * (n_cols - len(r.cells)))

        # Forward Fill 2 chiều
        for i in range(len(grid)):
            for j in range(n_cols):
                if not grid[i][j]:
                    if j > 0 and grid[i][j-1]:      # Ưu tiên kế thừa chiều ngang (colspan)
                        grid[i][j] = grid[i][j-1]
                    elif i > 0 and grid[i-1][j]:    # Nếu không, kế thừa chiều dọc (rowspan)
                        grid[i][j] = grid[i-1][j]
                        
        # Gộp cột (Join)
        flat = []
        for j in range(n_cols):
            col_parts = []
            for i in range(len(grid)):
                val = grid[i][j].strip()
                # Tránh lặp từ nếu cha và con có tên giống nhau
                if val and (not col_parts or val != col_parts[-1]):
                    col_parts.append(val)
            flat.append("_".join(col_parts))
            
        return flat

    def _refine_header_with_llm(self, flat_header: List[str]) -> List[str]:
        """
        Dùng LLM chạy nền để làm sạch nhiễu OCR, bỏ các gạch nối (_) thừa 
        do lỗi merge, và chuẩn hóa lại tên cột BCTC.
        """
        if not self.llm:
            return flat_header

        n_cols = len(flat_header)
        
        # Truyền tham số dưới dạng dict vào template_kwargs theo interface của LLM
        template_kwargs = {
            "flat_header": json.dumps(flat_header, ensure_ascii=False),
            "n_cols": n_cols
        }

        try:
            # Gọi phương thức generate với template_kwargs
            response = self.llm.generate(
                template_kwargs=template_kwargs,
                system_prompt="You are a precise data processing assistant that outputs raw JSON only."
            )

            # Trích xuất đoạn JSON array từ output của LLM
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx : end_idx + 1]
                refined_header = json.loads(json_str)

                # Đảm bảo kết quả trả về đúng số lượng cột ban đầu
                if isinstance(refined_header, list) and len(refined_header) == n_cols:
                    return [str(item).strip() for item in refined_header]

        except Exception:
            # Fallback an toàn về header gốc nếu có lỗi parse JSON hoặc lỗi runtime
            pass

        return flat_header

class LLMBasedFixer(TableFixer):
    """
    Sửa chữa các dòng dữ liệu bị xô lệch cấu trúc (is_corrupted == True 
    hoặc len(row.cells) != n_cols) bằng cơ chế LLM Batch Inference.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def fix(self, table: HtmlTable) -> HtmlTable:
        if not table.rows or not self.llm:
            return table

        # 1. Trích xuất Header 1D chuẩn đã được xử lý từ HeaderFixer
        header_rows = [r for r in table.rows if r.is_header]
        if not header_rows:
            return table

        flat_header = header_rows[0].cells
        n_cols = len(flat_header)
        if n_cols == 0:
            return table

        # 2. Sàng lọc danh sách dòng lỗi và chuẩn bị batch context
        corrupted_indices: List[int] = []
        template_kwargs_list: List[dict] = []

        for idx, row in enumerate(table.rows):
            if row.is_header:
                continue

            # Đánh dấu xử lý nếu dòng bị gán flag is_corrupted hoặc độ dài ô không khớp
            if row.is_corrupted or len(row.cells) != n_cols:
                corrupted_indices.append(idx)
                template_kwargs_list.append({
                    "flat_header": json.dumps(flat_header, ensure_ascii=False),
                    "n_cols": n_cols,
                    "corrupted_row": json.dumps(row.cells, ensure_ascii=False)
                })

        # Nếu không có dòng nào bị lỗi, trả về bảng giữ nguyên
        if not template_kwargs_list:
            return table

        # 3. Thực thi Batch Inference
        try:
            responses = self.llm.generate_batch(
                template_kwargs_list=template_kwargs_list,
                system_prompt="You are a data cleaning assistant. Output raw JSON arrays only."
            )
        except Exception:
            # Nếu có sự cố trong quá trình gọi LLM, giữ nguyên bảng để đảm bảo an toàn
            return table

        # 4. Parse kết quả và ghi đè dòng lỗi (Fallback giữ nguyên nếu parse thất bại)
        for idx, response in zip(corrupted_indices, responses):
            target_row = table.rows[idx]
            parsed_cells = self._parse_json_array(response, expected_len=n_cols)

            if parsed_cells is not None:
                target_row.cells = parsed_cells
                target_row.is_corrupted = False  # Hạ cờ sau khi sửa thành công

        return table

    def _parse_json_array(self, response: str, expected_len: int) -> Optional[List[str]]:
        """
        Trích xuất, parse và validate độ dài mảng JSON từ kết quả phản hồi của LLM.
        """
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']')

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx : end_idx + 1]
                arr = json.loads(json_str)

                # Validation khắt khe: Phải là mảng và có số phần tử bằng đúng số cột
                if isinstance(arr, list) and len(arr) == expected_len:
                    return [str(item).strip() for item in arr]
        except Exception:
            pass

        return None