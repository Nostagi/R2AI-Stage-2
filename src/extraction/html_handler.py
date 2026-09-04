import __future__

from dataclasses import dataclass, field, replace
from typing import List, Dict, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from bs4.element import Tag

from ..contracts.schemas import Table
from ..utils.io import write_csv, write_json
from ..utils.spell_check import numeric_translate, is_numeric_like, is_empty
from src.config import get_settings

SETTINGS = get_settings()
INTERIM_DIR = SETTINGS.paths.interim
PROCESSED_DIR = SETTINGS.paths.processed


@dataclass
class Cell:
    content: Union[str, float, int]
    width: int = 1
    height: int = 1
    x: int = 0
    y: int = 0

    def is_numeric(self):
        return isinstance(self.content, (int, float)) \
            or is_numeric_like(self.content)

    def is_empty(self):
        return is_empty(self.content) if isinstance(self.content, str) else False

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return (self.content == other.content and 
                self.width == other.width and 
                self.height == other.height)

@dataclass
class GridTable:
    grid: List[List[Cell]]          # list of rows, each row have cells. Pointer might be duplicate.
    column_count: int = 0
    header_count: int = 0
    corrupted_rows: List[Tuple[int, List[Cell | None]]] = None
    primary_key: List[int] = field(default_factory=list)

    def row_to_str(self, i: int, sep: str = " | ") -> str:
        if i < 0 or i >= len(self.grid):
            return ""
        return sep.join([str(c.content) for c in self.grid[i]])

    def to_dataframe(self) -> Tuple[pd.DataFrame, Dict[int, str]]:
        data = []
        for r_idx in range(self.header_count, len(self.grid)):
            row = self.grid[r_idx]
            data.append([c.content for c in row[:self.column_count]])

        columns = None
        if self.header_count > 0:
            columns = [c.content if c.content else f"Unnamed_{i}" for i, c in enumerate(self.grid[self.header_count - 1])]
            
        df = pd.DataFrame(data, columns=columns)
                
        corrupted_dict = {}
        if self.corrupted_rows:
            for r_idx, _ in self.corrupted_rows:
                corrupted_dict[r_idx] = self.row_to_str(r_idx, sep=", ")
            
        return df, corrupted_dict

    def numeric_ratio(self, index:int, columns:bool=True) -> float:
        """
        Tính tỉ lệ cell numeric / tổng số cell trong một hàng hoặc cột.
        """
        line = [self.grid[r][index] for r in range(len(self.grid))] if columns \
                else self.grid[index] 

        total_count, numeric_count = 0, 0
        ptr = 0
        while ptr < len(line):
            cell = line[ptr]
            leng = cell.height if columns else cell.width
            ptr += leng
            total_count += 1
            if cell.is_numeric() or cell.is_empty():
                numeric_count += leng

        return float(numeric_count) / total_count if total_count > 0 else 0.0

    def is_numeric(self, index:int, columns:bool=True, acceptance:float = 1/2) -> Tuple[bool, List[Cell]]:
        """
        Kiểm tra xu hướng của một dòng (not column) hoặc cột (column) có phải kiểu dữ liệu dạng số không.
        Một dòng được cho là numeric nếu tỉ lệ `is_numeric_like` lớn hơn mức acceptance.
        **Lưu ý**: Các dòng trống cũng được xem là numeric (giá trị rỗng). 
        """

        line = [self.grid[r][index] for r in range(len(self.grid)) if len(self.grid[r]) > index] if columns \
                else self.grid[index] 

        numeric_cell:List[Cell] = []
        non_numeric_cell:List[Cell] = []
        empty_cell: List[Cell] = []
        total_count = 0

        ptr = 0
        while ptr < len(line):
            cell = line[ptr]
            ptr += cell.height if columns else cell.width
            total_count += 1

            if columns:
                if cell.width >= self.column_count:
                    continue

            if cell.is_numeric() :
                numeric_cell.append(cell)
            elif cell.is_empty() :
                empty_cell.append(cell)
            else:
                non_numeric_cell.append(cell)

        # empty_count
        if len(numeric_cell) != 0:    
            accept = float(len(numeric_cell) + len(empty_cell)) / total_count > acceptance
        else:       # Clueless
            accept = False 
        outlier = non_numeric_cell if accept else (numeric_cell + empty_cell)

        return accept, outlier

    def is_corrupted_row(self, row_index:int):
        return self.column_count != len(self.grid[row_index])

    def is_empty(self, index:int, columns:bool=True):
        line = set([self.grid[r][index] for r in range(len(self.grid))] if columns \
                        else self.grid[index] )

        for cell in line:
            if not cell.is_empty(): return False
        return True

    @classmethod
    def finalize(cls, parts: List['GridTable'], original_table: Table) -> List[Table]:
        results = []
        postfix = [f"_{i+1}" for i in range(len(parts))] if len(parts) > 1 else [""]

        for i, part in enumerate(parts):
            df, _ = part.to_dataframe()
            
            csv_path = _save_dataframe(original_table, df, postfix[i])
            
            new_table = replace(original_table,
                csv_path=csv_path,
                html_table=None
            )
            results.append(new_table)
            
        return results

    @classmethod
    def build(cls, table:Table) -> List['GridTable']:
        html = table.html_table

        parts:List[GridTable] = []
        for part_tag in html:
            parts.append(MakeGrid.from_html(part_tag))

        parts = RefineGrid.process(parts)
        for p in parts:
            FlattenGrid.process(p) 

        _save_parsed_table(table, parts)
        return parts


class MakeGrid:
    """
    Thực hiện parse một HTML <table> tag thành GridTable
    """

    @staticmethod
    def from_html(table:Tag) -> GridTable:
        rows = table.find_all("tr")
        grid:List[List[Cell]] = []

        for i, row in enumerate(rows):
            row_tag = row.find_all(["td", "th"], recursive=False)
            row = []
            prev_row = MakeGrid._get_row(grid, i-1)

            ptr = 0
            for cell_tag in row_tag :
                cell = Cell(
                        content=cell_tag.get_text(" ", strip=True),
                        width=MakeGrid._span(cell_tag.get("colspan")),
                        height=MakeGrid._span(cell_tag.get("rowspan")),
                        y = i
                    )

                # Rowspan from previous row (above)
                while True:
                    above = prev_row[ptr] if (prev_row and len(prev_row) > ptr) else None
                    if above and above.y + above.height > i:
                        row.append(above)
                        ptr += 1
                    else: break

                cell.x = ptr
                row.extend([cell] * cell.width)
                ptr += cell.width  

            grid.append(row)
        return GridTable(grid=grid, column_count=MakeGrid._count_columns(grid))     

    @staticmethod
    def _span(value: str | None) -> int:
        """Parse rowspan/colspan, fallback về 1 nếu HTML không hợp lệ."""
        try:
            return max(1, int(value or 1))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _get_row(grid: List[List[Cell | None]], row_index: int,) -> List[Cell | None]:
        """Đảm bảo grid có row_index và trả về row đó."""
        if row_index < 0: return None

        while len(grid) <= row_index:
            grid.append([])

        return grid[row_index]

    @staticmethod
    def _count_columns(grid: List[List[Cell | None]]) -> tuple[List[List[Cell]], int]:
        """
        Tính column_count bằng mean độ dài các row rồi round.

        Sau đó biến grid thành hình chữ nhật.
        """
        if not grid:
            return [], 0

        column_count = round(np.mean([len(row) for row in grid]))

        return column_count

class RefineGrid:
    @staticmethod
    def process(parts: List[GridTable]) -> List[GridTable]:
        for part in parts:
            RefineGrid.fix_content(part)
        
        last = RefineGrid.split_hybrid(parts[-1], columns=True)
        parts[-1] = last.pop(0)

        final = RefineGrid.merge_grid(parts)
        last.append(final)

        for part in last:
            RefineGrid.primary_key(part)

        return last

    @staticmethod
    def merge_grid(parts: List[GridTable]) -> GridTable:
        """
        Với những bảng nằm trên nhiều trang, chúng sẽ được parse thành nhiều GridTable.
        Dựa trên lượng dòng đầu tiên tối thiểu mà chúng sử dụng chung (sharing header), hàm này merge chúng thành một GridTable duy nhất.

        **Lưu ý**: Hàm này không xác nhận lại việc `parts` có phải cùng 1 bảng nằm ở nhiều trang không nha. Assumption như vậy rồi.
        """
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
            
        min_rows = min(len(p.grid) for p in parts)
        max_header_idx = 0
        
        for r in range(min_rows):
            row_0 = parts[0].grid[r]
            
            if len(set(id(c) for c in row_0)) <= 1:
                break
                
            is_shared = True
            for i in range(1, len(parts)):
                row_i = parts[i].grid[r]
                if len(row_0) != len(row_i):
                    is_shared = False
                    break
                    
                for c0, ci in zip(row_0, row_i):
                    if c0 != ci:
                        is_shared = False
                        break
                if not is_shared:
                    break
                    
            if is_shared:
                max_header_idx = r + 1
            else:
                break
                
        merged_grid = []
        merged_grid.extend(parts[0].grid)
        merged_corrupted = list(parts[0].corrupted_rows) if parts[0].corrupted_rows else []
        
        for i in range(1, len(parts)):
            offset = len(merged_grid) - max_header_idx
            merged_grid.extend(parts[i].grid[max_header_idx:])
            if parts[i].corrupted_rows:
                for r_idx, row in parts[i].corrupted_rows:
                    if r_idx >= max_header_idx:
                        merged_corrupted.append((r_idx + offset, row))
                
        result = GridTable(grid=merged_grid, column_count=parts[0].column_count, header_count=max_header_idx)
        result.corrupted_rows = merged_corrupted
        RefineGrid.fix_coordinate(result)
        return result

    @staticmethod
    def split_hybrid(junk: GridTable, columns:bool = True) -> List[GridTable]:
        """
        Một số GridTable thực chất tồn tại 2 bảng bị trộn lẫn. 
        Trong trường hợp này, ta sẽ dựa trên `is_numeric` của từng columns để xác định điểm cắt giữa các bảng.
        Thường sẽ chỉ bị lỗi ở part cuối cùng (trang cuối cùng bảng đó được ghi).
        """
        RefineGrid.normalize_size(junk, columns=False)
        
        header_rows = RefineGrid._identify_header(junk, columns)
        
        if not header_rows:
            junk.header_count = 0
            RefineGrid.normalize_size(junk, columns=True)
            RefineGrid.fix_coordinate(junk)
            return [junk]
            
        blocks = []
        start = header_rows[0]
        count = 1
        for i in range(1, len(header_rows)):
            if header_rows[i] == header_rows[i-1] + 1:
                count += 1
            else:
                blocks.append((start, count))
                start = header_rows[i]
                count = 1
        blocks.append((start, count))
        
        if len(blocks) == 1:
            junk.header_count = blocks[0][1]
            RefineGrid.normalize_size(junk, columns=True)
            RefineGrid.fix_coordinate(junk)
            return [junk]
            
        parts:List[GridTable] = []
        for i in range(len(blocks)):
            start_row = blocks[i][0]
            end_row = blocks[i+1][0] if i + 1 < len(blocks) else len(junk.grid)
            
            part = GridTable(junk.grid[start_row:end_row], column_count=junk.column_count)
            
            if junk.corrupted_rows:
                part.corrupted_rows = []
                for r_idx, row in junk.corrupted_rows:
                    if start_row <= r_idx < end_row:
                        part.corrupted_rows.append((r_idx - start_row, row))
                        
            parts.append(part)
            
        for part in parts:
            RefineGrid.normalize_size(part, columns=True)
            RefineGrid.fix_coordinate(part)
            part.header_count = len(RefineGrid._identify_header(part, columns))
        
        return parts

    @staticmethod
    def fix_coordinate(table: GridTable):
        """
        Cập nhật lại tọa độ cell.y sao cho khớp với index thực tế của row trong grid.
        Duyệt theo từng cột và nhảy r_ptr theo cell.height cho nhanh.
        """
        for c in range(table.column_count):
            r_ptr = 0
            while r_ptr < len(table.grid):
                if c >= len(table.grid[r_ptr]):
                    r_ptr += 1
                else:
                    cell = table.grid[r_ptr][c]
                    cell.y = r_ptr
                    r_ptr += cell.height

    @staticmethod
    def normalize_size(table: GridTable, columns: bool = True):
        """
        Dồn các cột/dòng bị over-expand.
        """
        if columns:
            c = 0
            while c < table.column_count:
                min_end = table.column_count
                for r in range(len(table.grid)):
                    if c < len(table.grid[r]):
                        cell = table.grid[r][c]
                        min_end = min(min_end, cell.x + cell.width)
                
                n = min_end - (c + 1)
                if n > 0:
                    for r in range(len(table.grid)):
                        del table.grid[r][c+1 : c+1+n]
                    
                    seen = set()
                    for r in range(len(table.grid)):
                        if c >= len(table.grid[r]):
                            break
                        cell = table.grid[r][c]
                        if id(cell) not in seen:
                            cell.width -= n
                            seen.add(id(cell))
                            
                        for col_idx in range(c+1, len(table.grid[r])):
                            right_cell = table.grid[r][col_idx]
                            if id(right_cell) not in seen:
                                right_cell.x -= n
                                seen.add(id(right_cell))
                                
                    table.column_count -= n
                c += 1
        else:
            r = 0
            while r < len(table.grid):
                min_end = len(table.grid)
                for c in range(table.column_count):
                    if c >= len(table.grid[r]):
                        break
                    cell = table.grid[r][c]
                    min_end = min(min_end, cell.y + cell.height)
                
                n = min_end - (r + 1)
                if n > 0:
                    del table.grid[r+1 : r+1+n]
                    
                    seen = set()
                    for c in range(table.column_count):
                        if c >= len(table.grid[r]):
                            break

                        cell = table.grid[r][c]
                        if id(cell) not in seen:
                            cell.height -= n
                            seen.add(id(cell))
                            
                    for r_idx in range(r+1, len(table.grid)):
                        for c in range(table.column_count):
                            if c < len(table.grid[r_idx]):
                                below_cell = table.grid[r_idx][c]
                                if id(below_cell) not in seen:
                                    below_cell.y -= n
                                    seen.add(id(below_cell))
                r += 1

    @staticmethod
    def _identify_header(table:GridTable, columns:bool=True) -> List[int]:
        """
        Xác định danh sách các dòng có tỉ lệ cao là header của một bảng.
        Dựa trên `is_numeric`.
        """
        header_rows = []
            
        if columns:
            row_outliers = [0] * len(table.grid)
            numeric_col_index = []
            
            for c in range(table.column_count):
                accept, outliers = table.is_numeric(c, columns=True, acceptance=1/4)
                if accept:
                    numeric_col_index.append(c)
                    for cell in outliers:
                        last_row = min(cell.y + cell.height, len(table.grid))
                        for r in range(cell.y, last_row):
                            row_outliers[r] += 1
                        
            if len(numeric_col_index) == 0:      # Everything is text, surrender
                return header_rows

            mean = np.mean(row_outliers)
            if np.mean(row_outliers[:4]) >= mean:     # Nếu 4 dòng đầu có mật độ text cell cao hơn toàn bảng
                header_rows = [i for i, outliers_count in enumerate(row_outliers) if outliers_count > mean]     # Lấy toàn bộ các dòng có tỉ lệ text cao   

        if not columns or len(header_rows) == 0:
            ratios = [table.numeric_ratio(r, columns=False) for r in range(len(table.grid))]
        
            mean = np.mean(ratios)

            if np.mean(ratios[:4]) <= mean:     # Nếu 4 dòng đầu có mật độ text cell cao hơn toàn bảng
                header_rows = [i for i, rat in enumerate(ratios) if rat < mean]     # Lấy toàn bộ các dòng có tỉ lệ text cao   

        accept = 1.0 / table.column_count
        numeric_col = []
        for i, row in enumerate(header_rows):
            if table.is_numeric(row, columns=False, acceptance=accept):
                numeric_col.append(i)

        for i in reversed(numeric_col):
            header_rows.pop(i)

        return header_rows

    @staticmethod
    def fix_content(table:GridTable):
        """
        Xử lý chính tả và kiểu dữ liệu cho toàn bộ bảng.
        - Áp dụng symspell và numeric_translate cho mọi dòng.
        - Khôi phục colspan cho các token tách ra (nếu khớp).
        - Đánh dấu và lấp đầy các dòng lỗi (corrupted) bằng ô trống "-".
        """
        def _find_split_widths(cell:Cell, ref_rows:List[Cell]):
            if not ref_rows: return None
            split_widths = []
                
            ptr = cell.x
            while ptr < cell.x + cell.width:
                if len(ref_rows) <= ptr:
                    return None

                ref_cell = ref_rows[ptr]
                split_widths.append(ref_cell.width)
                ptr += ref_cell.width
            return split_widths

        # Sửa định dạng
        for r_idx, row in enumerate(table.grid):
            c_ptr = 0
            while c_ptr < len(row):
                cell = row[c_ptr]
                
                if isinstance(cell.content, str) and cell.y == r_idx:
                    # Áp dụng numeric_translate cho mọi dòng.
                    numbers = numeric_translate(cell.content)

                    if len(numbers) == 1:
                        cell.content = numbers[0]
                    elif len(numbers) > 1:
                        split_widths:List[int] = None

                        # Khôi phục colspan cho các token tách ra (nếu khớp).
                        if len(numbers) == cell.width:
                            split_widths = [1] * len(numbers)
                        else:
                            prev_row = table.grid[r_idx - 1] if r_idx > 0 else None
                            next_row = table.grid[r_idx + 1] if r_idx < len(table.grid) - 1 else None    
                                
                            for ref in [prev_row, next_row] :
                                split_widths = _find_split_widths(cell, ref)
                                if split_widths and (len(split_widths) != len(numbers) or sum(split_widths) != cell.width): 
                                    split_widths = None
                                    continue 
                                    
                        if split_widths:
                            for value, width in zip(numbers, split_widths):
                                new_cell = Cell(content=value, width=width, height=cell.height, x=c_ptr, y=cell.y)
                                ptr = c_ptr
                                while ptr < new_cell.x + new_cell.width:
                                    row[ptr] = new_cell
                                    ptr += 1

                c_ptr += cell.width

        # Đánh dấu và lấp đầy các dòng lỗi (corrupted) bằng ô trống "-".
        corrupted_rows:List[int] = []
        for r_idx in range(len(table.grid)):
            if table.is_corrupted_row(r_idx):
                corrupted_rows.append(r_idx)
                
        table.corrupted_rows = []
        
        for r_idx in corrupted_rows:
            row = table.grid[r_idx]
            table.corrupted_rows.append((r_idx, row))

            c_ptr = 0
            while c_ptr < len(row):
                if cell.height > 1:
                    span_rows = range(cell.y, cell.y + cell.height)
                    if any(r not in corrupted_rows for r in span_rows):
                        # This cell is flawless
                        c_ptr += cell.width
                        continue
                            
                cell.content = "-"
                c_ptr += cell.width

    @staticmethod
    def primary_key(table:GridTable) :
        text_col_index = []

        for c in range(table.column_count):
            accept, _ = table.is_numeric(c, columns=True, acceptance=1/4)
            if not accept:
                text_col_index.append(c)

        if len(text_col_index) == table.column_count:    # Everything is text
            return

        if len(text_col_index) == 0:    # Everything is number
            return 

        if min(text_col_index) != 0:    # First column is number, there are something wrong.
            return

        for c in text_col_index:
            table.primary_key.append(c)

class FlattenGrid:
    """
    Xử lý flatten index cho header và cột, nếu có.
    """
    @staticmethod
    def process(table: GridTable):
        FlattenGrid.fulfill(table)
        FlattenGrid.flatten_header(table)
        FlattenGrid.flatten_data(table)

    @staticmethod
    def fulfill(table: GridTable):
        if not table.column_count:
            return

        for y, row in enumerate(table.grid):
            row.extend([Cell(content="", y=y)] * (table.column_count - len(row)))

    @staticmethod
    def flatten_header(table: GridTable):
        """
        Duyệt qua từng dòng (trong giới hạn header_count) và ffill thành 1 dòng tổng hợp.
        """

        if table.header_count == 0:
            return

        new_header_row = []
        for c in range(table.column_count):
            parts = []
            r = 0
            while r < table.header_count:
                cell = table.grid[r][c]
                if not cell.is_empty():
                    content = str(cell.content).strip()
                    if content :
                        parts.append(content)
                r += cell.height
            
            merged_content = "_".join(parts) if parts else ""
            new_header_row.append(Cell(content=merged_content, width=1, height=1, x=c, y=0))
            
        table.grid = [new_header_row] + table.grid[table.header_count:]
        table.header_count = 1

    @staticmethod
    def flatten_data(table: GridTable):
        index_rows = []
        if table.column_count <= 1: return

        for r_idx in range(table.header_count, len(table.grid)):   # data section
            row = table.grid[r_idx]
            if not row: continue

            index_cell = row[0]

            if index_cell.height == 1 and index_cell.width > 1 and \
                index_cell.width in [table.column_count, table.column_count - 1]:
                
                index_rows.append(r_idx)
                
        if not index_rows:
            return
            
        active_cell = Cell(content="", width=1, height=1)
        
        for r_idx in range(len(table.grid)):
            row = table.grid[r_idx]
            
            if r_idx in index_rows:
                index_cell = row[0]
                active_cell = Cell(content=index_cell.content, width=1, height=1)
                row.insert(0, Cell(content="", width=1, height=1))
            else:
                row.insert(0, active_cell)
                
        # Xóa các dòng index từ dưới lên trên (reverse order) để không bị lệch index của list gốc
        for r_idx in reversed(index_rows):
            table.grid.pop(r_idx)

        # Cập nhật primary key
        for i in range(len(table.primary_key)):
            table.primary_key[i] += 1

        table.primary_key.append(0)            
        table.column_count += 1

#-----------------
# Save checkpoint (Interim data) and CSV
#-----------------

def _save_parsed_table(raw_table:Table, parsed_table: List[GridTable]) -> None:

    write_json(
        obj = {
            "raw": {
                line: str(tag) for line, tag in zip(raw_table.line, raw_table.html_table)
            },
            "parsed": [
                {
                    "header count": t.header_count,
                    "columns count": t.column_count,
                    "primary key": t.primary_key,
                    "header": [t.row_to_str(i) for i in range(t.header_count)],
                    "data": [t.row_to_str(i) for i in range(t.header_count, len(t.grid))],
                    "corrupted rows": {r_idx: t.row_to_str(r_idx) for r_idx, _ in (t.corrupted_rows or [])},
                } for t in parsed_table
            ],
            
        },
        path = INTERIM_DIR / raw_table.docs.get_id() / f"at_{raw_table.line[0]}.json"
    )

def _save_dataframe(table: Table, df: pd.DataFrame, post_fix:str=None) -> Path:
        line_no = table.line[0]
        filename = f"at_{line_no}"
        if post_fix:
            filename += post_fix

        csv_path = PROCESSED_DIR / "table_csv" / table.docs.get_id() / f"{filename}.csv"
        write_csv(df, csv_path)
        return csv_path


