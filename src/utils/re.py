
from typing import List
import re

def truncate_text(lines: List[str], max_words: int = 200, keep_last: bool = False) -> str:
    """
    Nối các dòng và cắt đúng số lượng từ (word) mà hoàn toàn không làm mất dấu xuống dòng (\n).
    """
    if not lines:
        return ""
        
    text = "\n".join(lines)
    # Tìm tất cả các từ (chuỗi ký tự không phải khoảng trắng)
    matches = list(re.finditer(r'\S+', text))
        
    if len(matches) <= max_words:
        return text.strip()
            
    if keep_last:
        # Lấy từ word thứ (âm max_words) trở về cuối (Dùng cho pre_text)
        start_idx = matches[-max_words].start()
        return text[start_idx:].strip()        
    else:            
        # Lấy từ đầu đến hết word thứ max_words (Dùng cho post_text)
        end_idx = matches[max_words - 1].end()
        return text[:end_idx].strip()