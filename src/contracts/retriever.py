import abc
from typing import Any, Dict, List
import pandas as pd


class Corpus(abc.ABC):
    """Abstract Base Class cho Corpus, định nghĩa các giao diện bắt buộc."""

    @abc.abstractmethod
    def add(self, item: Any) -> int:
        """Thêm 1 phần tử và trả về index ID."""
        pass

    @abc.abstractmethod
    def get(self, index: int) -> Any:
        """Lấy phần tử gốc theo index ID."""
        pass

    @abc.abstractmethod
    def get_batch(self, indices: List[int]) -> List[Any]:
        """Lấy danh sách phần tử gốc theo danh sách index IDs."""
        pass

    @abc.abstractmethod
    def get_info(self, index: int) -> Dict[str, Any]:
        """Trả về thông tin dạng Dict rút gọn để đóng gói vào DataFrame kết quả."""
        pass

    @abc.abstractmethod
    def to_text(self, index: int) -> str:
        """Chuyển đổi dữ liệu tại index thành chuỗi văn bản phục vụ lập chỉ mục BM25."""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Tổng số lượng phần tử trong Corpus."""
        pass


class BaseRetriever(abc.ABC):
    """
    Interface chung cho mọi phương pháp Retrieval (BM25s, Dense, Hybrid) và Reranker.
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    @abc.abstractmethod
    def retrieve(self, query: Any, top_k: int = 10) -> pd.DataFrame:
        """
        Thực hiện tìm kiếm và trả về top_k kết quả cao điểm nhất.
        
        Args:
            query (Any): Truy vấn đầu vào (thường là str, hoặc object chứa query phức tạp).
            top_k (int): Số lượng kết quả trả về.
            
        Returns:
            pd.DataFrame: DataFrame chứa các kết quả. 
                          Khuyến nghị bao gồm các cột cơ bản: ['doc_id', 'score', 'content'].
                          Việc lấy 'content' sẽ được gọi thông qua self.corpus.
        """
        pass