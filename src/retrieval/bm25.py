import bm25s
import pandas as pd
from pathlib import Path

from src.config import get_settings
from ..contracts.retriever import Corpus, BaseRetriever
from ..utils.logging import get_logger
from ..utils.spell_check import tokenize


class BM25Retriever(BaseRetriever):
    """
    BM25 Retriever triển khai dựa trên thư viện BM25s.
    Hỗ trợ chế độ index / search bằng 'flat' (numpy/default) hoặc 'hnsw' (faiss-hnsw).
    """

    def __init__(self, corpus: Corpus, index_name: str = "bm25_index", index_dir: Path = None):
        self.corpus = corpus
        self.index_name = index_name
        self.logger = get_logger(f"bm25.{index_name}")

        settings = get_settings()
        
        if index_dir is None:
            index_dir = settings.paths.index
            
        self.save_dir = index_dir / f"{self.index_name}.pickle"
        
        # Load hyperparams từ retrieval.yaml
        bm25_cfg = settings.retrieval.get("bm25", {})
        k1 = float(bm25_cfg.get("k1", 1.5))
        b = float(bm25_cfg.get("b", 0.75))
        
        db_cfg = settings.retrieval.get("vectordb", {})
        self.backend_type = "hnsw" if db_cfg.get("backend") == "faiss" else "flat"

        # Cấu hình BM25s Model
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self.is_indexed = False

    def _resolve_backend(self) -> str:
        """Map cấu hình người dùng sang backend tương thích của BM25s."""
        if self.backend_type == "hnsw":
            return "faiss-hnsw"
        elif self.backend_type == "flat":
            return "numpy"
        return self.backend_type

    def build_index(self) -> None:
        """Xây dựng bộ chỉ mục BM25s trên toàn bộ văn bản trong Corpus và lưu file ngay lập tức."""
        self.logger.progress(f"Đang tiến hành tokenize và lập chỉ mục BM25s ({len(self.corpus)} mục)...")

        # 1. Trích xuất toàn bộ văn bản từ corpus
        corpus_texts = [self.corpus.to_text(i) for i in range(len(self.corpus))]

        # 2. Tokenize bằng Pyvi tokenizer chuyên dụng cho tiếng Việt
        corpus_tokens = [tokenize(text) for text in corpus_texts]

        # 3. Lập chỉ mục BM25s
        resolved_backend = self._resolve_backend()
        self.logger.detail(f"Khởi tạo BM25 index (backend sẽ được sử dụng khi search: {resolved_backend})")
        
        self.retriever.index(corpus_tokens)
        self.is_indexed = True

        # 4. Lưu lại bộ chỉ mục ngay lập tức
        self.save()

    def save(self) -> None:
        """Lưu bộ chỉ mục BM25s ra đĩa."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(self.save_dir))
        self.logger.progress(f"Đã lưu BM25s index thành công tại: {self.save_dir}")

    def load(self) -> None:
        """Tải bộ chỉ mục BM25s đã lưu từ đĩa."""
        if not self.save_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy file index tại: {self.save_dir}")
        
        self.retriever = bm25s.BM25.load(str(self.save_dir))
        self.is_indexed = True
        self.logger.progress(f"Đã tải thành công BM25s index từ: {self.save_dir}")

    def retrieve(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Thực hiện truy vấn BM25 và trả về top_k kết quả dưới dạng pandas DataFrame.
        """
        if not self.is_indexed:
            if self.save_dir.exists():
                self.load()
            else:
                self.build_index()

        # 1. Tokenize câu hỏi truy vấn
        query_tokens = [tokenize(query)]

        # 2. Tìm kiếm top_k bằng BM25s
        resolved_backend = self._resolve_backend()
        results, scores = self.retriever.retrieve(
            query_tokens, 
            k=min(top_k, len(self.corpus)), 
            backend_selection=resolved_backend
        )

        # 3. Đóng gói kết quả thành DataFrame sử dụng corpus.get_info()
        records = []
        top_indices = results[0]  # Lấy danh sách index của câu truy vấn đầu tiên
        top_scores = scores[0]

        for idx, score in zip(top_indices, top_scores):
            idx_int = int(idx)
            info = self.corpus.get_info(idx_int)
            info["corpus_id"] = idx_int
            info["score"] = float(score)
            records.append(info)

        df_results = pd.DataFrame(records)

        # Đảm bảo các cột quan trọng xuất hiện đầu tiên
        priority_cols = ["table_id", "doc_id", "title", "content", "corpus_id", "score", ]
        existing_priority = [col for col in priority_cols if col in df_results.columns]
        other_cols = [col for col in df_results.columns if col not in existing_priority]
        
        return df_results[existing_priority + other_cols]