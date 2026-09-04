import os
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import get_settings
from ..contracts.retriever import Corpus, BaseRetriever
from ..utils.logging import get_logger

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass


class DenseRetriever(BaseRetriever):
    """
    Dense Retriever sử dụng SentenceTransformers.
    """

    def __init__(self, corpus: Corpus, index_name: str = "dense_index", batch_size: int = 32, index_dir: Path = None):
        super().__init__(corpus)
        self.index_name = index_name
        self.logger = get_logger(f"dense.{index_name}")

        settings = get_settings()
        
        if index_dir is None:
            index_dir = settings.paths.index
            
        self.save_dir = index_dir / f"{self.index_name}.faiss"

        # Load hyperparams từ retrieval.yaml
        embed_cfg = settings.retrieval.get("embedding", {})
        self.batch_size = embed_cfg.get("batch_size", batch_size)
        self.alias = embed_cfg.get("alias", "bi-encoder")
        
        db_cfg = settings.retrieval.get("vectordb", {})
        self.backend = db_cfg.get("backend", "faiss")
        self.index_type = db_cfg.get("index_type", "IndexHNSWFlat")
        
        # Khởi tạo alias
        self.alias = embed_cfg.get("alias", "bi-encoder")
        
        # Chỉ check xem alias có tồn tại trong LLMProvider không
        bi_encoder_cfg = settings.llm._configs.get(self.alias)
        if not bi_encoder_cfg:
            raise ValueError(f"Không tìm thấy cấu hình '{self.alias}' trong configs/llm.yaml. Yêu cầu dùng LLMProvider.")
            
        self.logger.detail(f"DenseRetriever sẽ sử dụng Provider '{self.alias}' cho embedding.")
        
        self.index = None
        self.is_indexed = False

    def release_resources(self):
        """Giải phóng tài nguyên LLM embedding."""
        settings = get_settings()
        settings.llm.release_llm(self.alias)
        self.logger.detail("Đã gọi LLMProvider giải phóng bộ nhớ của bi-encoder.")

    def build_index(self) -> None:
        """Tính toán embeddings cho toàn bộ corpus."""
        self.logger.progress(f"Đang tính toán embeddings cho {len(self.corpus)} mục...")
        corpus_texts = [self.corpus.to_text(i) for i in range(len(self.corpus))]
        
        if not corpus_texts:
            self.index = None
            self.is_indexed = True
            return
            
        settings = get_settings()
        
        embeddings_list = []
        # Tiến hành chạy theo batch size
        for i in range(0, len(corpus_texts), self.batch_size):
            batch_texts = corpus_texts[i: i + self.batch_size]
            batch_embs = settings.llm.embed(self.alias, batch_texts)
            embeddings_list.extend(batch_embs)
            
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # Chuẩn hóa vector cho inner product = cosine
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        if self.backend == "faiss" and self.index_type == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(dim, 32)
            self.index.hnsw.efConstruction = 40
            self.index.add(embeddings)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            
        self.is_indexed = True
        self.save()

    def save(self) -> None:
        if self.index is not None:
            self.save_dir.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.save_dir))
            self.logger.progress(f"Đã lưu Dense FAISS index tại {self.save_dir}")

    def load(self) -> None:
        if not self.save_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy file index tại: {self.save_dir}")
        self.index = faiss.read_index(str(self.save_dir))
        self.is_indexed = True
        self.logger.progress(f"Đã tải thành công Dense FAISS index từ: {self.save_dir}")

    def retrieve(self, query: str, top_k: int = 10) -> pd.DataFrame:
        if not self.is_indexed:
            if self.save_dir.exists():
                self.load()
            else:
                self.build_index()
                
        if self.index is None or self.index.ntotal == 0:
            return pd.DataFrame()

        settings = get_settings()
        query_embedding = settings.llm.embed(self.alias, [query])[0]
            
        query_embedding = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Lấy top_k, FAISS hỗ trợ max top_k = ntotal
        k = min(top_k, self.index.ntotal)
        scores_arr, indices_arr = self.index.search(query_embedding, k)
        
        top_scores = scores_arr[0]
        top_indices = indices_arr[0]

        records = []
        for idx, score in zip(top_indices, top_scores):
            idx_int = int(idx)
            info = self.corpus.get_info(idx_int)
            info["corpus_id"] = idx_int
            info["score"] = float(score)
            records.append(info)

        df_results = pd.DataFrame(records)
        
        priority_cols = ["table_id", "doc_id", "title", "content", "corpus_id", "score"]
        existing_priority = [col for col in priority_cols if col in df_results.columns]
        other_cols = [col for col in df_results.columns if col not in existing_priority]
        
        return df_results[existing_priority + other_cols]
