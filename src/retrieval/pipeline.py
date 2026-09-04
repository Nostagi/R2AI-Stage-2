import pandas as pd
from typing import List
from pathlib import Path

from src.config import get_settings
from ..contracts.schemas import Document, Table
from .corpus import TableMetadataCorpus, TableFewShotCorpus
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .reranker import LLMReranker
from ..utils.logging import get_logger

LOGGER = get_logger("retrieval.pipeline")

class RetrievalPipeline:
    """
    Pipeline thực hiện quá trình Retrieval toàn diện:
    - Load các index (BM25, Dense).
    - Thực hiện multi-retrieval.
    - Áp dụng RRF (Reciprocal Rank Fusion).
    - Rerank kết quả bằng LLM.
    """

    def __init__(self, documents: List[Document] = None, processed_dir: Path = None, index_dir: Path = None):
        self.settings = get_settings()
        
        if documents:
            self.documents = documents
            # 1. Trích xuất toàn bộ tables và khởi tạo Corpus
            all_tables: List[Table] = []
            for doc in documents:
                all_tables.extend(doc.tables)
                
            self.metadata_corpus = TableMetadataCorpus()
            self.fewshot_corpus = TableFewShotCorpus()
            
            for table in all_tables:
                self.metadata_corpus.add(table)
                self.fewshot_corpus.add(table)
                
            # 2. Khởi tạo các Retriever
            self.retrievers = [
                BM25Retriever(corpus=self.metadata_corpus, index_name="bm25_metadata", index_dir=index_dir),
                BM25Retriever(corpus=self.fewshot_corpus, index_name="bm25_fewshot", index_dir=index_dir),
                DenseRetriever(corpus=self.metadata_corpus, index_name="dense_metadata", index_dir=index_dir),
                DenseRetriever(corpus=self.fewshot_corpus, index_name="dense_fewshot", index_dir=index_dir),
            ]
        else:
            # 2.1 Factory khôi phục từ ổ cứng
            from .build import load_all_indexes
            self.documents = []  # hoặc lấy từ factory nếu cần thiết, hiện tại chỉ cần retrievers
            self.retrievers = load_all_indexes(processed_dir=processed_dir, index_dir=index_dir)
        
        # 3. Tải index từ đĩa (chỉ khi khởi tạo từ documents, vì load_all_indexes đã load)
        if documents:
            LOGGER.progress("Đang nạp các chỉ mục (indexes)...")
            for retriever in self.retrievers:
                try:
                    retriever.load()
                except Exception as e:
                    LOGGER.warning(f"Không thể nạp index {retriever.index_name}, sẽ build lại: {e}")
                    retriever.build_index()

        self.reranker = LLMReranker(threshold=0.1)

    def search(self, query: str, top_k_retrieval: int = 10, top_k_rerank: int = 5) -> pd.DataFrame:
        """
        Thực hiện tìm kiếm cho một câu hỏi duy nhất (gọi qua search_batch).
        """
        return self.search_batch([query], top_k_retrieval, top_k_rerank)[0]

    def search_batch(self, queries: List[str], top_k_retrieval: int = 10, top_k_rerank: int = 5) -> List[pd.DataFrame]:
        """
        Thực hiện tìm kiếm cho một danh sách câu hỏi.
        Chạy đồng loạt các retrievers bằng ThreadPoolExecutor.
        Kiểm soát tải/hủy LLM tập trung để giảm overhead.
        """
        LOGGER.progress(f"Bắt đầu batch search cho {len(queries)} truy vấn...")
        import concurrent.futures
        
        # Bước 1: Multi-Retrieval song song
        # Tạo ma trận kết quả: retrieval_results[query_idx] = [df_retriever1, df_retriever2, ...]
        retrieval_results: List[List[pd.DataFrame]] = [[] for _ in range(len(queries))]
        
        def _run_retriever(q_idx: int, q_text: str, ret: object):
            LOGGER.detail(f"Đang retrieve q_idx={q_idx} bằng {ret.index_name}...")
            return q_idx, ret.retrieve(q_text, top_k=top_k_retrieval)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for q_idx, q_text in enumerate(queries):
                for retriever in self.retrievers:
                    futures.append(executor.submit(_run_retriever, q_idx, q_text, retriever))
                    
            for future in concurrent.futures.as_completed(futures):
                try:
                    q_idx, df = future.result()
                    if not df.empty:
                        retrieval_results[q_idx].append(df)
                except Exception as e:
                    LOGGER.warning(f"Lỗi trong quá trình retrieval song song: {e}")

        # Bước 2: Giải phóng tài nguyên bi-encoder (nếu Dense Retriever dùng tới)
        # Giả định DenseRetriever dùng bi-encoder
        for ret in self.retrievers:
            if isinstance(ret, DenseRetriever):
                LOGGER.detail(f"Tiến hành giải phóng tài nguyên {ret.alias}...")
                self.settings.llm.release_llm(ret.alias)

        # Bước 3: Reciprocal Rank Fusion (RRF)
        LOGGER.detail("Thực hiện Reciprocal Rank Fusion (RRF) cho toàn bộ batch...")
        fused_dfs = []
        for q_idx in range(len(queries)):
            fused_df = reciprocal_rank_fusion(retrieval_results[q_idx], k=60)
            if fused_df.empty:
                LOGGER.warning(f"Không tìm thấy kết quả phù hợp nào cho truy vấn {q_idx}.")
            fused_dfs.append(fused_df)

        # Lọc ra top_k_retrieval để đưa vào Rerank
        top_fused_dfs = [df.head(top_k_retrieval) if not df.empty else df for df in fused_dfs]
        
        # Bước 4: Reranking theo Batch
        LOGGER.detail("Tiến hành Rerank bằng LLM cho toàn bộ batch...")
        final_results = self.reranker.rerank_batch(queries, top_fused_dfs, top_k=top_k_rerank)
        
        LOGGER.progress(f"Batch Search hoàn tất cho {len(queries)} câu hỏi.")
        return final_results


if __name__ == "__main__":
    # Ví dụ cách sử dụng pipeline (Giả lập)
    print("Vui lòng khởi tạo documents và truyền vào RetrievalPipeline để test.")
    # pipeline = RetrievalPipeline(documents=my_documents)
    # result_df = pipeline.search("Cho tôi biết doanh thu năm 2022 của công ty ABC là bao nhiêu?")
    # print(result_df)
