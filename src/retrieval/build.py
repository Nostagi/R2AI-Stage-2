from typing import List, Dict
import os
import json
from pathlib import Path

from src.config import get_settings
from ..contracts.schemas import Document, Table
from .corpus import TableMetadataCorpus, TableFewShotCorpus
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from ..utils.logging import get_logger
from ..utils.io import write_json

LOGGER = get_logger("retrieval.build")

def build_all_indexes(documents: List[Document], index_dir: Path = None) -> None:
    """
    Xây dựng toàn bộ chỉ mục (BM25, Dense) trên TableMetadataCorpus và TableFewShotCorpus.
    Đầu vào: Danh sách các Document đã bóc tách.
    """
    settings = get_settings()
    if index_dir is None:
        index_dir = settings.paths.index
        
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Thu thập tất cả các bảng từ Documents với thứ tự đồng nhất
    documents.sort(key=lambda d: d.doc_id)
    all_tables: List[Table] = []
    for doc in documents:
        doc.tables.sort(key=lambda t: t.line)
        all_tables.extend(doc.tables)
        
    LOGGER.progress(f"Tổng hợp được {len(all_tables)} tables từ {len(documents)} documents. Bắt đầu build index...")
    
    # 2. Khởi tạo các corpus
    metadata_corpus = TableMetadataCorpus()
    fewshot_corpus = TableFewShotCorpus()
    
    for table in all_tables:
        metadata_corpus.add(table)
        fewshot_corpus.add(table)
        
    # 3. Khởi tạo danh sách các Retrievers
    retrievers = [
        # BM25 trên Metadata
        BM25Retriever(corpus=metadata_corpus, index_name="bm25_metadata", index_dir=index_dir),
        # BM25 trên FewShot
        BM25Retriever(corpus=fewshot_corpus, index_name="bm25_fewshot", index_dir=index_dir),
        # Dense trên Metadata
        DenseRetriever(corpus=metadata_corpus, index_name="dense_metadata", index_dir=index_dir),
        # Dense trên FewShot
        DenseRetriever(corpus=fewshot_corpus, index_name="dense_fewshot", index_dir=index_dir),
    ]
    
    # 4. Build index
    for retriever in retrievers:
        LOGGER.detail(f"Đang tiến hành build index cho: {retriever.index_name}...")
        try:
            retriever.build_index()
        except Exception as e:
            LOGGER.warning(f"Lỗi khi build index {retriever.index_name}: {e}")
            
    # 5. Lưu lại metadata cấu hình vào index directory
    config_data = {
        "num_documents": len(documents),
        "num_tables": len(all_tables),
        "indexes": [r.index_name for r in retrievers],
        "status": "success"
    }
    
    config_path = index_dir / "retrieval_config.json"
    write_json(config_data, config_path)
    
    LOGGER.progress(f"Hoàn tất quá trình lập chỉ mục! Đã lưu cấu hình tại {config_path}")


def load_all_indexes(processed_dir: Path = None, index_dir: Path = None) -> List:
    """
    Factory khôi phục toàn bộ các Retrieval Indexes đã lưu.
    Sẽ nạp lại Document/Tables từ ổ cứng và liên kết tới Retrievers.
    Trả về danh sách các Retrievers (BaseRetriever).
    """
    settings = get_settings()
    
    if processed_dir is None:
        processed_dir = settings.paths.processed
        
    LOGGER.progress("Đang đọc dữ liệu JSON Documents từ đĩa...")
    documents = []
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Thư mục processed không tồn tại: {processed_dir}")
        
    for json_path in processed_dir.glob("*.json"):
        doc = Document.from_json(json_path)
        documents.append(doc)
        
    # Đảm bảo đúng order khi built
    documents.sort(key=lambda d: d.doc_id)
    
    all_tables: List[Table] = []
    for doc in documents:
        doc.tables.sort(key=lambda t: t.line)
        all_tables.extend(doc.tables)
        
    LOGGER.detail(f"Đã nạp {len(documents)} documents và {len(all_tables)} tables.")
    
    # Khôi phục corpus
    metadata_corpus = TableMetadataCorpus()
    fewshot_corpus = TableFewShotCorpus()
    
    for table in all_tables:
        metadata_corpus.add(table)
        fewshot_corpus.add(table)
        
    # Khôi phục Retrievers
    retrievers = [
        BM25Retriever(corpus=metadata_corpus, index_name="bm25_metadata", index_dir=index_dir),
        BM25Retriever(corpus=fewshot_corpus, index_name="bm25_fewshot", index_dir=index_dir),
        DenseRetriever(corpus=metadata_corpus, index_name="dense_metadata", index_dir=index_dir),
        DenseRetriever(corpus=fewshot_corpus, index_name="dense_fewshot", index_dir=index_dir),
    ]
    
    LOGGER.progress("Tiến hành nạp index vector/bm25 từ đĩa...")
    for retriever in retrievers:
        try:
            retriever.load()
        except Exception as e:
            LOGGER.warning(f"Lỗi tải index {retriever.index_name}, vui lòng build lại: {e}")
            
    return retrievers
