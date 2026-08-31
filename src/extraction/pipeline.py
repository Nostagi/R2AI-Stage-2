from pathlib import Path
from typing import Iterator, Dict

from ..contracts.schemas import Document
from .ocr_parser import OCRParser
from ..utils.logging import get_logger

LOGGER = get_logger("ocr_pipeline")

class OCRExtractorPipeline:
    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.documents: Dict[str, Document] = {}

        self.parser = OCRParser()

    def run(self) -> Dict[str, Document]:
        """
        Khởi chạy tiến trình phân tích toàn bộ dữ liệu OCR.
        """
        LOGGER.progress("Bắt đầu khởi chạy OCR Parsing Pipeline...")
        
        for file_path in self._discover_files():
            # 1. Trích xuất metadata từ đường dẫn
            doc = self._extract_metadata_from_path(file_path)
            LOGGER.progress(f"Đang xử lý Document: {doc.doc_id}")
            
            # 2. Xử lý nội dung file stream
            LOGGER.detail(f"Bắt đầu xử lý Document: {doc.doc_id}")
            LOGGER.detail(f"Tiến hành parse Document: {doc.doc_id}")
            doc = self.parser.process_single_ocr_file(doc)
                
            # Lưu trữ Document
            self.documents[doc.doc_id] = doc
            
        LOGGER.progress(f"Hoàn tất! Đã nạp thành công {len(self.documents)} documents.")
        return self.documents

    def _discover_files(self) -> Iterator[Path]:
        """
        Quét thư mục để tìm tất cả các file .txt báo cáo tài chính.
        """
        statements_dir = self.dataset_dir / "financial_statements"
        if not statements_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {statements_dir}")
            
        yield from statements_dir.rglob("*.txt")

    def _extract_metadata_from_path(self, file_path: Path) -> Document:
        """
        Phân tích cấu trúc thư mục để lấy metadata cơ bản (chưa đụng tới nội dung file).
        Đường dẫn mẫu: .../financial_statements/HSG/2015/HSG_financial_statements_2015_separate/HSG_..._extracted.txt
        """
        parts = file_path.parts
        
        ticker = parts[-4]
        year_str = parts[-3]
        doc_id = parts[-2] # Lấy tên thư mục chứa file làm doc_id chuẩn
        
        # Suy luận loại báo cáo từ tên doc_id
        doc_id_lower = doc_id.lower()
        report_type = "Consolidated" if "consolidated" in doc_id_lower else \
                      "Separate" if "separate" in doc_id_lower else \
                      "Aggregated" if "aggregated" in doc_id_lower else "Other"
                      
        year = int(year_str) if year_str.isdigit() else None
        
        return Document(
            ticker      = ticker,
            doc_id      = doc_id,
            doc_path    = file_path,
            year        = year,
            report_type = report_type
        )




if __name__ == "__main__":
    from huggingface_hub import snapshot_download
    from src.config import get_settings

    settings = get_settings()
    settings.paths.ensure()

    # snapshot_download(
    #     repo_id=settings.dataset,
    #     repo_type="dataset",
    #     local_dir=settings.paths.raw,
    #     token=settings.hf_token
    # )
    print("Đã tải dataset thành công. Bắt đầu xử lý OCR...")

    pipeline = OCRExtractorPipeline(dataset_dir=settings.paths.raw)
    parsed_documents = pipeline.run()