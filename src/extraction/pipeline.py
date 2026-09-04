import json
from pathlib import Path
from typing import Iterator, Dict

from ..contracts.schemas import Document
from .ocr_parser import OCRParser
from .ocr_fixer import OCRTableFixerPipeline
from .table_describe import TableDescribePipeline
from ..utils.logging import get_logger
from ..utils.io import read_csv, write_json
from ..config import get_settings

LOGGER = get_logger("ocr_pipeline")

df = read_csv(get_settings().paths.raw / "code_stock.csv")
CODE_STOCK = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

class OCRExtractorPipeline:
    def __init__(self, process_dir: str | Path | None = None):
        from ..config import get_settings
        settings = get_settings()
        
        self.process_dir = Path(process_dir) if process_dir else settings.paths.raw
        self.interim_dir = settings.paths.interim
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        
        self.doc_processed_dir = settings.paths.processed / "document"
        self.doc_processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.table_csv_dir = settings.paths.processed / "table_csv"
        self.table_csv_dir.mkdir(parents=True, exist_ok=True)

        self.documents: Dict[str, Document] = {}

        self.parser = OCRParser()
        self.fixer = OCRTableFixerPipeline()
        self.describer = TableDescribePipeline()

    def run(self) -> Dict[str, Document]:
        """
        Khởi chạy tiến trình phân tích toàn bộ dữ liệu OCR.
        """
        LOGGER.progress("Bắt đầu khởi chạy OCR Parsing Pipeline...")
        
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for file_path in self._discover_files():
                # 1. Trích xuất metadata từ đường dẫn
                doc = self._extract_metadata_from_path(file_path)
                LOGGER.progress(f"Đang trích xuất metadata cho Document: {doc.doc_id}")
                
                # 2. Xử lý nội dung file stream
                LOGGER.detail(f"Bắt đầu xử lý Document: {doc.doc_id}")

                LOGGER.detail(f"Tiến hành parse Document: {doc.doc_id}")
                doc = self.parser.process_single_ocr_file(doc)
                    
                # 3. Chạy qua Fixer Pipeline để chuẩn hóa bảng
                LOGGER.detail(f"Tiến hành fix table cho Document: {doc.doc_id}")
                doc = self.fixer.process_document(doc)

                # Lưu trữ Document vào bộ nhớ
                self.documents[doc.doc_id] = doc
                
                # 4 & 5. Submit describe và lưu JSON vào background thread
                futures.append(executor.submit(self._describe_and_save, doc))
                
            LOGGER.progress("Đang chờ LLM hoàn tất quá trình mô tả bảng...")
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    LOGGER.warning(f"Lỗi trong background task: {e}")
            
        LOGGER.progress(f"Hoàn tất! Đã nạp thành công {len(self.documents)} documents.")
        return self.documents

    def _describe_and_save(self, doc: Document):
        """
        Background task: Chạy mô tả bảng (LLM) sau đó lưu file JSON.
        """
        LOGGER.detail(f"Tiến hành describe Document: {doc.doc_id}")
        doc = self.describer.process_document(doc)
        
        self._save_document_metadata(doc)
        LOGGER.progress(f"Đã lưu JSON cho Document: {doc.doc_id}")

    def _discover_files(self) -> Iterator[Path]:
        """
        Quét thư mục để tìm tất cả các file .txt báo cáo tài chính.  
        """
        statements_dir = self.process_dir / "financial_statements"
        if not statements_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {statements_dir}")
            
        yield from statements_dir.rglob("*.txt")

    def _save_document_metadata(self, doc: Document):
        """
        Lưu metadata của document và danh sách các table vào JSON.
        Sắp xếp doc.tables theo line trước khi lưu và loại trừ html_table.
        """
        # Sắp xếp lại doc.tables theo line trước khi in
        doc.tables.sort(key=lambda t: t.line)
        
        # Sử dụng logic đóng gói từ class Document
        doc_dict = doc.to_dict()
            
        out_path = self.doc_processed_dir / f"{doc.doc_id}.json"
        write_json(doc_dict, out_path)

    def _extract_metadata_from_path(self, file_path: Path) -> Document:
        """
        Phân tích cấu trúc thư mục để lấy metadata cơ bản (chưa đụng tới nội dung file).
        Đường dẫn mẫu: .../financial_statements/HSG/2015/HSG_financial_statements_2015_separate/HSG_..._extracted.txt
        """
        parts = file_path.parts
        
        ticker = parts[-4]
        company = CODE_STOCK[ticker]
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
            company     = company,
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

    snapshot_download(
        repo_id=settings.dataset,
        repo_type="dataset",
        local_dir=settings.paths.raw,
        token=settings.hf_token
    )
    print("Đã tải dataset thành công. Bắt đầu xử lý OCR...")

    pipeline = OCRExtractorPipeline(process_dir=settings.paths.raw)
    parsed_documents = pipeline.run()