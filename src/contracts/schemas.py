"""Kieu du lieu dung chung toan pipeline.

Mot file duy nhat thay cho folder dto/. Nhom theo stage:
  - Ingestion:    Document
  - Extraction:   Table
  - Submission:   Evidence, SubmissionItem 
  - Retrieval:    Question, RetrievedTable, RetrievalResult
  - Generation:   GeneratedQuery
  - Execution:    ExecutionResult

"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterable
from bs4.element import Tag

@dataclass
class Table:
    """
    Represents a single table extracted from the document.
    Metadata is flattened directly into this class.
    """
    # Core Data
    docs: Document
    line: List[int]                     # Table line in OCR document (crucial for submission ID)
    csv_path: Path = None              # Local path to the extracted CSV file
    
    # metadata
    title: Optional[str] = None          # Table caption/title
    description: Optional[str] = None
    
    company: Optional[str] = None        # e.g., "VCB"
    year: Optional[int] = None           # e.g., 2023
    report_type: Optional[str] = None    # e.g., "Consolidated"
    statement: Optional[str] = None      # e.g., "Balance Sheet"

    # raw text
    pre_text: Optional[str] = field(default=None, repr=False)        # last 100 words before the table content
    post_text: Optional[str] = field(default=None, repr=False)      # First 100 words after the table content

    html_table: Optional[List[Tag]] = field(default=None, repr=False)     # Table content (HTML tag) before converting to csv

    def get_id(self) -> str:
        """Generates the required string format for 'relevant_tables'."""
        return [f"{self.docs.doc_id}|{line}" for line in self.line]

@dataclass
class Document:
    """
    Represents a full financial document (e.g., a PDF converted to text).
    Metadata is flattened directly into the document object.
    """
    ticker: str                # Group identifier for docs from the same source
    doc_id: str                # Filename without .txt (e.g., "AAA_financial_statements_2015_consolidated")
    doc_path: Optional[Path]
    
    # Flattened Metadata
    company: Optional[str] = None
    year: Optional[int] = None
    report_type: Optional[str] = None
    
    tables: List[Table] = field(default_factory=list)

    def get_id(self) -> str :
        return self.doc_id

    def to_dict(self, root_dir: Path = None) -> dict[str, Any]:
        """Chuyển đổi Document và Tables sang dict để ghi JSON."""
        if root_dir is None:
            from src.config import get_settings
            root_dir = get_settings().paths.root
        
        tables_data = []
        for t in self.tables:
            csv_path_str = ""
            if t.csv_path:
                try:
                    csv_path_str = str(t.csv_path.relative_to(root_dir)).replace("\\", "/")
                except ValueError:
                    csv_path_str = str(t.csv_path).replace("\\", "/")
                    
            tables_data.append({
                "line": t.line,
                "csv_path": csv_path_str,
                "title": t.title,
                "description": t.description,
                "company": t.company,
                "year": t.year,
                "report_type": t.report_type,
                "statement": t.statement,
                "pre_text": t.pre_text,
                "post_text": t.post_text
            })
            
        return {
            "doc_id": self.doc_id,
            "ticker": self.ticker,
            "company": self.company,
            "year": self.year,
            "report_type": self.report_type,
            "tables": tables_data
        }

    @classmethod
    def from_json(cls, json_path: Path, root_dir: Path = None) -> "Document":
        """Tái cấu trúc Document và Tables từ file JSON."""
        import json
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if root_dir is None:
            from src.config import get_settings
            root_dir = get_settings().paths.root
        
        doc = cls(
            ticker=data.get("ticker", ""),
            doc_id=data.get("doc_id", ""),
            doc_path=None,
            company=data.get("company"),
            year=data.get("year"),
            report_type=data.get("report_type")
        )
        
        for t_data in data.get("tables", []):
            csv_path_str = t_data.get("csv_path")
            csv_path = None
            if csv_path_str:
                csv_path = root_dir / csv_path_str
                    
            table = Table(
                docs=doc,
                line=t_data.get("line", []),
                csv_path=csv_path,
                title=t_data.get("title"),
                description=t_data.get("description"),
                company=t_data.get("company"),
                year=t_data.get("year"),
                report_type=t_data.get("report_type"),
                statement=t_data.get("statement"),
                pre_text=t_data.get("pre_text"),
                post_text=t_data.get("post_text")
            )
            doc.tables.append(table)
            
        return doc

# ──────────────────────────────────────────────────────────
# SUBMISSION
# ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class Evidence:
    variable: str                      # ten bien DataFrame dung trong pandas_query
    csv_path: str                      # phai bat dau bang "data/"

    def to_dict(self) -> dict[str, str]:
        return {"variable": self.variable, "csv_path": self.csv_path}


@dataclass(slots=True)
class SubmissionItem:
    id: int
    question: str
    answer: float
    relevant_docs: list[Document] = field(default_factory=list)
    relevant_tables: list[Table] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    pandas_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "question": self.question,
            "answer": float(self.answer),
            "relevant_docs": list([doc.get_id() for doc in self.relevant_docs]),
            "relevant_tables": list([table.get_id() for table in self.relevant_tables]),
            "evidence": [e.to_dict() for e in self.evidence],
            "pandas_query": self.pandas_query,
        }
