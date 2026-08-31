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
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Iterable

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

    html_table: Optional[List[str]] = field(default=None, repr=False)     # Table content (HTML tag) before converting to csv

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
    
    tables: Dict[int, Table] = field(default_factory=dict) # Mapping from line to corresponding table

    def get_id(self) -> str :
        return self.doc_id

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
