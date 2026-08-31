"""Logging tập trung — mỗi module gọi get_logger(name)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGING_DIR: Path = Path("logs")


class ModuleLogger:
    """Class điều hướng logger hỗ trợ theo dõi tiến trình (Console) và ghi log chi tiết (File)."""

    def __init__(self, name: str, logging_dir: Path) -> None:
        self.name = name
        self.file_path = logging_dir / name
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        # 1. Terminal Logger dành cho Progress Tracking
        self._console_logger = logging.getLogger(f"{name}.progress")
        self._console_logger.setLevel(logging.INFO)
        self._console_logger.propagate = False
        if not self._console_logger.handlers:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            self._console_logger.addHandler(sh)

        # 2. File Logger dành cho thông tin chi tiết ({logging_dir}/{name})
        self._file_logger = logging.getLogger(f"{name}.detail")
        self._file_logger.setLevel(logging.DEBUG)
        self._file_logger.propagate = False
        if not self._file_logger.handlers:
            fh = logging.FileHandler(self.file_path, mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            self._file_logger.addHandler(fh)

    def progress(self, text: str) -> None:
        """Đưa thông điệp theo dõi tiến trình ra terminal."""
        self._console_logger.info(text)

    def detail(self, text: str) -> None:
        """Ghi lịch sử thao tác chi tiết vào file nhật ký riêng của module."""
        self._file_logger.debug(text)


def setup_logging(logging_dir: str | Path = "logs") -> None:
    """Khởi tạo thư mục chứa log và cấu hình bộ lọc cảnh báo."""
    global _LOGGING_DIR, _CONFIGURED
    _LOGGING_DIR = Path(logging_dir)
    _LOGGING_DIR.mkdir(parents=True, exist_ok=True)

    # Lọc bớt log từ các thư viện thứ ba
    for noisy in ("httpx", "urllib3", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str, logging_dir: str | Path | None = None) -> ModuleLogger:
    """Factory cấp phát một ModuleLogger sở hữu file_path ghi chi tiết riêng biệt."""
    dir_path = Path(logging_dir) if logging_dir else _LOGGING_DIR
    return ModuleLogger(name=name, logging_dir=dir_path)