"""Load configs/*.yaml + .env thanh mot object Settings duy nhat.

Moi module khac chi import `get_settings()`, khong tu doc YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.contracts.llm import LLMProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Thieu file config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Paths:
    raw: Path
    interim: Path
    processed: Path
    index: Path
    questions: Path
    outputs: Path
    labels: Path
    logging: Path
    models: Path

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.index,
                 self.outputs, self.labels, self.models):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    root: Path
    paths: Paths
    corpus: dict[str, Any]
    llm: LLMProvider
    execution: dict[str, Any]
    submission: dict[str, Any]
    retrieval: dict[str, Any] = field(default_factory=dict)

    # Secrets tu .env — khong bao gio de trong YAML
    hf_token: str | None = None
    dataset:str | None = None

    def raw_value(self, dotted: str, default: Any = None) -> Any:
        """Truy cap sau, vd: settings.raw_value("retrieval.bm25.k1")."""
        node: Any = {
            "corpus": self.corpus, "llm": self.llm, "execution": self.execution,
            "submission": self.submission, "logging": self.logging,
            "retrieval": self.retrieval,
        }
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    main = _read_yaml(CONFIG_DIR / "config.yaml")
    llm = _read_yaml(CONFIG_DIR / "llm.yaml")
    retrieval = _read_yaml(CONFIG_DIR / "retrieval.yaml")

    raw_paths = main.get("paths", {})
    paths = Paths(**{k: PROJECT_ROOT / v for k, v in raw_paths.items()})

    return Settings(
        root=PROJECT_ROOT,
        paths=paths,
        corpus=main.get("corpus", {}),
        execution=main.get("execution", {}),
        submission=main.get("submission", {}),
        llm=LLMProvider(llm),
        retrieval=retrieval,
        hf_token=main.get("HF_token", None),
        dataset=main.get("dataset", None)
    )
