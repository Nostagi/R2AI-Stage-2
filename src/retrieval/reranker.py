import pandas as pd
import numpy as np
import re
from typing import List
from ..contracts.schemas import Table
from src.config import get_settings
from ..utils.logging import get_logger


class LLMReranker:
    """
    Reranker module sử dụng LLM để chấm điểm (relevance score).
    """

    def __init__(self, threshold: float = 0.5):
        self.logger = get_logger("reranker")
        self.settings = get_settings()
        self.llm = self.settings.llm
        
        # Đọc cấu hình từ retrieval.yaml
        rerank_cfg = self.settings.retrieval.get("rerank", {})
        self.alias = rerank_cfg.get("alias", "reranker")
        self.threshold = threshold
        
        # Thêm prompt động vào configs nếu chưa có
        if "rerank" not in self.llm.prompts:
            self.llm.prompts["rerank"] = {
                "system_prompt": "You are a relevance ranking expert. Given a query and a table data, evaluate how relevant the table is to the query. Output a single numerical score between 0.0 and 100.0 representing the relevance. Do not output any other text.",
                "user_prompt": "Query: {query}\n\nTable Content: {content}\n\nRelevance score (0-100):"
            }

    def _extract_score(self, text: str) -> float:
        """Trích xuất điểm số từ phản hồi của LLM."""
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if match:
            try:
                score = float(match.group())
                return score
            except ValueError:
                pass
        return 0.0

    def rerank(self, query: str, df_results: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Rerank dataframe kết quả bằng LLM.
        Lấy điểm thuần, min-max scale và lọc qua threshold.
        """
        if df_results.empty:
            return df_results
            
        self.logger.progress(f"Tiến hành rerank {len(df_results)} kết quả...")
        
        scores = []
        for _, row in df_results.iterrows():
            content = row.get("content", row.get("fewshot", ""))
            if not content:
                content = str(row.to_dict())
                
            template_kwargs = {
                "query": query,
                "content": str(content)[:2000] # Giới hạn token
            }
            
            try:
                # LLM chấm điểm
                response = self.llm.generate(
                    alias=self.alias,
                    prompt_name="rerank",
                    template_kwargs=template_kwargs,
                    max_tokens=10,
                    temperature=0.0
                )
                raw_score = self._extract_score(response)
            except Exception as e:
                self.logger.warning(f"Lỗi khi LLM rerank: {e}")
                raw_score = 0.0
                
            scores.append(raw_score)

        df_results["raw_score"] = scores
        
        # Min-Max Scaling
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 1
        
        if max_score > min_score:
            df_results["scaled_score"] = (df_results["raw_score"] - min_score) / (max_score - min_score)
        else:
            df_results["scaled_score"] = 0.0 if max_score == 0 else 1.0

        # Lọc theo threshold và lấy top_k
        df_filtered = df_results[df_results["scaled_score"] >= self.threshold].copy()
        
        if df_filtered.empty:
            self.logger.warning("Không có kết quả nào vượt qua threshold sau khi rerank. Trả về top kết quả gốc.")
            df_filtered = df_results.sort_values(by="scaled_score", ascending=False).head(top_k)
        else:
            df_filtered = df_filtered.sort_values(by="scaled_score", ascending=False).head(top_k)
            
        return df_filtered.reset_index(drop=True)

    def rerank_batch(self, queries: List[str], dfs: List[pd.DataFrame], top_k: int = 5) -> List[pd.DataFrame]:
        """
        Thực hiện rerank hàng loạt các truy vấn để tối ưu hóa việc nạp và giải phóng tài nguyên LLM.
        Sau khi rerank xong toàn bộ batch, mô hình sẽ được giải phóng khỏi VRAM/RAM.
        """
        if len(queries) != len(dfs):
            raise ValueError("Số lượng queries phải bằng số lượng DataFrames kết quả.")
            
        self.logger.progress(f"Bắt đầu chạy batch rerank cho {len(queries)} truy vấn...")
        
        final_results = []
        try:
            # LLM sẽ được lazy-load ở lần generate đầu tiên trong vòng lặp và giữ nguyên trên RAM
            for q, df in zip(queries, dfs):
                res_df = self.rerank(q, df, top_k=top_k)
                final_results.append(res_df)
        finally:
            # Sau khi xong batch, chủ động gọi giải phóng tài nguyên
            self.logger.detail(f"Đã hoàn thành batch. Tiến hành giải phóng tài nguyên LLM '{self.alias}'...")
            self.llm.release_llm(self.alias)
            
        return final_results

