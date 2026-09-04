import pandas as pd
from typing import List

def reciprocal_rank_fusion(results: List[pd.DataFrame], k: int = 60) -> pd.DataFrame:
    """
    Kết hợp nhiều kết quả truy vấn (DataFrames) bằng thuật toán Reciprocal Rank Fusion (RRF).
    Hợp nhất các dataframe, tính rank cho từng kết quả và thêm cột rrf_score.
    Vẫn giữ lại các cột điểm số ban đầu của các retriever (được đổi tên để không trùng lặp).
    """
    if not results:
        return pd.DataFrame()
        
    combined_scores = {}
    metadata_map = {}
    
    # Duyệt qua từng dataframe kết quả
    for idx, df in enumerate(results):
        if df.empty:
            continue
            
        # Đổi tên cột score để phân biệt giữa các retriever
        retriever_name = f"retriever_{idx+1}"
        
        # Sắp xếp lại theo điểm số từ cao xuống thấp (nếu chưa)
        df_sorted = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        
        for rank, row in df_sorted.iterrows():
            # Sử dụng corpus_id hoặc doc_id làm khóa chính
            key = row.get("corpus_id")
            if key is None:
                # Fallback nếu không có corpus_id
                key = str(row.get("doc_id", "")) + "_" + str(row.get("line", ""))
                
            if key not in combined_scores:
                combined_scores[key] = 0.0
                metadata_map[key] = row.to_dict()
                
            # Cập nhật điểm RRF
            combined_scores[key] += 1.0 / (k + rank + 1)
            
            # Lưu lại điểm gốc của retriever này
            metadata_map[key][f"score_{retriever_name}"] = row["score"]

    if not metadata_map:
        return pd.DataFrame()
        
    # Tạo dataframe mới từ dictionary
    fused_records = []
    for key, rrf_score in combined_scores.items():
        record = metadata_map[key]
        record["rrf_score"] = rrf_score
        fused_records.append(record)
        
    df_fused = pd.DataFrame(fused_records)
    
    # Sắp xếp theo RRF score giảm dần
    df_fused = df_fused.sort_values(by="rrf_score", ascending=False).reset_index(drop=True)
    
    return df_fused
