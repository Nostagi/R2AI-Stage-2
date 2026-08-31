# R2AI Stage 2 — Financial Table Retrieval & Text-to-Pandas

Dự án này là giải pháp toàn diện cho **Stage 2** của cuộc thi **ViFinQA (Financial Table Retrieval & Text-to-Pandas)**. Mục tiêu của hệ thống là tự động trả lời các câu hỏi tài chính tiếng Việt dựa trên kho Báo cáo tài chính (BCTC) của các doanh nghiệp niêm yết bằng cách:
1. **Truy hồi chính xác bảng số liệu mục tiêu** từ hàng trăm nghìn bảng BCTC phức tạp.
2. **Sinh mã Pandas (Text-to-Pandas)** thông qua mô hình ngôn ngữ lớn (LLM).
3. **Thực thi mã an toàn trong Sandbox AST** kèm vòng lặp tự sửa lỗi (**Self-Correction Loop**).
4. **Chuẩn hóa đơn vị và đóng gói tự động** file `submission.zip` đạt 100% chuẩn quy định của Ban tổ chức.

---

## 1. Kiến trúc Hệ thống

Hệ thống được thiết kế theo mô hình xử lý theo đợt (**Batch Pipeline**) chịu lỗi cao:

```
[00. Raw BCTC .txt] ──► 01. Corpus Pipeline (Table Stitching ghép bảng gãy + Prefix phân cấp)
                                  │
                                  ▼
[02. Index Pipeline] ──► BM25 + Dense Vectors (BGE-M3) + Lọc BCTC Mẹ vs Hợp nhất
                                  │
                                  ▼
[03. Inference Pipeline] ──► Hybrid Retrieval ──► Few-Shot CoT Prompt ──► LLM (Qwen2.5-Coder-7B-Instruct)
                                  │                                                │
                                  ▼                                                ▼
                             Checkpointing ◄── Pandas Sandbox AST ◄── Self-Repair (Tối đa 3 lượt)
                                  │
                                  ▼
[04. Packaging Pipeline] ──► Validate Schema 100% ──► Đóng gói submission.zip
                                  │
                                  ▼
[05. Evaluation Pipeline] ──► Đo lường 3 trục
```

---

## 2. Cấu trúc Thư mục Dự án

```
R2AI-Stage-2/
├── main.py                   # Điểm điều phối toàn bộ pipeline (chạy 1 lệnh hoặc từng stage)
├── requirements.txt          # Danh sách thư viện cần thiết (Torch CUDA, Transformers, vLLM...)
├── .env.example              # Mẫu cấu hình biến môi trường (HF_TOKEN, LLM_API_KEY)
├── configs/                  # Thư mục cấu hình hệ thống
│   ├── config.yaml           # Cấu hình đường dẫn, model LLM, vLLM, và tham số sandbox
│   ├── retrieval.yaml        # Tham số cho BM25, Dense Vector, Selector và Hard Filters
│   └── prompts/              # File prompt mẫu sinh code Pandas và tự sửa lỗi
├── src/                      # Mã nguồn lõi của hệ thống
│   ├── ingestion/            # Bộ quét và bóc tách BCTC thô
│   ├── extraction/           # Xử lý bảng HTML, sửa lỗi dính chữ/số OCR, ghép bảng gãy
│   ├── normalization/        # Chuẩn hóa cấu trúc dọc (Long CSV), số liệu VNĐ và đơn vị
│   ├── embeddings/           # Tạo Table Cards và quản lý Embedding Model (BGE-M3)
│   ├── vectordb/             # Lưu trữ và truy vấn BM25, FAISS Vector và MetadataStore
│   ├── retrieval/            # Phân tích câu hỏi, áp bộ lọc cứng và bộ chọn bảng TableSelector
│   ├── generation/           # Chuẩn bị context và gọi LLM sinh code Pandas
│   ├── execution/            # Sandbox AST an toàn và module tự sửa lỗi (SelfRepairExecutor)
│   ├── submission/           # Đóng gói và kiểm tra tính hợp lệ của bài nộp
│   ├── evaluation/           # Bộ đánh giá toàn diện 3 trục (Doc F2, Table F2, Answer Accuracy)
│   └── pipeline/             # Các lớp điều phối toàn trình (Corpus, Index, Answer)
├── notebooks/                # Jupyter Notebooks điều phối
│   ├── pipeline_runner_colab.ipynb   # Master Notebook cho Google Colab (NVMe + Drive Backup)
│   └── pipeline_runner_kaggle.ipynb  # Master Notebook cho Kaggle GPU T4 (/kaggle/working)
├── data/                     # Thư mục dữ liệu (được loại trừ khỏi Git)
│   ├── raw/                  # File văn bản BCTC gốc của BTC (.txt)
│   ├── processed/            # 119,045 File CSV bảng số liệu chuẩn hóa
│   ├── index/                # Chỉ mục BM25, FAISS Vector và manifest.jsonl
│   └── questions/            # Tập câu hỏi kiểm thử (questions.jsonl) & mã CK (code_stock.csv)
├── labels/                   # Tập nhãn chuẩn gold.json để eval local (được giữ trong Git)
└── outputs/                  # Thư mục lưu kết quả dự đoán và ZIP bài nộp
```

---

## 3. Cài đặt & Hướng dẫn Vận hành trên 3 Môi trường

Dự án được thiết kế để hoạt động mượt mà trên cả 3 môi trường: **Local**, **Google Colab**, và **Kaggle**.

### Cách Clone & Khôi phục Dữ liệu (Dành cho Repo không chứa data thô)
Khi bạn hoặc người dùng khác clone repo từ GitHub:
```bash
git clone https://github.com/Nostagi/R2AI-Stage-2.git
cd R2AI-Stage-2
```
Thư mục `data/` nặng (>100,000 file CSV) được loại trừ khỏi Git để tối ưu dung lượng repo. Bạn có 2 cách để nạp dữ liệu:
* **Cách 1 (Nhanh nhất - Khuyên dùng):** Sử dụng file `data_backup.zip` đã được đóng gói sẵn. Chỉ cần giải nén file zip này vào thư mục `data/` của dự án.
* **Cách 2 (Xây dựng từ đầu):** Chạy `python main.py fetch` $\rightarrow$ `python main.py corpus` $\rightarrow$ `python main.py index` để hệ thống tự tải BCTC gốc từ HuggingFace và bóc tách tự động.

---

### 3.1. Chạy trên Google Colab (GPU T4 / A100)
Mở file [notebooks/pipeline_runner_colab.ipynb](notebooks/pipeline_runner_colab.ipynb) trên Google Colab:
1. **Mục 1 (Môi trường):** Mount Google Drive, đồng bộ mã nguồn vào ổ SSD NVMe tạm (`/content/R2AI-Stage-2`) và cài đặt `requirements.txt`.
2. **Mục 2 (Dữ liệu - Chọn 2.A hoặc 2.B):**
   * **Phương án 2.A:** Nạp nhanh và giải nén `data_backup.zip` từ Google Drive `/MyDrive/backup/data_backup.zip` vào ổ NVMe.
   * **Phương án 2.B:** Xây dựng toàn bộ từ đầu theo từng bước: `fetch` $\rightarrow$ `corpus` $\rightarrow$ `index` $\rightarrow$ đóng gói `data_backup.zip` lưu lại vào Drive.
3. **Mục 3 (Suy luận):** Chạy lệnh suy luận với mô hình `Qwen/Qwen2.5-Coder-7B-Instruct` (4-bit BitsAndBytes). Hệ thống tự động sao lưu checkpoint kép (vừa lưu local vừa sync sang Drive sau mỗi 5 câu).
4. **Mục 4 & 5 (Đóng gói & Đánh giá):** Tạo file `submission.zip` lưu vào Google Drive và đánh giá trên tập nhãn chuẩn `gold.json`.

---

### 3.2. Chạy trên Kaggle Notebook (GPU Tesla T4 16GB)
Mở file [notebooks/pipeline_runner_kaggle.ipynb](notebooks/pipeline_runner_kaggle.ipynb) trên Kaggle:
1. **Mục 1 (Môi trường):** Clone repo vào `/kaggle/working/R2AI-Stage-2` và cài đặt `requirements.txt`.
2. **Mục 2 (Dữ liệu - Chọn 2.A hoặc 2.B):**
   * **Phương án 2.A (Nạp nhanh):** Tự động xử lý cả 2 cách tải lên Kaggle Dataset:
     - *Cách 1:* File nén `.zip.bin` (hoặc `.zip`) được giải nén tự động vào `data/`.
     - *Cách 2:* Folder `data/` được Kaggle giải nén sẵn được đồng bộ trực tiếp vào `data/`.
   * **Phương án 2.B (Build từ đầu):** Xây dựng toàn bộ: `fetch` $\rightarrow$ `corpus` $\rightarrow$ `index` $\rightarrow$ đóng gói `data_backup.zip` tại `/kaggle/working/`.
3. **Mục 3 (Suy luận & Resume):** Chạy suy luận với mô hình 4-bit, tự động lưu checkpoint vào `/kaggle/working/backup/` (hoặc nhận file `questions_pred.json` upload từ máy tính để resume).
4. **Mục 4 & 5 (Đóng gói & Đánh giá):** Đóng gói `submission.zip` tại `/kaggle/working/submission.zip` để tải trực tiếp từ giao diện Kaggle.

---

### 3.3. Chạy trên Máy Cá nhân (Local GPU / CPU)
1. **Khởi tạo môi trường ảo:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
2. **Cài đặt thư viện:**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. **Chạy kiểm thử hệ thống:**
   ```bash
   pytest tests/ -q
   ```

---

## 4. Cấu hình Mô hình LLM (`configs/config.yaml`)

Hệ thống cho phép chuyển đổi linh hoạt mô hình và backend suy luận tại [configs/config.yaml](file:R2AI-Stage-2/configs/config.yaml):

```yaml
llm:
  # Cấu hình MẶC ĐỊNH cho GPU T4 (Colab / Kaggle 16GB) & Local GPU
  model_id: Qwen/Qwen2.5-Coder-7B-Instruct
  backend: transformers          # transformers | vllm | openai
  base_url: http://localhost:8000/v1
  max_tokens: 512                # so token sinh moi luot
  temperature: 0.0
  top_p: 1.0
  seed: 42
  dtype: auto
  quantization: awq              # awq | gptq | none (chống tràn VRAM 16GB)
  gpu_memory_utilization: 0.75
  max_model_len: 4096

llm_local:
  # Cấu hình MẶC ĐỊNH cho Local GPU/CPU
  model_id: Qwen/Qwen2.5-Coder-1.5B-Instruct
  backend: transformers
  max_tokens: 512
  temperature: 0.0
  device_map: auto
  torch_dtype: float16
```

Nếu muốn chạy suy luận qua API từ xa (OpenAI-compatible server hoặc máy chủ vLLM tự host):
1. Đổi `llm.backend: openai` và cấu hình `llm.base_url`.
2. Điền `LLM_API_KEY` trong file `.env`.

---

## 5. Hướng dẫn Vận hành Hệ thống

### Cách 1: Chạy 1 Lệnh Duy nhất Toàn Chu trình (End-to-End)

Chạy liên hoàn toàn bộ từ fetch data (nếu chưa có) $\rightarrow$ tạo corpus $\rightarrow$ tạo chỉ mục $\rightarrow$ suy luận LLM $\rightarrow$ đóng gói ZIP:
```bash
python main.py all --questions data/questions/questions.jsonl --name submission_final
```

---

### Cách 2: Chạy Tuần tự Từng Giai đoạn (Scripts 00 $\rightarrow$ 05)

#### Bước 0: Tải dữ liệu từ Hugging Face
```bash
python main.py fetch
# hoặc: python scripts/00_fetch_data.py
```

#### Bước 1: Bóc tách, Ghép bảng gãy & Chuẩn hóa kho dữ liệu (Build Corpus)
```bash
python main.py corpus
# hoặc: python scripts/01_build_corpus.py
```
*Tạo 119,045 bảng CSV chuẩn hóa tại `data/processed/`. Có cơ chế **Resume tự động** (tiếp tục từ doc dở dang nếu bị ngắt quãng).*

#### Bước 2: Xây dựng Chỉ mục Tìm kiếm (Build Index)
```bash
python main.py index
# hoặc: python scripts/02_build_index.py
```
*Tự động quét `data/processed/` để tạo/cập nhật `data/index/manifest.jsonl`, sau đó xây dựng chỉ mục từ khóa `data/index/bm25.pkl` và `vectors.pkl` (Dense Vectors).*

#### Bước 3: Chạy Suy luận Trả lời Câu hỏi (Batch Inference & Checkpoint)
```bash
python main.py infer \
# hoặc: python scripts/03_generate_answers.py \
    --questions data/questions/questions.jsonl \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --backend vllm \
    --pred outputs/predictions/questions_pred.json
```
*Tự động lưu checkpoint mỗi 5 câu hỏi; nếu bị gián đoạn, chỉ cần chạy lại lệnh để tiếp tục.*

#### Bước 4: Kiểm tra & Đóng gói File Nộp bài (Packaging)
```bash
# Kiểm tra hợp lệ định dạng
python main.py package --pred outputs/predictions/questions_pred.json --check-only
# hoặc: python scripts/04_package.py --pred outputs/predictions/questions_pred.json --check-only

# Đóng gói file ZIP nộp bài
python main.py package \
# hoặc: python scripts/04_package.py \
    --pred outputs/predictions/questions_pred.json \
    --questions data/questions/questions.jsonl \
    --name submission_vfinqa
```
*File nộp bài sẵn sàng tại: `outputs/submissions/submission_vfinqa.zip`.*

#### Bước 5: Đánh giá Điểm Hệ thống (Evaluation trên Gold Set)
```bash
python main.py eval --pred outputs/predictions/questions_pred.json
# hoặc: python scripts/05_evaluate.py --pred outputs/predictions/questions_pred.json
```

---

### Cách 3: Chạy qua Jupyter Notebook

* Mở [notebooks/pipeline_runner_colab.ipynb](file:R2AI-Stage-2/notebooks/pipeline_runner_colab.ipynb) trên Google Colab.
* Hoặc mở [notebooks/pipeline_runner_kaggle.ipynb](file:R2AI-Stage-2/notebooks/pipeline_runner_kaggle.ipynb) trên Kaggle Notebook.
    Sau đó chạy các cell code cần thiết để tương tác trực quan và theo dõi tiến trình từng ô lệnh.

---

## 6. Kiểm thử Toàn diện (Testing)

Hệ thống đi kèm bộ kiểm thử tự động toàn diện kiểm tra tất cả các module cốt lõi:
```bash
pytest tests/ -q
```

**Chi tiết các bộ test suite:**
* `tests/test_html_table.py`: Kiểm thử trích xuất bảng HTML, sửa lỗi dính chữ/số OCR và thuật toán ghép bảng gãy qua trang (`TableStitcher`).
* `tests/test_number_parser.py`: Kiểm thử chuyển đổi định dạng số tài chính Việt Nam (dấu chấm/phẩy, số âm ngoặc đơn, regex đơn vị tiền tệ triệu/tỷ).
* `tests/test_hierarchical_prefix.py`: Kiểm thử thuật toán lan truyền tiền tố nhóm cha vào cột `item`.
* `tests/test_report_type_filter.py`: Kiểm thử phân loại câu hỏi (BCTC mẹ vs hợp nhất) và bộ lọc cứng `report_type`.
* `tests/test_selector_coverage.py`: Kiểm thử Multi-Section Coverage cho câu hỏi tính chỉ số phái sinh (ROE/ROA).
* `tests/test_sandbox_repair.py`: Kiểm thử môi trường thực thi an toàn AST `PandasSandbox` và chẩn đoán tự sửa lỗi Self-Repair.
* `tests/test_submission_schema.py`: Đảm bảo cấu trúc file JSON nộp bài luôn khớp 100% với yêu cầu của Ban tổ chức.

---

## 7. Quy chuẩn File Nộp bài (Submission Format)

File `submission.zip` được đóng gói trực tiếp ở cấp ngoài cùng:
```
submission.zip
├── submission.json
└── data/
    ├── <bang_evidence_1>.csv
    ├── <bang_evidence_2>.csv
    └── ...
```

Cấu trúc mỗi phần tử trong `submission.json`:
```json
{
  "id": 1,
  "question": "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?",
  "answer": 208253.2,
  "relevant_docs": ["VJC_financial_statements_2018_separate"],
  "relevant_tables": ["VJC_financial_statements_2018_separate|33"],
  "evidence": [
    {
      "variable": "df1",
      "csv_path": "data/VJC_financial_statements_2018_separate_table_33.csv"
    }
  ],
  "pandas_query": "sub = df1[(df1['year'] == 2018) & (df1['item'].str.contains('Lãi tiền gửi', case=False, na=False))]\nresult = float(sub['value'].iloc[0]) if not sub.empty else 0.0"
}
```

---

## 8. Kiến trúc Công nghệ & Tài liệu Tham khảo (Core Technologies & References)

### 8.1. Các Công nghệ & Lý do Lựa chọn

1. **Mô hình Ngôn ngữ Lớn (LLM) — Qwen2.5-14B-Instruct / Qwen2.5-Coder**:
   * **Lựa chọn:** `Qwen2.5-Coder-7B-Instruct` hoặc `Qwen2.5-14B-Instruct-AWQ`.
   * **Lý do:** Qwen2.5 hiện là dòng mô hình mã nguồn mở dưới 14B có khả năng suy luận logic, hiểu cấu trúc bảng biểu và sinh mã Python/Pandas mạnh và đáp ứng yêu cầu của cuộc thi. Model 14B thì bản lượng tử hóa 4-bit AWQ giúp mô hình 14B chạy được trên 16GB VRAM GPU T4 nhưng vẫn còn hạn chế, cần tùy chỉnh thêm nhiều thông số bổ sung để hoạt động tối ưu. Còn model 7B thì cũng cần phải lượng tử hóa 4-bit với bitsandbytes để chạy được với 16GB VRAM GPU T4 (bản thường 16-bit vẫn chiếm dụng tới ~14GB VRAM).
2. **Backend Suy luận — Hugging Face Transformers & AutoAWQ**:
   * **Lựa chọn:** `transformers` với `autoawq` / `gptqmodel`.
   * **Lý do:** Chạy đồng bộ trực tiếp trong tiến trình chính, tương thích trên cả Windows, Linux và Google Colab. Tránh được các lỗi xung đột CUDA context và deadlock IPC của các runtime multiprocessing trên GPU T4.
3. **Mô hình Nhúng Vector Đa ngữ — BAAI/bge-m3 & FAISS**:
   * **Lựa chọn:** `BAAI/bge-m3` kết hợp `FAISS (IndexFlatIP)`.
   * **Lý do:** BGE-M3 hỗ trợ hơn 100 ngôn ngữ với khả năng biểu diễn ngữ nghĩa tiếng Việt tài chính sâu sắc, nhận diện mối quan hệ giữa từ ngữ câu hỏi và thẻ mô tả bảng (Table Card). FAISS cho phép tìm kiếm độ tương đồng Cosine (Inner Product sau chuẩn hóa) trên 119,000 vector chiều 1024 chỉ trong vài mili-giây.
4. **Hợp nhất Chỉ mục Lai — BM25Okapi + Reciprocal Rank Fusion (RRF)**:
   * **Lựa chọn:** `BM25Okapi` (từ khóa chính xác: mã ticker, năm, chỉ tiêu) + `Dense Vector` (ngữ nghĩa) $\rightarrow$ `RRF (k=60)`.
   * **Lý do:** BM25 bắt chính xác các từ khóa số và tên mã chứng khoán viết tắt; Dense vector bắt các từ đồng nghĩa tài chính. RRF hợp nhất hai bảng xếp hạng một cách phi tham số, đạt điểm **Recall F2 tối đa** mà không bị lệch trọng số.
5. **Môi trường Thực thi An toàn Sandbox AST & Self-Repair Loop**:
   * **Lựa chọn:** `PandasSandbox` (kiểm tra cây cú pháp trừu tượng AST) + `SelfRepairExecutor`.
   * **Lý do:** Ngăn chặn tuyệt đối các lệnh nguy hiểm (file I/O, network, shell). Khi code Pandas gặp lỗi runtime (`KeyError`, `IndexError`), traceback và schema bảng được phản hồi ngược lại LLM để tự sửa lỗi (tối đa 3 lần), nâng cao tỷ lệ sinh câu trả lời thành công.
6. **Bộ Chuẩn hóa Đơn vị Tài chính Tự động (Financial Unit Auto-Scaling)**:
   * **Lý do:** Đề thi hỏi bằng nhiều đơn vị khác nhau (*tỷ đồng, triệu đồng, nghìn đồng, USD, %*). Module tự động phát hiện đơn vị trong câu hỏi và đơn vị gốc của bảng để áp dụng hệ số nhân chuẩn hóa ($10^9, 10^6, 10^3$) trước khi trả lời.

---

### 8.2. Tài liệu Tham khảo (References)

* **Qwen2.5 Technical Report**: Yang, A., et al. (2024). *Qwen2.5 Technical Report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).
* **BGE-M3 (Multilingual Embedding)**: Chen, J., et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).
* **AWQ: Activation-aware Weight Quantization**: Lin, J., et al. (2023). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
* **Reciprocal Rank Fusion (RRF)**: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). *Reciprocal rank fusion outperformsres ranking methods*. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 868-869).
* **FAISS (Billion-scale Similarity Search)**: Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data, 7(3), 535-547.
* **BM25Okapi**: Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389.

---

## Bạn có thể lấy dataset đã xử lý sẵn tại:
* Dataset dạng nén `.zip.bin`: 
    [https://www.kaggle.com/datasets/anhtu25/s2-backup](https://www.kaggle.com/datasets/anhtu25/s2-backup)
* Dataset thư mục đã giải nén sẵn: 
    [https://www.kaggle.com/datasets/anhtu25/r2ai-s2-backup](https://www.kaggle.com/datasets/anhtu25/r2ai-s2-backup)
