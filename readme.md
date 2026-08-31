# R2AI Stage 2 — Financial Table Retrieval & Text-to-Pandas

Dự án này là giải pháp cho Stage 2 của cuộc thi **ViFinQA (Financial Table Retrieval & Text-to-Pandas)**. Mục tiêu của hệ thống là tự động trả lời các câu hỏi tài chính tiếng Việt dựa trên kho Báo cáo tài chính (BCTC) của các công ty niêm yết bằng cách truy hồi đúng bảng số liệu và sinh câu lệnh Pandas thực thi được để tính toán đáp án.

Đây là **pipeline chạy offline theo lô (batch)**: chạy một lần trên toàn bộ tập câu hỏi kiểm thử và đóng gói kết quả thành file ZIP nộp lên hệ thống chấm điểm của Ban tổ chức (BTC).

---

## 1. Luồng hoạt động của Hệ thống (Pipeline)

Hệ thống bao gồm các bước chính sau:

1. **Ingestion & Parsing**: Duyệt qua kho BCTC gốc dạng `.txt` OCR, trích xuất các bảng biểu dạng HTML, sửa lỗi dính chữ/số (glued numbers) do rowspan gây ra.
2. **Standardization (Chuẩn hóa cấu trúc)**: Ép các bảng từ định dạng bảng ngang (Wide) về cấu trúc dọc (Long Schema) thống nhất. Nhận diện các đơn vị đo lường (`vnd`, `trieu dong`, `ty dong`) và giải mã kỳ báo cáo (đầu kỳ `opening` hay cuối kỳ `closing`).
3. **Indexing**: Lập chỉ mục từ khóa (BM25) và ngữ nghĩa (Dense Vector dùng model `BAAI/bge-m3`) trên các thẻ mô tả bảng (Table Cards).
4. **Hybrid Retrieval & Reranking**:
   * Phân tích thực thể trong câu hỏi (Mã chứng khoán, năm, chỉ tiêu cần tìm).
   * Áp dụng bộ lọc cứng (Hard Filter) theo mã chứng khoán và năm.
   * Truy hồi lai (BM25 + Dense) kết hợp RRF (Reciprocal Rank Fusion).
   * Sắp xếp lại ứng viên (Reranking) bằng Cross-Encoder `BAAI/bge-reranker-v2-m3`.
   * Chỉ định động số lượng bảng cần chọn (`TableSelector`) để tối ưu điểm F2.
5. **Pandas Code Generation**: Sinh truy vấn Pandas bằng LLM (mặc định là `Qwen/Qwen2.5-14B-Instruct`) dựa trên schema bảng đã truy hồi và các gợi ý công thức.
6. **Sandbox Execution & Self-Repair**: Thực thi code trong môi trường AST an toàn. Nếu xảy ra lỗi runtime (KeyError, IndexError...), hệ thống sẽ chẩn đoán lỗi và gửi gợi ý để LLM tự sửa lỗi (`Self-Repair` tối đa 3 lần).
7. **Packaging**: Kiểm tra định dạng schema khắt khe và đóng gói file ZIP sẵn sàng đem nộp.

---

## 2. Cấu trúc Thư mục Dự án

```
R2AI-Stage-2-main/
├── main.py                   # Điểm khởi chạy toàn bộ pipeline
├── requirements.txt          # Danh sách thư viện cần thiết
├── .env                      # File cấu hình biến môi trường (token, khóa API) - KHÔNG COMMIT
├── configs/                  # Chứa cấu hình YAML và Prompts
│   ├── config.yaml           # Cấu hình đường dẫn, tham số LLM và thực thi sandbox
│   ├── retrieval.yaml        # Tham số cho BM25, Dense Vector, Rerank và Selector
│   └── prompts/              # Thư mục chứa prompt sinh code và self-repair
├── src/                      # Mã nguồn chính của dự án
│   ├── config.py             # Parser nạp configs/config.yaml và biến môi trường
│   ├── schemas.py            # Khai báo các Dataclass dùng chung toàn pipeline
│   ├── ingestion/            # Bộ quét và phân tích file BCTC thô
│   ├── extraction/           # Trích xuất bảng HTML và sửa lỗi dính số OCR
│   ├── normalization/        # Chuẩn hóa bảng, chuyển đổi số và phát hiện kỳ báo cáo/đơn vị
│   ├── embeddings/           # Tạo card mô tả bảng và quản lý model Embedding
│   ├── vectordb/             # Quản lý kho dữ liệu BM25, FAISS Vector và Metadata
│   ├── retrieval/            # Bộ phân tích truy vấn, áp bộ lọc cứng, tìm kiếm lai và xếp hạng lại
│   ├── generation/           # Chuẩn bị dữ liệu và gọi LLM để sinh code Pandas
│   ├── execution/            # Chạy thử code Pandas trong Sandbox AST và bộ tự sửa lỗi (Self-Repair)
│   ├── submission/           # Xây dựng và kiểm tra tính hợp lệ của file nộp bài
│   ├── evaluation/           # Tính điểm đánh giá hệ thống (Precision, Recall, F2, Accuracy)
│   ├── pipeline/             # Các lớp điều phối luồng xử lý (Corpus, Index, Answer)
│   └── utils/                # Các thư viện bổ trợ về logging, xử lý văn bản tiếng Việt
├── scripts/                  # Các file script CLI chạy từng bước độc lập
│   ├── 00_fetch_data.py      # Tải dữ liệu ViFinQA từ Hugging Face
│   ├── 01_build_corpus.py    # Xử lý OCR -> Tạo các file CSV chuẩn hóa và manifest
│   ├── 02_build_index.py     # Tạo chỉ mục BM25 và Dense Vector Index
│   ├── 03_run_inference.py   # Chạy suy diễn trả lời bộ câu hỏi
│   ├── 04_package.py         # Kiểm tra tính hợp lệ và đóng gói ZIP nộp bài
│   └── 05_evaluate.py        # Đánh giá kết quả trên tập nhãn tự gán (gold labels)
├── tests/                    # Thư mục kiểm thử (pytest)
├── data/                     # Thư mục chứa dữ liệu đầu vào và các index (được gitignore)
│   ├── raw/                  # File BCTC .txt của BTC
│   ├── interim/              # Bảng thô đã trích xuất chưa chuẩn hóa
│   ├── processed/            # File CSV bảng số liệu chuẩn hóa
│   ├── index/                # Chỉ mục BM25, FAISS và manifest.jsonl
│   └── questions/            # Tập câu hỏi câu hỏi câu hỏi test (questions.jsonl)
├── labels/                   # Tập nhãn tự gán gold.json phục vụ test offline - GIỮ TRONG GIT
├── logs/                     # File log chạy ứng dụng
└── outputs/                  # Thư mục chứa kết quả predictions và submissions zip
```

---

## 3. Cài đặt Môi trường

Dự án yêu cầu cài đặt Python 3.10 trở lên. Hãy làm theo các bước sau để thiết lập môi trường:

```bash
# 1. Tạo môi trường ảo (Virtual Environment)
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# 3. Nâng cấp pip và cài đặt các thư viện cần thiết
pip install --upgrade pip
pip install -r requirements.txt

# 4. Cấu hình biến môi trường
cp .env.example .env
```

*Sau khi copy file `.env`, hãy mở file và điền thông tin `HF_TOKEN` nếu bạn sử dụng các mô hình gated hoặc cần tương tác với HuggingFace.*

**Lưu ý khi sử dụng GPU**:

* Dự án mặc định cài đặt Torch bản CPU. Nếu bạn có GPU CUDA, vui lòng cài đặt phiên bản Torch phù hợp trước qua hướng dẫn của [PyTorch](https://pytorch.org/get-started/locally/).
* Thư viện `vllm` hỗ trợ tăng tốc suy diễn và chỉ hoạt động trên môi trường Linux + GPU. Nếu chạy trên Windows, hãy đổi cấu hình `llm.backend` trong `configs/config.yaml` từ `vllm` sang `transformers`.

---

## 4. Hướng dẫn Chạy Hệ thống từng bước

### Bước 0: Tải dữ liệu cuộc thi

Tải toàn bộ bộ câu hỏi và kho báo cáo tài chính về thư mục dữ liệu cục bộ:

```bash
python scripts/00_fetch_data.py
```

*If you only want to quickly test with questions and stock tickers, run: `python scripts/00_fetch_data.py --questions-only`*

### Bước 1: Trích xuất và Chuẩn hóa kho dữ liệu (Build Corpus)

Đọc toàn bộ file `.txt` gốc, bóc tách bảng HTML, chuẩn hóa thành dạng dọc (Long CSV) và ghi lại file `manifest.jsonl`. Đây là bước tốn nhiều thời gian nhất:

```bash
python scripts/01_build_corpus.py
```

*Mẹo debug: Có thể chạy thử trên 20 file tài liệu đầu tiên bằng tham số `--limit 20`: `python scripts/01_build_corpus.py --limit 20`*

### Bước 2: Xây dựng chỉ mục tìm kiếm (Build Index)

Đọc manifest đã tạo ở Stage 1 để lập chỉ mục từ khóa (BM25) và ngữ nghĩa (Dense Vector). File index sẽ được ghi vào `data/index/bm25.pkl` và `data/index/vectors.pkl`:

```bash
python scripts/02_build_index.py
```

*Nếu bạn không có GPU hoặc muốn chạy nhanh bỏ qua phần dense vector, hãy dùng cờ `--skip-dense`.*

### Bước 3: Chạy suy diễn trả lời câu hỏi (Inference)

Đọc bộ câu hỏi, thực hiện tìm kiếm bảng, gọi LLM sinh code Pandas và chạy thực thi để tìm ra đáp án cuối cùng:

```bash
python scripts/03_run_inference.py --questions data/questions/questions.jsonl
```

*Bạn có thể giới hạn số câu chạy thử để kiểm thử nhanh bằng cờ `--limit 5`.*

### Bước 4: Kiểm tra và Đóng gói (Packaging)

Xác thực tính đúng đắn về định dạng của tệp kết quả dự đoán (schema, kiểu dữ liệu, đường dẫn tương đối...) và đóng gói tệp kết quả ZIP để nộp:

```bash
# Chỉ kiểm tra lỗi định dạng, không tạo file ZIP
python scripts/04_package.py --pred outputs/predictions/questions.json --check-only

# Kiểm tra định dạng và đóng gói tệp nộp bài
python scripts/04_package.py --pred outputs/predictions/questions.json --name run_01
```

### Bước 5: Đánh giá nội bộ (Local Evaluation)

Đánh giá độ chính xác của tệp kết quả dự đoán so với nhãn chuẩn tự gán trong thư mục `labels/gold.json`:

```bash
python scripts/05_evaluate.py --pred outputs/predictions/questions.json
```

*Xem 30 câu hỏi có điểm số tệ nhất để debug: `python scripts/05_evaluate.py --pred outputs/predictions/questions.json --worst 30`*

---

## 5. Hướng dẫn Chạy Kiểm thử (Testing)

Dự án cung cấp bộ unit test phong phú bảo vệ các cấu phần dễ sai sót nhất. Để chạy toàn bộ test, thực thi lệnh sau từ thư mục gốc của dự án:

```bash
# Chạy toàn bộ test
pytest

# Chạy ở chế độ gọn nhẹ (quiet)
pytest -q

# Chạy một file test cụ thể và hiển thị chi tiết (verbose)
pytest tests/test_number_parser.py -v
```

Các nhóm kiểm thử quan trọng bao gồm:

* `tests/test_number_parser.py`: Kiểm thử bộ chuyển đổi định dạng số Việt Nam (dấu chấm ngăn nghìn, dấu phẩy thập phân, dấu ngoặc âm kế toán) để đảm bảo không bị lệch giá trị số 1000 lần.
* `tests/test_table_detector.py`: Kiểm tra tính năng nhận diện bảng văn bản theo các dialect khác nhau.
* `tests/test_submission_schema.py`: Đảm bảo cấu trúc file JSON nộp bài luôn khớp 100% với yêu cầu từ dashboard của BTC.

---

## 6. Định dạng File Nộp bài (Submission Format)

Tệp ZIP nộp bài phải chứa file kết quả `submission.json` và thư mục `data/` chứa các CSV bảng được truy cập trực tiếp ở cấp ngoài cùng:

```
submission.zip
├── submission.json
└── data/
    ├── <bang_1>.csv
    └── ...
```

File `submission.json` là một danh sách, mỗi câu hỏi bắt buộc phải có đầy đủ 7 trường thông tin sau:

```json
{
  "id": 1,
  "question": "Doanh thu thuần của Công ty CP Sữa Việt Nam (VNM) năm 2023 là bao nhiêu?",
  "answer": 63075000000.0,
  "relevant_docs": ["VNM_financial_statements_2023_consolidated"],
  "relevant_tables": ["VNM_financial_statements_2023_consolidated|5"],
  "evidence": [
    {
      "variable": "df1",
      "csv_path": "data/VNM_financial_statements_2023_consolidated_table_5.csv"
    }
  ],
  "pandas_query": "result = df1[(df1.year == 2023)]['value'].sum()"
}
```

---

## 7. Các điểm Cần Cải tiến & Tối ưu hóa Tiếp theo (Roadmap)

Dựa trên phân tích thiết kế hiện tại, đây là các đầu việc cần thực hiện để đạt điểm số cao hơn trên Bảng xếp hạng:

1. **Xây dựng chỉ mục Dense Vector đầy đủ**:
   Cần chạy tạo tệp chỉ mục `vectors.pkl` thông qua mô hình `BAAI/bge-m3` để kích hoạt hoàn toàn cơ chế truy hồi ngữ nghĩa lai (Hybrid Search), tăng Recall khi câu hỏi dùng từ đồng nghĩa với bảng.
2. **Mở rộng Từ điển ánh xạ Chỉ tiêu (`term_mapper.py`)**:
   Bổ sung các biến thể tên gọi chỉ tiêu tài chính thực tế của doanh nghiệp Việt Nam vào từ điển `FINANCIAL_TERMS` trong [term_mapper.py](file:///D:/work/learn/R2AI-Stage-2-main/src/normalization/term_mapper.py) để ánh xạ chuẩn xác trường `metric`.
3. **Tích hợp bộ lọc Hợp nhất vs Riêng lẻ (Consolidated vs Separate)**:
   Cần bổ sung logic trích xuất thuộc tính BCTC hợp nhất hoặc riêng lẻ từ câu hỏi trong [query_analyzer.py](file:///D:/work/learn/R2AI-Stage-2-main/src/retrieval/query_analyzer.py) và biến nó thành một bộ lọc cứng (Hard Filter) theo `report_type` để tránh nhầm lẫn bảng số liệu.
4. **Tối ưu hóa Prompt với Few-shot**:
   Nâng cấp prompt sinh code Pandas trong `configs/prompts/pandas_gen.txt` từ zero-shot lên few-shot (thêm 3-5 ví dụ minh họa cách viết code Pandas trên Long Schema) để LLM sinh code ổn định và chính xác hơn.
5. **Cải tiến Regex nhận diện Đơn vị (Unit & Scaling)**:
   Tăng cường các regex nhận diện từ viết tắt của đơn vị tính tiền tệ (như `tr.đ`, `triệuđ`, `tỷđ`) trong [number_parser.py](file:///D:/work/learn/R2AI-Stage-2-main/src/normalization/number_parser.py) để tránh việc nhân/chia sai tỷ lệ 1,000 hoặc 1,000,000 lần.
