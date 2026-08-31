
# Tài Liệu Tóm Tắt API & Schema Cho Developer

Tài liệu này tổng hợp danh sách các hàm, lớp và cấu trúc dữ liệu thiết yếu dùng để phát triển các module trong pipeline Finance QA.

---

## 1. Utility

### 1.1. Module `io`

* `def read_json(path: str | Path)``-> Any`Đọc và giải mã dữ liệu từ file JSON tại đường dẫn chỉ định thành các đối tượng Python tương ứng. Hàm thực hiện xử lý trực tiếp nội dung file với chuẩn mã hóa UTF-8.
* `def write_json(obj: Any, path: str | Path, indent: int = 2)``-> None`Chuyển đổi đối tượng Python thành định dạng JSON và ghi xuống đĩa. Hàm tự động kiểm tra, khởi tạo toàn bộ các thư mục cha nếu chưa tồn tại và áp dụng định dạng UTF-8 chuẩn.
* `def read_jsonl(path: str | Path)``-> Iterator[dict]`Đọc file JSONL theo cơ chế nạp lười (generator), xử lý từng dòng một nhằm tối ưu hóa bộ nhớ RAM khi làm việc với các tập dữ liệu lớn.
* `def write_jsonl(rows: Iterable[dict], path: str | Path)``-> None`Ghi một tập hợp các dictionary thành file dạng JSONL (mỗi record một dòng). Tự động tạo cây thư mục chứa file nếu cần.
* `def write_jsonl_atomic(rows: Iterable[dict], path: str | Path)``-> int`Thực hiện ghi file JSONL theo cơ chế nguyên tử (atomic write): ghi dữ liệu ra file tạm, ép đồng bộ dữ liệu xuống ổ đĩa vật lý bằng `fsync`, sau đó mới ghi đè bằng `os.replace`. Cơ chế này đảm bảo dữ liệu không bị hỏng hoặc mất mát nếu quá trình bị ngắt đột ngột (như mất điện, hết đĩa, ngắt tiến trình). Trả về số lượng dòng đã ghi thành công.
* `def read_text(path: str | Path, encoding: str = "utf-8")``-> str`Đọc nội dung văn bản thô từ file (thường dùng cho dữ liệu OCR). Có cơ chế tự động hạ cấp (fallback) sang bảng mã `latin-1` nếu gặp lỗi giải mã UTF-8, đảm bảo quá trình đọc file luôn hoàn tất mà không ngắt luồng.
* `def read_csv(path: str | Path)``-> pd.DataFrame`Đọc dữ liệu từ file CSV nạp vào Pandas DataFrame, mặc định áp dụng chuẩn mã hóa `utf-8-sig` để xử lý chính xác ký tự Tiếng Việt và ký tự BOM.
* `def write_csv(df: pd.DataFrame, path: str | Path)``-> None`Xuất dữ liệu từ Pandas DataFrame ra file CSV với chuẩn mã hóa `utf-8-sig` và tự động xử lý thư mục lưu trữ.
* `def load_pickle(path: str | Path)``-> Any`Khôi phục đối tượng Python nguyên bản từ file nhị phân `.pickle`.
* `def save_pickle(obj: Any, path: str | Path)`
  `-> None`
  Đóng gói đối tượng Python dưới dạng file nhị phân `.pickle` sử dụng protocol cao nhất để đạt hiệu năng lưu trữ tối ưu. Tự động khởi tạo thư mục đích nếu chưa có.

### 1.2. Module `logging`

**Hàm cấp Module:**

* `def setup_logging(logging_dir: str | Path = "logs")``-> None`Khởi tạo môi trường nhật ký tập trung cho hệ thống, tự động tạo thư mục lưu log và giảm bớt độ nhiễu thông tin (warning level) từ các thư viện bên thứ ba như httpx, urllib3 hay faiss.
* `def get_logger(name: str, logging_dir: str | Path | None = None)`
  `-> ModuleLogger`
  Hàm factory cấp phát một instance `ModuleLogger` riêng cho từng module. Mỗi logger được gán một file ghi log chi tiết độc lập nằm trong thư mục log chỉ định.

**Class `ModuleLogger`:**

* `def progress(text: str)``-> None`Method thuộc `ModuleLogger`. In thông điệp ngắn gọn ra Console (sys.stdout) để theo dõi tiến trình thực thi trực quan trên terminal theo thời gian thực.
* `def detail(text: str)`
  `-> None`
  Method thuộc `ModuleLogger`. Ghi vết thao tác chi tiết (mức DEBUG) trực tiếp vào file log riêng của module tương ứng phục vụ mục đích truy vết và debug.

---

## 2. LLM Call

### 2.1. Class `LLM` (Interface)

* `def generate(prompt: str, system_prompt: Optional[str] = None, **kwargs: Any)``-> str`Sinh văn bản phản hồi cho một câu lệnh đơn lẻ. Phù hợp cho các tác vụ trích xuất đơn giản hoặc trích xuất ứng viên rộng (Recall).
* `def chat(messages: List[Dict[str, str]], **kwargs: Any)``-> str`Xử lý chuỗi tin nhắn hội thoại đa lượt theo định dạng chuẩn (role và content) để tạo ra câu phản hồi phù hợp từ mô hình.
* `def generate_batch(prompts: List[str], system_prompt: Optional[str] = None, **kwargs: Any)``-> List[str]`Xử lý đồng thời một danh sách các prompts đầu vào dưới dạng batch để tối ưu hóa hiệu năng tính toán và tốc độ xử lý của mô hình.
* `def release_resources()`
  `-> None`
  Cưỡng chế giải phóng mô hình khỏi bộ nhớ (VRAM/RAM), thu dọn tiến trình và giải phóng bộ nhớ đệm GPU/CPU để nhường tài nguyên cho các bước tiếp theo trong pipeline.

### 2.2. Class `LLMProvider`

* `def from_yaml(config_path: str)``-> LLMProvider`classmethod đọc file cấu hình YAML và khởi tạo đối tượng `LLMProvider` chứa danh sách cấu hình của tất cả backend.
* `def get_llm(alias: str)``-> LLM`Truy xuất client mô hình LLM tương ứng dựa trên tên bí danh (`alias`). Hàm thực hiện khởi tạo lười (lazy instantiation), chỉ thực sự nạp mô hình vào bộ nhớ ở lần gọi đầu tiên.
* `def release_llm(alias: str)``-> None`Giải phóng tài nguyên phần cứng (VRAM/RAM) của duy nhất một instance LLM được chỉ định bởi bí danh `alias`.
* `def release_all()`
  `-> None`
  Duyệt qua tất cả các instance LLM đang hoạt động và thực hiện giải phóng toàn bộ tài nguyên phần cứng mà chúng đang chiếm dụng.

---

## 3. Schemas

### 3.1. Dữ liệu Ingestion & Extraction

**Data Class `Document`:**

* `def get_all_tables()``-> List[Table]`Duyệt qua toàn bộ hệ thống phân cấp (trang, heading) trong tài liệu để trích xuất và trả về danh sách đầy đủ các bảng dữ liệu có sẵn.
* `def get_id()`
  `-> str`
  Trả về mã định danh chuẩn đại diện cho tài liệu (`doc_id`), phục vụ cho trường `relevant_docs` trong kết quả nộp bài.

**Data Class `Table`:**

* `def get_id()`
  `-> str`
  Tạo và trả về chuỗi ID duy nhất cho bảng dưới định dạng `<doc_id>|<index>`, dùng để ghi nhận vị trí chính xác của bảng cho trường `relevant_tables`.

### 3.2. Dữ liệu Submission

**Data Class `Evidence`:**

* `def to_dict()`
  `-> dict[str, str]`
  Đóng gói thông tin về tên biến DataFrame và đường dẫn lưu trữ file CSV thành cấu trúc dictionary chuẩn.

**Data Class `SubmissionItem`:**

* `def to_dict()`
  `-> dict[str, Any]`
  Chuyển đổi toàn bộ thông tin kết quả dự đoán (gồm câu hỏi, câu trả lời, tài liệu/bảng liên quan, evidence và truy vấn pandas) thành dictionary chuẩn để lưu thành file JSON nộp bài.
