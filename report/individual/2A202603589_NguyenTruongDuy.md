# Báo cáo Day 7 — Agentic RAG

**Học viên:** Nguyễn Trường Duy  
**MSSV:** 2A202603589  
**Lớp:** K5-L3A  
**Chủ đề:** Agentic Retrieval-Augmented Generation trên báo cáo World Bank *Commodity Markets Outlook* (October 2026).

---

## 1. Dữ liệu và benchmark

- **Nguồn dữ liệu & quyền sử dụng:** Báo cáo *Commodity Markets Outlook - October 2026* phát hành bởi World Bank Prospects Group (`starter/data/raw/CMO-October-2026.pdf`). Dữ liệu gồm 70 trang vật lý (58 trang nội dung chính loại trừ bìa, mục lục và trang trống).
- **Kho chunk & Vector Store:** Kế thừa từ Day 6; sử dụng Strategy A (637 chunks, fixed-size 500 ký tự, overlap 50 ký tự) và bảng `day6_chunks` trên PostgreSQL 16 + pgvector (`127.0.0.1:55433/day6_lab`).
- **Embedding model:** `openai/text-embedding-3-small` (1536 chiều, cosine distance metric `<=>`) truy xuất qua OpenRouter API.
- **Agent Model & điều phối:** `google/gemini-2.5-flash` qua OpenRouter Chat Completion API hỗ trợ function / tool-calling tự động (`tools=[search_vector, search_keyword, search_hybrid]`).
- **Benchmark & Phân chia:** Tổng cộng **24 câu hỏi** chuẩn mực đáp ứng yêu cầu Day 7:
  - **Dev set (17 câu):** 15 câu có đáp án trong tài liệu + 2 câu thiếu bằng chứng (`q_unans_dev_1`: Chilean copper mine labor strikes; `q_unans_dev_2`: Uranium concentrate 2027 forecast).
  - **Test set (7 câu):** 5 câu có đáp án trong tài liệu + 2 câu thiếu bằng chứng (`q_unans_test_1`: Brazil soybean contraction percentage; `q_unans_test_2`: August 2026 OPEC quotas).
- **Tiêu chí xác định & chấm điểm:**
  - *Câu có đáp án:* Đánh giá Hit@1, Evidence Hit@3, Context Evidence Rate. Chấm câu trả lời của agent thang điểm 0 (sai/không căn cứ), 1 (đúng một phần), 2 (đúng và đủ).
  - *Câu thiếu bằng chứng (Unanswerable):* Kiểm tra tính từ chối (Abstention = `abstain`) khi không tìm thấy bằng chứng hợp lệ, cấm hallucinate hoặc phỏng đoán ngoài ngữ cảnh. Các câu này không tính vào mẫu số retrieval Hit@k.

---

## 2. Retrieval và thử nghiệm

Thực hiện triển khai và đo lường độc lập **3 phương pháp retrieval**:
1. **Dense Retrieval (Baseline):** Query embedding với `text-embedding-3-small`, tìm kiếm Cosine Similarity trên pgvector.
2. **Lexical Retrieval (BM25):** Thuật toán Okapi BM25 thuần ngữ nghĩa từ khóa trên toàn bộ 637 chunks.
3. **Hybrid Retrieval (RRF):** Kết hợp Dense và BM25 bằng Reciprocal Rank Fusion ($k=60$).

Sau đó thử nghiệm **2 cấu hình cải thiện** trên dev:
- *Cải thiện 1:* Hybrid RRF tăng $top\_k=5$ và mở rộng context budget lên 1500 ký tự.
- *Cải thiện 2:* Hybrid RRF $top\_k=5$ kết hợp nới lỏng context budget lên 2000 ký tự (giữ trọn vẹn nhiều chunk bằng chứng liên quan hơn).

### Bảng kết quả tổng hợp trên Dev và Test

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline (Phương pháp 1)** | Dense vector search, top-k=3, budget=1000 | Baseline dense từ Day 6 | dev | 5/15 (33.3%) | 7/15 (46.7%) | 5/15 (33.3%) | 15.82s / $0.00030 |
| **Phương pháp 2** | Lexical BM25, top-k=3, budget=1000 | Tìm kiếm từ khóa thuần túy | dev | 5/15 (33.3%) | 7/15 (46.7%) | 5/15 (33.3%) | 0.05s / $0.00000 |
| **Phương pháp 3** | Hybrid RRF (Dense + BM25), top-k=3, budget=1000 | Kết hợp điểm mạnh ngữ nghĩa và từ khóa | dev | 5/15 (33.3%) | 9/15 (60.0%) | 6/15 (40.0%) | 17.59s / $0.00030 |
| **Cải thiện 1** | Hybrid RRF, top-k=5, budget=1500 | Tăng candidate pool và mở rộng context | dev | 5/15 (33.3%) | 9/15 (60.0%) | 6/15 (40.0%) | 15.30s / $0.00030 |
| **Cải thiện 2 (CHỌN)** | Hybrid RRF, top-k=5, budget=2000 | Tối ưu context budget để không làm rơi rớt chunk mang evidence | dev | 5/15 (33.3%) | 9/15 (60.0%) | 9/15 (60.0%) | 16.65s / $0.00030 |
| **Cấu hình đã chọn** | Hybrid RRF, top-k=5, budget=2000 | Đánh giá độc lập trên test sau khi đã chốt cấu hình | test | 2/5 (40.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 5.81s / $0.00010 |
| **Baseline** | Dense vector search, top-k=3, budget=1000 | Baseline đối chứng trên test | test | 1/5 (20.0%) | 2/5 (40.0%) | 2/5 (40.0%) | 5.34s / $0.00010 |

### Phân tích lỗi và lý do chọn cấu hình:
1. **Lỗi dev 1 (`q03` - Oil inventories drawdown rate):** Dense retrieval đơn thuần xếp chunk chứa "inventories" ở rank 4 do câu hỏi dùng từ vựng trừu tượng. Hybrid RRF đã đưa chunk này lên rank 2 nhờ BM25 boost từ khóa "2.4 million barrels". Tuy nhiên với context budget 1000 ký tự, chunk rank 2 bị cắt bỏ do chunk rank 1 chiếm hơn 500 ký tự. Mở rộng budget lên 2000 ký tự ở *Cải thiện 2* đã giải quyết hoàn toàn vấn đề này, nâng Context Evidence Rate từ 33.3% lên 60.0%.
2. **Lỗi dev 2 (`q04` - Commodity price increase 23% in 2026):** Cụm từ "increase by 23 percent" xuất hiện ở nhiều phần tóm tắt. Lexical BM25 bị nhiễu bởi các trang phụ lục bảng biểu, trong khi Dense giữ được ngữ cảnh Executive Summary. RRF kết hợp điểm số của cả hai giúp đưa đúng chunk trang 9 vào top-3.
3. **Lý do chọn Cải thiện 2:** Cải thiện 2 vượt trội hoàn toàn về Evidence Hit@3 (60.0% so với 46.7%) và Context Evidence Rate (60.0% so với 33.3%), trong khi chi phí embedding gần như tương đương và độ trễ chấp nhận được (~1.1s/truy vấn).

---

## 3. Agent và đánh giá

- **Thiết kế Agent:** Kiến trúc ReAct / Tool-calling với Gemini 2.5 Flash. Agent được trang bị 3 tools: `search_hybrid` (mặc định), `search_vector`, `search_keyword`.
- **Cơ chế điều phối:**
  - Tối đa 3 turns tương tác.
  - Sau mỗi turn tìm kiếm, các chunk mới được chèn vào bộ nhớ context tuân thủ nghiêm ngặt ngân sách `context_budget=2000` ký tự.
  - Agent được chỉ dẫn bắt buộc kèm citation `[chunk_id]` và chủ động từ chối (`status: abstain`) khi không đủ dữ kiện.

### Bảng kết quả Agent trên Dev và Test

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---:|---|---|---|---|---|
| **Agent / Dev (17 câu)** | 16/16 (100%) | 1.12 | 0: 1 câu, 1: 5 câu, 2: 8 câu (ĐTB: **1.50/2**) | 14/14 (100%) | 9/14 (64.3%) | 2/2 (100%) | 0/14 (0%) |
| **Agent / Test (7 câu)** | 7/7 (100%) | 1.00 | 0: 1 câu, 1: 2 câu, 2: 2 câu (ĐTB: **1.20/2**) | 11/11 (100%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20%) |

*Ghi chú:* 1 lỗi provider error xuất hiện trên 1 câu dev do timeout mạng tạm thời được tách riêng theo đúng chuẩn hướng dẫn (`starter/agent_metrics.py`).

### Bảng ví dụ chi tiết các trường hợp tiêu biểu

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q01** | Direct (Dev) | `search_hybrid(query="refined oil product flows Gulf region September")` -> 4 hits | "Refined oil product flows from the Gulf region fell to less than half of preconflict levels in September... [cmo-october-2026-p9-fixed-0001]" | **Grade 2**: Câu trả lời đầy đủ, số liệu chính xác tuyệt đối, trích dẫn đúng nguồn. |
| **q_unans_dev_1** | Unanswerable (Dev) | `search_hybrid(query="impact of Chilean copper mine labor strikes 2026")` -> 4 hits | "The report mentions copper price trends and mining disruptions in Chile, but does NOT specifically detail the impact of Chilean copper mine labor strikes... Therefore, insufficient evidence." | **Abstention đúng**: Agent nhận biết tài liệu không có thông tin về cuộc đình công cụ thể và từ chối trả lời. |
| **q18** | Condition (Test) | `search_hybrid(query="oil prices average 105 dollars per barrel 2027")` -> 4 hits | "The provided document does not contain information suggesting that oil prices will average $105 per barrel in 2027. Instead, it forecasts Brent to recede to $79... [cmo-october-2026-p70-fixed-0001]" | **Grade 0 (Từ chối nhầm)**: Chunk chứa 'adverse scenario' ($105/bbl) nằm ở trang 10 nhưng retrieval chỉ mang về trang 70 (kịch bản cơ sở $79), dẫn tới agent tưởng nhầm là tài liệu không có. |
| **q_unans_test_2** | Unanswerable (Test) | `search_hybrid(query="revised official OPEC crude oil output quota August 2026")` -> 4 hits | "The provided document does not contain information about a revised official OPEC crude oil output quota for member nations announced in August 2026. Therefore, there is insufficient evidence to answer." | **Abstention đúng**: Agent từ chối khẳng định vô căn cứ, không bịa đặt số liệu hạn ngạch OPEC. |

### Thống kê tài nguyên, thời gian và chi phí:
- **Thời gian thực thi:**
  - Dev (17 câu): Tổng thời gian 59.67s, trung bình **3.51s / câu**.
  - Test (7 câu): Tổng thời gian 24.38s, trung bình **3.48s / câu**.
- **Chi phí Token & LLM:**
  - Dev: 25,144 prompt tokens, 1,960 completion tokens $\rightarrow$ **$0.00247**
  - Test: 10,304 prompt tokens, 992 completion tokens $\rightarrow$ **$0.00107**
  - Tổng chi phí toàn bộ quá trình chạy benchmark agent: **~$0.0035** (rất tiết kiệm).
- **Đường dẫn artifact & trace:**
  - Dữ liệu câu hỏi: `data/benchmark_questions_24.json` và `data/converted/questions.json`
  - Dữ liệu chunks: `data/converted/chunks_a.json`
  - Kết quả Dev: `starter/results/agent-dev/traces.json`, `summary.json`, `reviews.json`
  - Kết quả Test: `starter/results/agent-test/traces.json`, `summary.json`, `reviews.json`

---

## 4. Hạn chế và Hướng cải tiến

- **Hạn chế:**
  - Các câu hỏi điều kiện phức tạp (như kịch bản rủi ro $105 ở câu `q18`) đòi hỏi agent phải thực hiện query rewriting (viết lại câu hỏi từ "105 dollars" thành "adverse scenario / upside risk") thay vì chỉ tìm kiếm trực diện.
  - Kích thước chunk 500 ký tự có thể cắt ngang một bảng số liệu, khiến agent chỉ nhìn thấy một phần dữ kiện.
- **Hướng cải tiến:**
  - Tích hợp thêm bước Query Transformation / Sub-query Decomposition trước khi gọi retrieval.
  - Áp dụng Cross-Encoder Reranker (như BGE-Reranker hoặc Cohere Rerank) để tái sắp xếp các candidate trước khi cắt theo context budget.
