# CLAUDE.md — Vietnamese Medical NER/Assertion/Mapping Pipeline

Hướng dẫn dự án cho Claude Code. Đọc kỹ trước khi viết bất kỳ file nào.

## 1. Mục tiêu

Xây pipeline nhận văn bản y khoa tiếng Việt tự do (`.txt`), xuất JSON chứa các khái niệm y tế đã chuẩn hóa. Input: 100 file `.txt` trong `data/input/`. Output: 100 file `.json` tương ứng trong `output/`.

## 2. Ràng buộc bắt buộc (không được vi phạm)

- **Chỉ dùng model LLM/agent tự host (self-host), tối đa 9B tham số.** KHÔNG gọi API ngoài (Gemini, Claude, GPT, DeepSeek, Kimi, Grok...) cho phần sinh văn bản.
- Không fine-tune (không có dữ liệu train có nhãn) — chỉ dùng few-shot prompting.
- Không có GPU cá nhân — chạy trên Kaggle Notebook (30h GPU/tuần, T4/P100) hoặc Colab free tier.
- Embedding model cho retrieval (encoder-only, không sinh văn bản) tạm coi KHÔNG nằm trong giới hạn 9B — quyết định dùng dứt khoát (ViHealthBERT + multilingual-e5-base), không chuyển sang BM25/fuzzy matching dù chưa có xác nhận rõ từ BTC, vì lexical matching kém ổn định hơn với tên y khoa dài.

## 3. Schema output

Mỗi entity trong JSON output gồm:
- `text` (string): cụm từ xác định trong input
- `position` ([int, int]): vị trí bắt đầu/kết thúc theo ký tự, 0-indexed, tính trên toàn bộ file gốc
- `type` (string): một trong 5 giá trị — `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`
- `assertions` (list of string, tối đa 3 phần tử, CHỈ áp dụng cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC): `isNegated`, `isFamily`, `isHistorical`
- `candidates` (list of string, CHỈ áp dụng cho CHẨN_ĐOÁN [mã ICD-10] và THUỐC [mã RxNorm])

Bảng field theo type:

| type | assertions? | candidates? |
|---|:---:|:---:|
| TRIỆU_CHỨNG | có | không |
| CHẨN_ĐOÁN | có | có |
| THUỐC | có | có |
| TÊN_XÉT_NGHIỆM | không | không |
| KẾT_QUẢ_XÉT_NGHIỆM | không | không |

## 4. Model đã chọn (self-host)

**Model sinh văn bản chính (dùng cho MỌI bước cần LLM — NER, type, assertion, re-rank):**
- **Qwen/Qwen3-8B** (HuggingFace), 8.2B tham số tổng (~6.95B non-embedding), dense decoder-only, 36 layer, GQA 32 query-head/8 KV-head, hidden dim 4096, context native 32,768 token (mở rộng 131,072 qua YaRN), hỗ trợ 119 ngôn ngữ, hỗ trợ JSON mode.
- Serve qua **Ollama**: `ollama pull qwen3:8b`
- KHÔNG đổi sang model khác trừ khi có bằng chứng cụ thể Qwen3-8B cho kết quả tệ trên domain y tế Việt. Backup nếu cần: SeaLLMs-v3-7B hoặc Vistral-7B-Chat.

**Embedding model cho retrieval (bước candidate mapping, dùng CẢ 2 song song):**
- `demdecuong/vihealthbert-base-word` (~135M tham số) — domain-specific y tế Việt, cần input đã qua `underthesea.word_tokenize()` trước khi encode
- `intfloat/multilingual-e5-base` (~278M tham số) — đa ngôn ngữ tổng quát, encode trực tiếp câu gốc không cần tách từ trước

Hợp nhất kết quả 2 model bằng rank fusion: ưu tiên candidate xuất hiện ở cả 2 danh sách top-5, sau đó bổ sung theo thứ hạng trung bình, gộp thành danh sách tối đa ~8-10 candidate đưa sang bước re-rank.

## 5. Tokenizer

Chỉ dùng **underthesea** cho toàn bộ nhu cầu tokenization:
- `underthesea.sentence_tokenize()` — tách câu ở bước chunking
- `underthesea.word_tokenize()` — tách từ chuẩn bị input cho ViHealthBERT

KHÔNG cài VnCoreNLP/Java. Benchmark độc lập (huybik.github.io) cho thấy underthesea đạt 80.0% độ chính xác tách từ so với VnCoreNLP 78.4%, đồng thời nhanh hơn ~2 lần.

Cài đặt: `pip install underthesea --break-system-packages` (KHÔNG cài extra `[agent]`/`[agent-server]` — đó là framework AI agent riêng biệt không liên quan đến chunking).

## 6. Sơ đồ pipeline

```
[0] PREPROCESSING (đọc file, chuẩn hóa UTF-8/whitespace)
        │
[1] CHUNKING (underthesea.sentence_tokenize + lưu offset gốc)
        │
[2] NER + TYPE + ASSERTION (Qwen3-8B, 1 lần gọi/chunk, gộp 3 việc trong 1 JSON,
    few-shot prompting, KHÔNG ensemble/multi-model)
        │
[3] POSITION RECOVERY (str.find() + fuzzy fallback, rule-based, KHÔNG dùng model)
        │
[4a] RETRIEVAL — chỉ với type=CHẨN_ĐOÁN/THUỐC
    (ViHealthBERT + multilingual-e5-base song song → rank fusion → top ~8-10 candidate)
        │
[4b] RE-RANK (Qwen3-8B, chọn candidate đúng nhất trong danh sách đã lọc,
    KHÔNG để model tự bịa mã ngoài danh sách)
        │
[5] MERGE + VALIDATE (ghép chunk, check schema, check text khớp position)
        │
Output: file.json
```

## 6.5. Ví dụ cụ thể cho bước [0] Preprocessing

**Trước khi viết `preprocess.py`, khảo sát nhanh 5-10 file trong `data/input/` (chọn rải rác, không chỉ vài file đầu)** để xác nhận: (1) có bao nhiêu dạng cấu trúc khác nhau (article/FAQ dài vs note ngắn kiểu bệnh án vs trộn lẫn), (2) còn trường hợp censor/ẩn thông tin (dấu `*`) ở file khác ngoài ví dụ dưới đây không, (3) encoding có nhất quán UTF-8 không (hay có BOM marker/ký tự lạ), (4) whitespace/xuống dòng có kiểu bất thường nào khác (`\r\n` lẫn `\n`, tab, nhiều dấu cách liên tiếp). Không cần xem hết 100 file, chỉ cần đủ mẫu để không viết code chỉ khớp với 1 dạng văn bản duy nhất.

Input thực tế đa dạng hơn ví dụ mẫu trong đề bài. Ví dụ đã quan sát được từ file test thật (bài về thiếu men G6PD):

- **Từ bị censor bằng dấu `*`**: `"Thuốc giảm đau, hạ sốt chứa ******* hoặc **********"` — GIỮ NGUYÊN chuỗi dấu `*`, KHÔNG cố đoán/thay thế tên thuốc thật. Nếu model detect đây là `THUỐC`, để `candidates: []` (không tìm được mã vì text đã ẩn).
- **Không có bệnh nhân cụ thể**: nhiều file là văn bản kiến thức chung (dạng FAQ/article), không giống ví dụ mẫu đề bài (luôn có "bệnh nhân nam 70 tuổi..."). Khi không có patient context rõ ràng, `assertions: []` là kết quả hợp lệ, không phải lỗi.
- **Đoạn văn dài, nhiều dòng trống giữa đoạn**: chuẩn hóa nhiều dòng trống liên tiếp thành 1 dòng trống, nhưng KHÔNG xóa hoàn toàn xuống dòng (ảnh hưởng đến offset tính theo ký tự — phải giữ nguyên độ dài chuỗi gốc nếu có thể, hoặc nếu chuẩn hóa thì phải tính lại offset tương ứng, không dùng offset theo file đã chuẩn hóa mà báo cáo theo file gốc chưa chuẩn hóa).
- **Entity lặp lại nhiều lần với cách diễn đạt hơi khác** (VD: "vàng da, vàng mắt" xuất hiện dạng "vàng da vàng mắt", "vàng da nặng", "vàng da sơ sinh" ở nhiều câu khác nhau): KHÔNG dedupe, mỗi lần xuất hiện là 1 entity riêng trong output.
- **QUAN TRỌNG về offset**: mọi bước chuẩn hóa whitespace ở [0] PHẢI đảm bảo `position` cuối cùng trong output JSON tham chiếu đúng theo **file `.txt` gốc chưa qua xử lý**, không phải theo bản đã chuẩn hóa nội bộ. Nếu preprocessing làm thay đổi độ dài chuỗi (xóa bớt whitespace thừa), cần giữ một mapping offset giữa bản gốc và bản đã xử lý, hoặc đơn giản nhất là KHÔNG xóa whitespace/newline (chỉ dùng để dễ đọc khi debug), giữ nguyên bản gốc làm input thật cho toàn bộ pipeline.

## 7. Dataset

| Dataset | Vị trí | Vai trò |
|---|---|---|
| ICD-10 tiếng Việt (Excel, QĐ 4469/QĐ-BYT 28/10/2020, bản BV Quảng Trị) | `data/icd10_vn.xlsx` | Nguồn candidate cho CHẨN_ĐOÁN, encode bằng cả 2 embedding model |
| RxNorm | Gọi REST API `https://rxnav.nlm.nih.gov/REST/` trực tiếp, không tải file | Nguồn candidate cho THUỐC |
| 100 file test | `data/input/*.txt` | Input chính |
| Few-shot examples | `data/few_shot_examples.json` | Đưa vào prompt bước [2] — **cần prompt sinh riêng** (gọi Qwen3-8B 1 lần, không phải lúc chạy pipeline chính), review lại output trước khi dùng |
| Bảng trigger words assertion | `data/assertion_triggers.json` | isNegated: không/chưa từng/phủ nhận; isFamily: bố/mẹ/anh chị em/người nhà; isHistorical: tiền sử/đã từng/trước đây — **tự liệt kê tay, KHÔNG cần prompt sinh** |
| Negative examples (chống false trigger) | `data/negative_examples.json` | Câu có từ phủ định nhưng KHÔNG áp dụng isNegated — **cần prompt sinh riêng**, review kỹ vì đây là case khó (dễ sinh sai ý đồ với model 8B) |
| Validation set tự gán nhãn (5-10 câu) | `data/validation_set.json` | Sanity-check trước khi nộp — **KHÔNG dùng LLM sinh cả câu lẫn nhãn** (mất tính khách quan để validate); lấy câu thật từ `hungnm/vietnamese-medical-qa` hoặc chính 100 file test, tự gán nhãn tay |

RxNorm chỉ nhận diện tên hoạt chất/biệt dược tiếng Anh — nếu API không trả kết quả, thử tách tên hoạt chất chính từ tên biệt dược trước khi query lại.

**Dataset tham khảo bổ sung (KHÔNG phải ground truth chính thức, chỉ dùng để tham khảo/đa dạng hóa test case, không train):**

| Dataset | Nguồn | Vai trò |
|---|---|---|
| acrDrAid (135 bộ từ khóa viết tắt) | github.com/demdecuong/vihealthbert/tree/main/dataset/acrDrAid | Tham khảo khi viết thêm few-shot examples về viết tắt y khoa |
| hungnm/vietnamese-medical-qa (9.335 cặp QA thật) | HuggingFace `hungnm/vietnamese-medical-qa` | Lấy mẫu 10-20 câu hỏi/trả lời thật để bổ sung validation set (nhanh hơn tự viết synthetic) và test chunking trên văn phong tự nhiên đa dạng |
| ViMQ (có nhãn NER + Intent Classification riêng) | github.com/tadeephuy/ViMQ | Tham khảo cách định nghĩa ranh giới entity — KHÔNG dùng trực tiếp làm label vì schema khác 5 type của đề bài |
| urnus11/Vietnamese-Healthcare (QA + article vinmec.com) | HuggingFace `urnus11/Vietnamese-Healthcare` | Bổ sung test case dạng article dài, đa dạng hóa ngoài 100 file chính thức |

**Hướng dẫn cụ thể cách lấy data từ từng nguồn:**

```python
# 1. hungnm/vietnamese-medical-qa — dùng cho validation_set.json
from datasets import load_dataset
ds = load_dataset("hungnm/vietnamese-medical-qa")
# Lấy ngẫu nhiên 10-20 dòng, ưu tiên câu hỏi/trả lời có độ dài vừa phải (không quá ngắn/quá dài)
# → tự đọc và gán nhãn tay theo schema 5 type + assertions, lưu vào data/validation_set.json

# 2. urnus11/Vietnamese-Healthcare — dùng để test chunking đa dạng
ds2 = load_dataset("urnus11/Vietnamese-Healthcare")
# Lấy vài chục bài article dài, chạy thử qua chunker.py xem tách câu ổn không
# (không cần gán nhãn, chỉ dùng để test tính bền của [1] Chunking, không đưa vào output chính thức)

# 3. acrDrAid — clone repo, đọc trực tiếp file text/json trong thư mục dataset
# git clone https://github.com/demdecuong/vihealthbert.git
# Xem qua danh sách 135 bộ viết tắt, chọn lọc ra ~10-15 cái phổ biến nhất
# → viết tay thành vài dòng bổ sung trong prompt few-shot (không cần code xử lý tự động)

# 4. ViMQ — chỉ xem qua để tham khảo, không cần code trích xuất
# git clone https://github.com/tadeephuy/ViMQ.git
# Đọc vài ví dụ trong annotation guideline của họ để hiểu cách xử lý ranh giới entity phức tạp
```

**Lưu ý**: bước 1 và 2 cần cài thêm `pip install datasets --break-system-packages` (thư viện `datasets` của HuggingFace). Bước 3 và 4 chỉ cần đọc thủ công, không cần code tự động hóa vì quy mô nhỏ (135 bộ viết tắt, vài chục annotation guideline) — tự đọc và chọn lọc tay nhanh hơn viết script.

## 8. Setup môi trường

**Cấu trúc git repo (code) — tách riêng khỏi data nặng:**

```
vn-medical-ner/
├── CLAUDE.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── few_shot_examples.json
│   ├── assertion_triggers.json
│   ├── negative_examples.json
│   └── validation_set.json
├── prompts/
│   ├── data_generation_prompt.py
│   └── ner_assertion_prompt.py
├── preprocess.py
├── chunker.py
├── llm_client.py
├── position_recovery.py
├── icd_rxnorm_index.py
├── candidate_mapping.py
├── merge_validate.py
└── main.py
```

**KHÔNG đưa vào git repo** (tách riêng thành 1 Kaggle Dataset khác qua giao diện "New Dataset" → "Add Data"):
- 100 file `.txt` test (`data/input/*.txt`) — dữ liệu ban tổ chức cung cấp, không nên đẩy công khai lên GitHub
- File Excel ICD-10 (`icd10_vn.xlsx`) — nặng, không cần version-control qua git

`.gitignore` cần có:
```
output/
__pycache__/
*.pyc
.env
```

Trong `main.py`, trỏ đường dẫn tới Kaggle Dataset đã add:
```python
DATA_DIR = "/kaggle/input/ten-dataset-cua-ban"
INPUT_DIR = f"{DATA_DIR}/input"          # 100 file .txt
ICD10_PATH = f"{DATA_DIR}/icd10_vn.xlsx"
```

**Quy trình trên Kaggle Notebook:**
```bash
# Bật GPU T4/P100 trong Settings trước
!git clone https://github.com/<user>/vn-medical-ner.git
%cd vn-medical-ner
# Add Data (100 file test + Excel ICD-10) qua giao diện Kaggle, KHÔNG qua git

curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3:8b

pip install pandas openpyxl underthesea sentence-transformers requests ollama datasets --break-system-packages

!python main.py
```

Khi cần sửa code sau khi debug: sửa local → `git push` → trong Kaggle chạy `!git pull` (không cần clone lại/upload lại từ đầu).

Dùng **"Save & Run All" (Commit)** khi chạy full 100 file để job chạy độc lập trên hạ tầng Kaggle, không cần giữ tab trình duyệt mở — chỉ dùng chế độ tương tác lúc code/debug từng bước.

**File nào cần path tới data nào (path phải truyền vào qua tham số, KHÔNG hardcode trong từng file):**

| File | Cần path tới 100 file `.txt`? | Cần path tới Excel ICD-10? |
|---|:---:|:---:|
| `preprocess.py` | Có — nhận `INPUT_DIR` làm tham số | Không |
| `icd_rxnorm_index.py` | Không | Có — nhận `ICD10_PATH` làm tham số, đây là **file duy nhất** đọc Excel |
| `candidate_mapping.py` | Không | Không — chỉ `import` index đã build sẵn từ `icd_rxnorm_index.py`, không tự đọc lại Excel |
| `main.py` | Có — orchestrate toàn bộ, loop qua `INPUT_DIR` | Có — định nghĩa `ICD10_PATH` và truyền xuống `icd_rxnorm_index.py` |

Định nghĩa path tập trung 1 lần duy nhất ở `main.py`, các file khác viết dạng hàm nhận path làm tham số:

```python
# main.py
DATA_DIR = "/kaggle/input/ten-dataset-cua-ban"
INPUT_DIR = f"{DATA_DIR}/input"
ICD10_PATH = f"{DATA_DIR}/icd10_vn.xlsx"

from preprocess import load_and_clean
from icd_rxnorm_index import build_icd_index
from candidate_mapping import map_candidates
```

## 9. File cần viết, theo thứ tự

1. `preprocess.py` — đọc file, chuẩn hóa encoding/whitespace
2. `chunker.py` — underthesea sentence_tokenize + offset tracking
3. `prompts/data_generation_prompt.py` — prompt riêng (chạy 1 lần lúc chuẩn bị data, KHÔNG dùng lúc chạy pipeline chính) để gọi Qwen3-8B sinh `few_shot_examples.json` và `negative_examples.json`; sinh xong cần tự review lại trước khi dùng
4. `prompts/ner_assertion_prompt.py` — prompt template dùng lúc chạy pipeline chính thức: few-shot examples + bảng trigger words + negative examples, ép model liệt kê MỌI lần xuất hiện entity kể cả lặp lại (không dedupe)
5. `llm_client.py` — wrapper gọi Ollama local API (model Qwen3-8B, dùng chung cho bước [2] và [4b])
6. `position_recovery.py` — str.find() + fuzzy fallback, tính lại offset toàn cục theo offset chunk
7. `icd_rxnorm_index.py` — load Excel ICD-10, encode bằng cả ViHealthBERT (qua underthesea.word_tokenize trước) và multilingual-e5-base, build 2 index riêng; hàm gọi RxNorm API
8. `candidate_mapping.py` — retrieval song song 2 model → rank fusion → gọi Qwen3-8B re-rank chọn candidate cuối
9. `merge_validate.py` — ghép kết quả các chunk, validate schema JSON đúng, validate text khớp offset trong file gốc
10. `main.py` — chạy toàn bộ pipeline cho 100 file, xuất JSON vào `output/`

## 10. Timeline

- **Hôm nay (2/8, tối)**: [0]-[1]-[2] chạy MVP (candidates để rỗng tạm) → nếu còn thời gian, làm luôn [4a]-[4b] → test trên 2-3 file trước khi chạy full 100 file → **nộp thử lần 1 ngay trong đêm nếu kịp**
- **3/8**: nếu tối 2/8 chưa kịp candidate mapping, hoàn thiện nốt [4a]-[4b] → chạy full 100 file → nộp lần 1 (nếu chưa nộp đêm qua) hoặc sửa lỗi theo score (nếu đã nộp)
- **4/8 (deadline)**: buffer cuối cùng, sửa lỗi rõ ràng nếu còn kịp, KHÔNG mở rộng ensemble/thêm model ở giai đoạn này

## 11. Rủi ro đã biết

- Model 8B nhỏ hơn nhiều so với model lớn (Gemini/Claude/GPT) → chất lượng NER/assertion/reasoning thấp hơn đáng kể — chấp nhận đây là giới hạn cứng từ luật chơi
- Văn bản test đa dạng format (article dài kiểu FAQ lẫn note ngắn kiểu bệnh án) — test chunking trên cả 2 dạng
- Model nhỏ dễ dedupe nhầm entity lặp lại — nhấn mạnh rõ trong prompt
- Giới hạn GPU miễn phí (30h/tuần Kaggle) — canh thời gian, tránh chạy thử lặp lại quá nhiều làm hết quota
- Chưa xác nhận với BTC liệu embedding retrieval model có tính vào giới hạn 9B — **quyết định: vẫn ưu tiên dùng embedding (ViHealthBERT + multilingual-e5-base)**, chấp nhận rủi ro này thay vì chuyển sang BM25/fuzzy matching, vì kinh nghiệm thực tế cho thấy fuzzy matching kém ổn định với tên y khoa dài (threshold >80 vẫn hay fail do tích lũy sai lệch edit-distance trên chuỗi dài), còn embedding nắm bắt ngữ nghĩa tốt hơn hẳn cho biến thể diễn đạt cùng nghĩa khác mặt chữ.
