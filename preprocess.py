"""preprocess.py  — bước [0] PREPROCESSING

Đọc file .txt, chuẩn hóa encoding, GIỮ NGUYÊN offset ký tự theo file gốc.

NGUYÊN TẮC CỐT LÕI (CLAUDE.md mục 6.5):
- `position` trong output cuối PHẢI tham chiếu theo file .txt gốc.
- Cách an toàn nhất: KHÔNG xóa whitespace/newline khỏi chuỗi canonical. Chuỗi
  canonical (`text`) do hàm này trả về chính là hệ tọa độ ký tự cho toàn pipeline.
- Chỉ xử lý encoding: decode UTF-8, loại BOM. KHÔNG dịch \r\n -> \n trên chuỗi
  canonical (việc đó làm lệch offset). Bản `debug_text` (gộp dòng trống) chỉ để
  đọc cho dễ khi debug, TUYỆT ĐỐI không dùng để tính position.
"""

import os
import re
import unicodedata

_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def load_raw(path: str) -> str:
    """Đọc file, decode UTF-8 (loại BOM nếu có), KHÔNG dịch newline.

    Đọc bytes rồi decode để tránh universal-newline translation của text mode
    (giữ nguyên \r\n → offset khớp file gốc). NFC-normalize dấu tiếng Việt để
    thống nhất so khớp chuỗi, nhưng NFC không đổi độ dài với văn bản đã tổ hợp.
    """
    with open(path, "rb") as f:
        data = f.read()
    # utf-8-sig: strip BOM ở đầu nếu có; không đụng phần còn lại.
    text = data.decode("utf-8-sig", errors="replace")
    # NFC: đưa về dạng tổ hợp chuẩn (đa số file y tế VN đã ở NFC, giữ ổn định).
    text = unicodedata.normalize("NFC", text)
    return text


def normalize_for_debug(text: str) -> str:
    """Bản đọc cho dễ: gộp >=3 dòng trống liên tiếp thành 1 dòng trống.

    CHỈ dùng để in/log khi debug. KHÔNG dùng cho position (đổi độ dài chuỗi).
    """
    return _MULTI_BLANK_RE.sub("\n\n", text)


def load_and_clean(path: str) -> dict:
    """Trả về dict cho 1 file.

    {
      "filename": tên file (không path),
      "path": đường dẫn gốc,
      "text": chuỗi CANONICAL (hệ tọa độ position — dùng cho cả pipeline),
      "debug_text": bản gộp dòng trống chỉ để đọc,
      "n_chars": độ dài chuỗi canonical,
    }
    """
    text = load_raw(path)
    return {
        "filename": os.path.basename(path),
        "path": path,
        "text": text,
        "debug_text": normalize_for_debug(text),
        "n_chars": len(text),
    }


def iter_input_files(input_dir: str):
    """Yield từng dict load_and_clean cho mọi file .txt trong input_dir (đã sort)."""
    names = sorted(fn for fn in os.listdir(input_dir) if fn.lower().endswith(".txt"))
    for name in names:
        yield load_and_clean(os.path.join(input_dir, name))
