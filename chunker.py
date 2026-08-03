"""chunker.py  — bước [1] CHUNKING

Tách câu bằng underthesea.sentence_tokenize và LƯU OFFSET GỐC của mỗi câu trong
chuỗi canonical (do preprocess trả về). Offset này để bước [3] position_recovery
quy vị trí entity về toàn cục.

Mỗi chunk: {"text": <câu>, "start": <int>, "end": <int>, "idx": <int>}
với text == canonical_text[start:end] (đảm bảo khớp chính xác).
"""

import re

# Ngưỡng cắt nhỏ câu quá dài (article/FAQ có câu rất dài) để prompt không tràn ctx.
_HARD_MAX_CHARS = 1200


def _sentence_tokenize(text: str):
    """Trả về list câu (chuỗi). Fallback nếu thiếu underthesea."""
    try:
        from underthesea import sentence_tokenize
        return sentence_tokenize(text)
    except Exception:  # noqa: BLE001 — thiếu lib hoặc lỗi runtime
        # Fallback thô: tách theo xuống dòng và dấu kết câu.
        parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
        return [p for p in parts if p.strip()]


def _flexible_locate(haystack: str, needle: str, start: int):
    """Tìm needle trong haystack từ vị trí start. Trả (s, e) hoặc None.

    Thử str.find trước; nếu trượt (do underthesea chuẩn hóa khoảng trắng), dùng
    regex nới lỏng mọi chuỗi khoảng trắng.
    """
    idx = haystack.find(needle, start)
    if idx != -1:
        return idx, idx + len(needle)
    tokens = [re.escape(t) for t in needle.split()]
    if not tokens:
        return None
    pattern = r"\s+".join(tokens)
    m = re.search(pattern, haystack[start:])
    if m:
        return start + m.start(), start + m.end()
    return None


def _split_long(chunk_text: str, start: int, max_chars: int):
    """Nếu câu dài quá max_chars, cắt tiếp theo dấu câu/khoảng trắng, giữ offset."""
    if len(chunk_text) <= max_chars:
        return [(chunk_text, start)]
    pieces = []
    cursor = 0
    n = len(chunk_text)
    while cursor < n:
        end = min(cursor + max_chars, n)
        if end < n:
            # lùi về ranh giới khoảng trắng gần nhất để không cắt giữa từ
            back = chunk_text.rfind(" ", cursor, end)
            if back > cursor:
                end = back
        sub = chunk_text[cursor:end]
        if sub.strip():
            pieces.append((sub, start + cursor))
        cursor = end
    return pieces


def chunk_text(text: str, max_chars: int = _HARD_MAX_CHARS):
    """Tách câu + gắn offset. Trả về list chunk dict.

    Bất biến: text[chunk['start']:chunk['end']] == chunk['text'].
    """
    sentences = _sentence_tokenize(text)
    chunks = []
    cursor = 0
    idx = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        loc = _flexible_locate(text, sent, cursor)
        if loc is None:
            # không định vị được → bỏ qua an toàn (hiếm; tránh offset sai)
            continue
        s, e = loc
        located = text[s:e]  # dùng lát cắt thực tế để đảm bảo bất biến
        cursor = e
        for sub_text, sub_start in _split_long(located, s, max_chars):
            chunks.append({
                "text": sub_text,
                "start": sub_start,
                "end": sub_start + len(sub_text),
                "idx": idx,
            })
            idx += 1
    return chunks
