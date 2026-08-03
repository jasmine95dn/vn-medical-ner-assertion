"""position_recovery.py  — bước [3] POSITION RECOVERY

Định vị mỗi entity (text do LLM trả) trong chunk, rồi quy về offset toàn cục
(chunk_start + offset cục bộ). RULE-BASED, KHÔNG dùng model.

Yêu cầu đặc thù:
- KHÔNG dedupe: nhiều entity trùng text phải map vào các lần xuất hiện KHÁC NHAU
  → theo dõi các span đã "claim" để không hai entity trỏ cùng vị trí.
- str.find() trước; fuzzy fallback (khoảng trắng nới lỏng, rồi difflib) khi LLM
  chỉnh nhẹ text (thêm/bớt space, đổi hoa thường).
"""

import difflib
import re

_FUZZY_THRESHOLD = 0.82


def _exact_occurrences(hay: str, needle: str):
    """Mọi (s, e) khớp chính xác."""
    out = []
    start = 0
    while True:
        i = hay.find(needle, start)
        if i == -1:
            break
        out.append((i, i + len(needle)))
        start = i + 1
    return out


def _flex_occurrences(hay: str, needle: str):
    """Khớp nới lỏng khoảng trắng (\\s+ giữa các token)."""
    tokens = [re.escape(t) for t in needle.split()]
    if not tokens:
        return []
    pattern = r"\s+".join(tokens)
    return [(m.start(), m.end()) for m in re.finditer(pattern, hay)]


def _fuzzy_best(hay: str, needle: str):
    """Trượt cửa sổ ~ độ dài needle, tìm span có ratio cao nhất >= ngưỡng.

    Trả về [(s, e)] earliest-best hoặc [].
    """
    n = len(needle)
    if n == 0 or n > len(hay):
        return []
    sm = difflib.SequenceMatcher(b=needle.lower())
    best = None
    # cho phép cửa sổ dao động +-25% độ dài
    lo = max(1, int(n * 0.75))
    hi = min(len(hay), int(n * 1.25) + 1)
    for size in {n, lo, hi}:
        for s in range(0, len(hay) - size + 1):
            window = hay[s:s + size]
            sm.set_seq1(window.lower())
            r = sm.quick_ratio()
            if r < _FUZZY_THRESHOLD:
                continue
            r = sm.ratio()
            if r >= _FUZZY_THRESHOLD and (best is None or r > best[0]):
                best = (r, s, s + size)
    if best:
        return [(best[1], best[2])]
    return []


def _pick_unclaimed(occurrences, claimed):
    """Chọn span sớm nhất không đè lên span đã claim."""
    for s, e in sorted(occurrences):
        if all(e <= cs or s >= ce for cs, ce in claimed):
            return s, e
    return None


def recover_positions(entities, chunk):
    """Gắn 'position' toàn cục cho từng entity trong 1 chunk.

    entities: list dict {"text","type","assertions"?}
    chunk:    dict {"text","start",...}

    Trả về list entity đã thêm:
      - "position": [gstart, gend]  (theo canonical text toàn file)
      - "_matched_text": chuỗi thực tế trong file (khớp chính xác position)
      - "_match": "exact"|"flex"|"fuzzy"
    Entity không định vị được sẽ bị BỎ (tránh position sai).
    """
    hay = chunk["text"]
    base = chunk["start"]
    claimed = []
    out = []
    for ent in entities:
        text = (ent.get("text") or "").strip()
        if not text:
            continue
        occ = _exact_occurrences(hay, text)
        match_kind = "exact"
        if not occ:
            occ = _flex_occurrences(hay, text)
            match_kind = "flex"
        if not occ:
            occ = _fuzzy_best(hay, text)
            match_kind = "fuzzy"
        if not occ:
            continue  # không định vị được → bỏ
        picked = _pick_unclaimed(occ, claimed)
        if picked is None:
            continue  # mọi occurrence đã bị entity khác chiếm
        s, e = picked
        claimed.append((s, e))
        new_ent = dict(ent)
        new_ent["position"] = [base + s, base + e]
        new_ent["_matched_text"] = hay[s:e]
        new_ent["_match"] = match_kind
        out.append(new_ent)
    return out
