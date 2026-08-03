"""candidate_mapping.py  — bước [4a] rank fusion + [4b] re-rank

Chỉ chạy cho type CHẨN_ĐOÁN (ICD-10) và THUỐC (RxNorm).
- CHẨN_ĐOÁN: IcdIndex.search (ViHealthBERT + e5) → rank fusion → re-rank Qwen3-8B.
- THUỐC: rxnorm_candidates → re-rank Qwen3-8B.

KHÔNG tự đọc Excel: nhận `icd_index` đã build sẵn từ icd_rxnorm_index.build_icd_index.
KHÔNG để model tự bịa mã ngoài danh sách: lọc kết quả re-rank theo danh sách đầu vào.
"""

import json

from icd_rxnorm_index import rxnorm_candidates


def _fuse(search_result, final_k: int = 10):
    """Rank fusion 2 danh sách top-5.

    Ưu tiên candidate xuất hiện ở CẢ 2 danh sách; sau đó bổ sung theo rank trung
    bình. Trả về list (code, name) đã dedupe theo code, tối đa final_k.
    """
    vh = search_result.get("vihealthbert", [])
    e5 = search_result.get("e5", [])

    rank_vh = {code: i for i, (code, _n, _s) in enumerate(vh)}
    rank_e5 = {code: i for i, (code, _n, _s) in enumerate(e5)}
    names = {}
    for code, name, _ in vh + e5:
        names.setdefault(code, name)

    both = set(rank_vh) & set(rank_e5)
    only = (set(rank_vh) | set(rank_e5)) - both

    BIG = 999

    def avg_rank(code):
        return (rank_vh.get(code, BIG) + rank_e5.get(code, BIG)) / 2.0

    ordered = sorted(both, key=avg_rank) + sorted(only, key=avg_rank)
    fused = []
    seen = set()
    for code in ordered:
        if code in seen:
            continue
        seen.add(code)
        fused.append((code, names.get(code, "")))
        if len(fused) >= final_k:
            break
    return fused


def _rerank_llm(entity_text, context, candidates, entity_type):
    """Dùng Qwen3-8B chọn/sắp xếp mã đúng nhất trong `candidates`.

    candidates: list (code, name). Trả về list code đã sắp (best first), CHỈ gồm
    mã có trong candidates. Nếu model lỗi → giữ nguyên thứ tự fusion.
    """
    from llm_client import chat_json

    valid_codes = [c for c, _ in candidates]
    if not valid_codes:
        return []

    listing = "\n".join(f"{i+1}. [{c}] {n}" for i, (c, n) in enumerate(candidates))
    kind = "mã ICD-10" if entity_type == "CHẨN_ĐOÁN" else "mã RxNorm"
    messages = [
        {"role": "system", "content":
            "Bạn là chuyên gia mã hóa y khoa. Chọn mã phù hợp nhất cho thực thể, "
            "CHỈ được chọn trong danh sách cho sẵn, KHÔNG bịa mã mới."},
        {"role": "user", "content":
            f"Thực thể ({kind}): \"{entity_text}\"\n"
            f"Ngữ cảnh: \"{context}\"\n\n"
            f"Danh sách ứng viên:\n{listing}\n\n"
            "Sắp xếp các mã theo độ phù hợp giảm dần (mã đúng nhất trước). "
            "Chỉ dùng mã trong danh sách. Trả về JSON: "
            '{"ranked_codes": ["<mã>", ...]}'},
    ]
    data = chat_json(messages, default={})
    ranked = data.get("ranked_codes", []) if isinstance(data, dict) else []

    valid_set = set(valid_codes)
    out, seen = [], set()
    for c in ranked:
        c = str(c).strip()
        if c in valid_set and c not in seen:
            seen.add(c)
            out.append(c)
    # bổ sung mã còn lại theo thứ tự fusion (không mất candidate)
    for c in valid_codes:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def map_candidates(entity, context, icd_index, top_k=5, final_k=10, use_llm_rerank=True):
    """Trả về list mã (candidates) cho 1 entity.

    entity:  dict có 'text', 'type'.
    context: câu chứa entity (giúp re-rank).
    icd_index: IcdIndex đã build (dùng cho CHẨN_ĐOÁN). Có thể None nếu không cần.
    """
    etype = entity.get("type")
    text = (entity.get("text") or "").strip()
    if not text:
        return []

    if etype == "CHẨN_ĐOÁN":
        if icd_index is None:
            return []
        search_result = icd_index.search(text, top_k=top_k)
        fused = _fuse(search_result, final_k=final_k)
        if not fused:
            return []
        if use_llm_rerank:
            return _rerank_llm(text, context, fused, etype)
        return [c for c, _ in fused]

    if etype == "THUỐC":
        cands = rxnorm_candidates(text, max_results=final_k)
        pairs = [(rxcui, name) for rxcui, name, _score in cands]
        if not pairs:
            return []
        if use_llm_rerank:
            return _rerank_llm(text, context, pairs, etype)
        return [c for c, _ in pairs]

    return []  # TRIỆU_CHỨNG / TÊN_XÉT_NGHIỆM / KẾT_QUẢ_XÉT_NGHIỆM → không có candidates
