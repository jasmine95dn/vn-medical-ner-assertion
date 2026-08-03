"""icd_rxnorm_index.py  — bước [4a] (phần dựng index)

File DUY NHẤT đọc Excel ICD-10. Build 2 index embedding song song:
  - ViHealthBERT (demdecuong/vihealthbert-base-word): input phải qua
    underthesea.word_tokenize (nối từ bằng "_") trước khi encode.
  - multilingual-e5-base (intfloat/multilingual-e5-base): encode trực tiếp,
    tuân thủ prefix "passage: " / "query: " của họ e5.

Cũng chứa hàm gọi RxNorm REST API cho THUỐC (không tải file).

candidate_mapping.py chỉ IMPORT index đã build, KHÔNG tự đọc Excel.
"""

import re

import numpy as np

# --------------------------------------------------------------------------- #
# Model cache (nặng — chỉ load 1 lần).
# --------------------------------------------------------------------------- #
_VIHEALTH = None
_E5 = None


def _get_vihealthbert():
    global _VIHEALTH
    if _VIHEALTH is None:
        from sentence_transformers import SentenceTransformer, models
        word_emb = models.Transformer("demdecuong/vihealthbert-base-word", max_seq_length=128)
        pooling = models.Pooling(word_emb.get_word_embedding_dimension(),
                                 pooling_mode_mean_tokens=True)
        _VIHEALTH = SentenceTransformer(modules=[word_emb, pooling])
    return _VIHEALTH


def _get_e5():
    global _E5
    if _E5 is None:
        from sentence_transformers import SentenceTransformer
        _E5 = SentenceTransformer("intfloat/multilingual-e5-base")
    return _E5


def _word_seg(text: str) -> str:
    """underthesea.word_tokenize dạng text (nối bằng '_') cho ViHealthBERT."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:  # noqa: BLE001
        return text


# --------------------------------------------------------------------------- #
# Đọc Excel ICD-10 (linh hoạt cột) + build index.
# --------------------------------------------------------------------------- #
# Chấp nhận cả ký hiệu dao găm †/sao * của ICD-10 (mã kép etiology/manifestation).
_CODE_RE = re.compile(r"^[A-Z]\d{2}(\.\d+)?[†✝*]?$")
# Ký hiệu dao găm/sao ở cuối mã — strip để ra mã ICD-10 chuẩn (A17.0† -> A17.0).
_CODE_MARK_RE = re.compile(r"[†✝*]+\s*$")


def _normalize_code(raw) -> str:
    """Bỏ ký hiệu †/* ở cuối và khoảng trắng thừa, trả mã ICD-10 chuẩn."""
    return _CODE_MARK_RE.sub("", str(raw).strip()).strip()

# Tên cột đã biết ở bản ICD-10 đã làm sạch (ma_benh / ten_benh_vi ...).
_KNOWN_CODE_COLS = ["ma_benh", "ma_icd10", "ma_icd", "icd10", "code", "ma"]
_KNOWN_NAME_COLS = ["ten_benh_vi", "ten_benh", "ten_vi", "ten", "name_vi", "disease_name_vi"]


def _resolve_columns(df, code_col=None, name_col=None):
    """Ưu tiên: cột chỉ định → cột tên đã biết → cuối cùng mới auto-dò.

    Trả (code_col, name_col, known); known=True nếu lấy từ tên cột đã biết/chỉ định
    (khi đó KHÔNG siết _CODE_RE để không loại nhầm mã hợp lệ như 'A17†').
    """
    lower = {str(c).lower(): c for c in df.columns}
    if code_col is None:
        for k in _KNOWN_CODE_COLS:
            if k in lower:
                code_col = lower[k]
                break
    if name_col is None:
        for k in _KNOWN_NAME_COLS:
            if k in lower:
                name_col = lower[k]
                break
    known = code_col is not None and name_col is not None
    if not known:
        auto_c, auto_n = _detect_columns(df)
        code_col = code_col or auto_c
        name_col = name_col or auto_n
    return code_col, name_col, known


def _detect_columns(df):
    """Tự dò cột mã ICD và cột tên tiếng Việt.

    Trả (code_col, name_col). Heuristic: cột 'mã' = tỉ lệ ô khớp ^[A-Z]\\d\\d cao;
    cột 'tên' = cột text dài nhất còn lại.
    """
    code_col, best_code_ratio = None, 0.0
    for col in df.columns:
        vals = df[col].astype(str).str.strip()
        ratio = vals.apply(lambda v: bool(_CODE_RE.match(v))).mean()
        if ratio > best_code_ratio:
            code_col, best_code_ratio = col, ratio
    name_col, best_len = None, -1.0
    for col in df.columns:
        if col == code_col:
            continue
        avg_len = df[col].astype(str).str.len().mean()
        if avg_len > best_len:
            name_col, best_len = col, avg_len
    return code_col, name_col


def load_icd_rows(icd10_path, code_col=None, name_col=None, require_active=False,
                  dedupe=True):
    """Đọc Excel ICD-10 → (rows, code_col, name_col). Tách riêng để test không cần model.

    - Tự nhận schema đã làm sạch (ma_benh/ten_benh_vi); nếu không có thì auto-dò cột.
    - Chuẩn hóa mã: strip †/* ('A17.0†' -> 'A17.0') TRƯỚC khi kiểm tra/dedupe.
    - dedupe=True: gộp mã trùng sau chuẩn hóa, giữ dòng đầu (→ danh sách mã duy nhất).
    - require_active: nếu có cột 'hieu_luc' thì chỉ giữ dòng còn hiệu lực.
    """
    import pandas as pd

    # đọc mọi sheet, gộp lại (bản QĐ 4469 có thể nhiều sheet)
    sheets = pd.read_excel(icd10_path, sheet_name=None, dtype=str)
    frames = [s for s in sheets.values() if s is not None and not s.empty]
    df = pd.concat(frames, ignore_index=True) if frames else list(sheets.values())[0]

    code_col, name_col, known = _resolve_columns(df, code_col, name_col)
    if code_col is None or name_col is None:
        raise ValueError(f"Không xác định được cột mã/tên trong {icd10_path}")

    lower = {str(c).lower(): c for c in df.columns}
    active_col = lower.get("hieu_luc") if require_active else None
    active_ok = {"có", "co", "1", "true", "x", "yes"}

    rows = []
    seen = set()
    for _, r in df.iterrows():
        code = _normalize_code(r[code_col])
        name = str(r[name_col]).strip()
        if not name or name.lower() == "nan":
            continue
        if not code or code.lower() == "nan":
            continue
        if active_col is not None and str(r[active_col]).strip().lower() not in active_ok:
            continue
        if not known and not _CODE_RE.match(code):
            continue
        if dedupe:
            if code in seen:
                continue
            seen.add(code)
        rows.append({"code": code, "name": name})
    return rows, code_col, name_col


def build_icd_index(icd10_path, batch_size=128, code_col=None, name_col=None,
                    require_active=False, dedupe=True):
    """Load Excel, encode tên bệnh bằng CẢ 2 model, trả về IcdIndex.

    icd10_path: đường dẫn .xlsx (nhận qua tham số — KHÔNG hardcode).
    code_col/name_col: ép tên cột nếu muốn; mặc định tự nhận (ma_benh/ten_benh_vi).
    """
    rows, code_col, name_col = load_icd_rows(
        icd10_path, code_col=code_col, name_col=name_col,
        require_active=require_active, dedupe=dedupe)
    if not rows:
        raise ValueError(f"Không đọc được dòng ICD nào từ {icd10_path}")

    names = [r["name"] for r in rows]

    # ViHealthBERT: word-seg trước, normalize để dùng dot = cosine
    vh = _get_vihealthbert()
    vh_inputs = [_word_seg(n) for n in names]
    vh_emb = vh.encode(vh_inputs, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)

    # e5: prefix "passage: "
    e5 = _get_e5()
    e5_inputs = [f"passage: {n}" for n in names]
    e5_emb = e5.encode(e5_inputs, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)

    return IcdIndex(rows, vh_emb.astype(np.float32), e5_emb.astype(np.float32))


class IcdIndex:
    """Giữ danh sách ICD + 2 ma trận embedding, hỗ trợ search top-k mỗi model."""

    def __init__(self, rows, vh_emb, e5_emb):
        self.rows = rows
        self.vh_emb = vh_emb
        self.e5_emb = e5_emb

    def _topk(self, q_vec, emb, k):
        scores = emb @ q_vec  # cosine (đã normalize)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.rows[i]["code"], self.rows[i]["name"], float(scores[i])) for i in idx]

    def search(self, query_text: str, top_k: int = 5):
        """Trả {'vihealthbert': [(code,name,score)...], 'e5': [...]}."""
        vh = _get_vihealthbert()
        e5 = _get_e5()
        qv_vh = vh.encode([_word_seg(query_text)], convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)[0]
        qv_e5 = e5.encode([f"query: {query_text}"], convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)[0]
        return {
            "vihealthbert": self._topk(qv_vh.astype(np.float32), self.vh_emb, top_k),
            "e5": self._topk(qv_e5.astype(np.float32), self.e5_emb, top_k),
        }


# --------------------------------------------------------------------------- #
# RxNorm REST API cho THUỐC.
# --------------------------------------------------------------------------- #
_RXNAV = "https://rxnav.nlm.nih.gov/REST"


def _rx_get(path: str, params: dict, timeout: int = 20):
    import requests
    r = requests.get(f"{_RXNAV}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def rxnorm_candidates(drug_name: str, max_results: int = 8):
    """Trả list (rxcui, name, score) từ RxNorm cho tên thuốc.

    RxNorm chỉ nhận tên hoạt chất/biệt dược tiếng Anh. Chiến lược:
      1) approximateTerm cho cả cụm.
      2) nếu rỗng, thử tách token hoạt chất chính (từ dài nhất / bỏ dạng bào chế).
    """
    if not drug_name or set(drug_name.strip()) <= {"*"}:
        return []  # tên bị che bằng dấu * → không tra được

    def _approx(term):
        try:
            data = _rx_get("approximateTerm.json",
                           {"term": term, "maxEntries": max_results})
        except Exception:  # noqa: BLE001
            return []
        cands = (data.get("approximateGroup", {}) or {}).get("candidate", []) or []
        out, seen = [], set()
        for c in cands:
            rxcui = c.get("rxcui")
            if not rxcui or rxcui in seen:
                continue
            seen.add(rxcui)
            out.append((rxcui, c.get("name") or _rx_name(rxcui),
                        float(c.get("score", 0)) / 100.0))
        return out

    res = _approx(drug_name.strip())
    if res:
        return res[:max_results]

    # fallback: thử từng token "giống hoạt chất" (dài, chữ cái)
    for tok in sorted(re.findall(r"[A-Za-zÀ-ỹ]{4,}", drug_name), key=len, reverse=True):
        res = _approx(tok)
        if res:
            return res[:max_results]
    return []


def _rx_name(rxcui: str) -> str:
    try:
        data = _rx_get(f"rxcui/{rxcui}/property.json", {"propName": "RxNorm Name"})
        props = (data.get("propConceptGroup", {}) or {}).get("propConcept", []) or []
        return props[0]["propValue"] if props else rxcui
    except Exception:  # noqa: BLE001
        return rxcui
