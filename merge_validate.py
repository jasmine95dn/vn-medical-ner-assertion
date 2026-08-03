"""merge_validate.py  — bước [5] MERGE + VALIDATE

Ghép kết quả các chunk thành output cuối cho 1 file, ép đúng schema đề bài, và
KIỂM TRA text khớp position trên canonical text (== file gốc).

Schema field theo type:
  TRIỆU_CHỨNG        : text, position, type, assertions
  CHẨN_ĐOÁN          : text, position, type, assertions, candidates
  THUỐC              : text, position, type, assertions, candidates
  TÊN_XÉT_NGHIỆM     : text, position, type
  KẾT_QUẢ_XÉT_NGHIỆM : text, position, type
"""

import json
import os

VALID_TYPES = {
    "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC",
}
TYPES_WITH_ASSERTION = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
TYPES_WITH_CANDIDATES = {"CHẨN_ĐOÁN", "THUỐC"}
VALID_ASSERTIONS = ["isNegated", "isFamily", "isHistorical"]  # giữ thứ tự chuẩn


def _clean_assertions(raw):
    """Lọc assertion hợp lệ, dedupe, tối đa 3, theo thứ tự chuẩn."""
    if not isinstance(raw, list):
        return []
    present = set()
    for a in raw:
        if isinstance(a, str) and a in VALID_ASSERTIONS:
            present.add(a)
    return [a for a in VALID_ASSERTIONS if a in present][:3]


def _clean_candidates(raw):
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for c in raw:
        s = str(c).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_entity(ent, canonical_text):
    """Chuẩn hóa 1 entity về đúng schema. Trả dict hợp lệ hoặc None nếu loại bỏ.

    Bỏ khi: type sai, thiếu position, hoặc text KHÔNG khớp canonical_text[pos].
    """
    etype = ent.get("type")
    if etype not in VALID_TYPES:
        return None

    pos = ent.get("position")
    if (not isinstance(pos, (list, tuple)) or len(pos) != 2
            or not all(isinstance(x, int) for x in pos)):
        return None
    s, e = pos
    if not (0 <= s < e <= len(canonical_text)):
        return None

    # text chuẩn = đúng lát cắt trên file gốc (đảm bảo khớp position tuyệt đối)
    sliced = canonical_text[s:e]
    text = ent.get("text", "")
    if text != sliced:
        # nếu lệch, ưu tiên bám position: dùng lát cắt làm text chính thức
        text = sliced

    out = {"text": text, "position": [s, e], "type": etype}

    if etype in TYPES_WITH_ASSERTION:
        out["assertions"] = _clean_assertions(ent.get("assertions"))
    if etype in TYPES_WITH_CANDIDATES:
        out["candidates"] = _clean_candidates(ent.get("candidates"))
    return out


def merge_and_validate(entities, canonical_text):
    """Ghép + validate list entity của cả file. Trả list dict sạch, sort theo vị trí.

    KHÔNG dedupe entity trùng (mỗi lần xuất hiện là 1 entity riêng — chỉ loại các
    bản ghi lỗi schema/position).
    """
    cleaned = []
    for ent in entities:
        built = build_entity(ent, canonical_text)
        if built is not None:
            cleaned.append(built)
    cleaned.sort(key=lambda x: (x["position"][0], x["position"][1]))
    return cleaned


def write_output(entities, out_path):
    """Ghi list entity ra JSON UTF-8 (ensure_ascii=False)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
