"""prompts/ner_assertion_prompt.py

Prompt template dùng LÚC CHẠY PIPELINE CHÍNH (bước [2] NER + TYPE + ASSERTION).
Gộp 3 việc trong 1 lần gọi Qwen3-8B, trả về JSON.

Nguyên tắc quan trọng:
- Ép model liệt kê MỌI lần xuất hiện của entity, KỂ CẢ lặp lại (KHÔNG dedupe).
- Chỉ gán assertion cho TRIỆU_CHỨNG / CHẨN_ĐOÁN / THUỐC.
- KHÔNG sinh candidates ở bước này (mapping mã ICD-10/RxNorm là bước [4] riêng).
- KHÔNG tự bịa text không có trong câu; text phải copy y nguyên từ câu gốc.
"""

import json

TYPES = ["TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC"]
ASSERTION_TYPES = ["isNegated", "isFamily", "isHistorical"]
TYPES_WITH_ASSERTION = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}

SYSTEM_PROMPT = """\
Bạn là chuyên gia trích xuất thông tin y khoa từ văn bản tiếng Việt.
Nhiệm vụ: đọc MỘT đoạn văn bản y tế và trích xuất tất cả thực thể y khoa.

5 LOẠI THỰC THỂ (type):
- TRIỆU_CHỨNG: dấu hiệu/triệu chứng bệnh nhân cảm nhận hoặc quan sát được (sốt, ho, đau bụng, vàng da, khó thở...).
- TÊN_XÉT_NGHIỆM: tên xét nghiệm/thủ thuật chẩn đoán (công thức máu, X-quang, siêu âm, định lượng G6PD...).
- KẾT_QUẢ_XÉT_NGHIỆM: giá trị/kết quả cụ thể của xét nghiệm (bạch cầu 12 G/L, HbA1c 7%, dương tính, âm tính...).
- CHẨN_ĐOÁN: tên bệnh/chẩn đoán (viêm phổi, đái tháo đường, thiếu men G6PD...).
- THUỐC: tên thuốc/hoạt chất (paracetamol, amoxicillin, insulin...).

ASSERTION (chỉ áp dụng cho TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC — tối đa 3, có thể rỗng):
- isNegated: thực thể bị PHỦ ĐỊNH (không có / chưa từng / phủ nhận / âm tính).
- isFamily: thực thể thuộc về NGƯỜI NHÀ (bố, mẹ, anh chị em...), KHÔNG phải bản thân bệnh nhân.
- isHistorical: thực thể thuộc QUÁ KHỨ / tiền sử (đã từng, trước đây, tiền sử...).
TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM KHÔNG BAO GIỜ có assertion (luôn là []).

QUY TẮC TRÍCH XUẤT (RẤT QUAN TRỌNG — ƯU TIÊN CHÍNH XÁC HƠN SỐ LƯỢNG, TRÍCH ÍT MÀ ĐÚNG):
1. CHỈ trích khái niệm y khoa CHUẨN, NGẮN GỌN (tên triệu chứng/bệnh/thuốc/xét nghiệm/kết quả cụ thể).
   Thà BỎ SÓT còn hơn trích thừa. Nếu phân vân một cụm có phải thực thể không → KHÔNG trích.
2. TUYỆT ĐỐI KHÔNG trích (đây KHÔNG phải thực thể y khoa):
   - Câu tư vấn/khuyến nghị/câu hỏi/kiến thức giáo dục chung ("nên đi khám", "cần lưu ý", "bệnh này lây qua...").
   - Hành vi/sinh hoạt: "cho con bú", "chuyển về sống cùng gia đình", "ăn uống điều độ", "chơi với chó".
   - Cụm mơ hồ/không đặc hiệu: "bất thường", "không đặc hiệu", "khó khăn khi ra khỏi giường".
   - Mệnh đề mô tả dài: chỉ lấy thuật ngữ lõi, KHÔNG lấy cả câu ("đau bụng" chứ không phải "đau bụng dữ dội quanh rốn nhiều ngày kèm nôn").
   - Thời gian/số liệu đứng một mình không gắn với xét nghiệm.
   - Ký hiệu KÊ ĐƠN (liều/đường dùng/tần suất) — KHÔNG phải TÊN_XÉT_NGHIỆM/KẾT_QUẢ:
     "25mg", "500mg", "po", "iv", "im", "sc", "bid", "tid", "qd", "viên", "ml", "x2", "uống 2 lần".
     Với thuốc kèm liều ("metoprolol 25mg po bid") → CHỈ lấy tên thuốc ("metoprolol").
   - Cơ chế di truyền/nguyên nhân sinh học — KHÔNG phải CHẨN_ĐOÁN:
     "gen lặn", "nhiễm sắc thể", "đột biến gen", "di truyền lặn", "protein X" — trừ khi là TÊN BỆNH cụ thể.
   - Từ CHUNG CHUNG đứng một mình (không kèm tên cụ thể): "xét nghiệm", "kết quả", "thuốc", "bệnh", "điều trị".
3. Span gọn quanh THUẬT NGỮ LÕI, KHÔNG nuốt cả mệnh đề dài. Cho phép bổ ngữ ngắn gắn liền
   ("sốt cao", "đau bụng quanh rốn") nhưng CẮT phần mô tả lê thê ("...liên tục 3 ngày kèm nôn nhiều lần").
4. Văn bản kiến thức chung/FAQ (không có bệnh nhân cụ thể): CỰC KỲ dè dặt, chỉ lấy khái niệm y khoa nêu đích danh.
5. Liệt kê mọi lần xuất hiện của khái niệm THẬT (kể cả trùng lặp), nhưng KHÔNG bịa/suy diễn thêm.
6. "text" COPY Y NGUYÊN chuỗi con từ đoạn văn (giữ dấu, hoa/thường, dấu *). KHÔNG diễn giải, KHÔNG chuẩn hóa.
7. KHÔNG đoán tên thuốc bị che bằng dấu *. Nếu văn bản ghi "*******", giữ nguyên chuỗi dấu *.
8. Không có ngữ cảnh bệnh nhân rõ ràng → assertions để [] là hợp lệ.
9. Chỉ trả về JSON hợp lệ theo đúng định dạng, KHÔNG kèm giải thích, KHÔNG markdown.

ĐỊNH DẠNG OUTPUT (JSON):
{"entities": [{"text": "<chuỗi con>", "type": "<1 trong 5 type>", "assertions": ["isNegated", ...]}]}
Nếu không có thực thể nào: {"entities": []}
"""


def _format_triggers(triggers: dict) -> str:
    """Rút gọn bảng trigger words thành text ngắn cho prompt."""
    if not triggers:
        return ""
    lines = ["THAM KHẢO TỪ KHÓA GỢI Ý ASSERTION (chỉ là gợi ý, ngữ cảnh quyết định cuối cùng):"]
    for key in ASSERTION_TYPES:
        block = triggers.get(key, {})
        words = block.get("triggers", []) if isinstance(block, dict) else block
        if words:
            sample = ", ".join(words[:12])
            lines.append(f"- {key}: {sample}")
    return "\n".join(lines)


def _format_few_shot(examples: list) -> str:
    """examples: list các dict {"text": ..., "entities": [...]}."""
    if not examples:
        return ""
    blocks = ["VÍ DỤ MẪU:"]
    for ex in examples:
        inp = ex.get("text", "")
        out = {"entities": ex.get("entities", [])}
        blocks.append(
            f"Input:\n{inp}\nOutput:\n{json.dumps(out, ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


def _format_negatives(negatives: list) -> str:
    """negatives: list dict {"text": ..., "note": ..., "entities": [...]}
    Các case có từ phủ định NHƯNG không nên gán isNegated (chống false trigger)."""
    if not negatives:
        return ""
    blocks = ["VÍ DỤ KHÓ — CÓ TỪ PHỦ ĐỊNH NHƯNG KHÔNG ÁP DỤNG ASSERTION SAI:"]
    for ex in negatives:
        inp = ex.get("text", "")
        note = ex.get("note", "")
        out = {"entities": ex.get("entities", [])}
        line = f"Input:\n{inp}\n"
        if note:
            line += f"(Lưu ý: {note})\n"
        line += f"Output:\n{json.dumps(out, ensure_ascii=False)}"
        blocks.append(line)
    return "\n\n".join(blocks)


def build_messages(chunk_text, few_shot_examples=None, triggers=None, negative_examples=None):
    """Trả về list messages [{'role','content'}] cho llm_client.chat().

    chunk_text: 1 câu/đoạn đã tách ở bước [1].
    """
    parts = []
    fs = _format_few_shot(few_shot_examples or [])
    tg = _format_triggers(triggers or {})
    ng = _format_negatives(negative_examples or [])
    if fs:
        parts.append(fs)
    if tg:
        parts.append(tg)
    if ng:
        parts.append(ng)
    parts.append(
        "Bây giờ trích xuất thực thể từ đoạn sau. TRÍCH ÍT MÀ ĐÚNG — chỉ khái niệm y khoa "
        "rõ ràng, ngắn gọn; BỎ QUA câu chung/tư vấn/mô tả dài/cụm mơ hồ. Chỉ trả về JSON.\n"
        f"Input:\n{chunk_text}\nOutput:"
    )
    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


VERIFY_SYSTEM = """\
Bạn là chuyên gia y khoa rà soát kết quả trích xuất thực thể.
Cho MỘT câu và danh sách cụm đã đánh số được rút ra từ câu đó. Nhiệm vụ: giữ lại
CHỈ những cụm là thực thể y khoa THẬT SỰ, cụ thể (triệu chứng/xét nghiệm/kết quả
xét nghiệm/chẩn đoán/thuốc). LOẠI những cụm:
- Từ chung chung, không đặc hiệu, mô tả dài dòng, hành vi/sinh hoạt.
- Cơ chế/nguyên nhân sinh học, câu tư vấn/kiến thức chung.
- Không mang nghĩa lâm sàng cụ thể.
Chỉ trả về JSON: {"keep": [danh sách CHỈ SỐ (số nguyên) cần GIỮ]}. Không giải thích."""


def build_verify_messages(chunk_text, entities):
    """Bước [2b] tự phản biện: hỏi model cụm nào là thực thể THẬT để giữ.

    entities: list dict có 'text','type'. Trả messages cho chat_json.
    Model trả {"keep": [idx...]} — chỉ số (0-based) các entity cần giữ.
    """
    listing = "\n".join(
        f"{i}. [{e.get('type')}] {e.get('text')}" for i, e in enumerate(entities)
    )
    user = (
        f"Câu:\n{chunk_text}\n\n"
        f"Các cụm đã trích (giữ cụm y khoa thật, loại cụm mơ hồ/chung/mô tả):\n{listing}\n\n"
        'Trả về JSON {"keep": [chỉ số cần giữ]}.'
    )
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": user},
    ]
