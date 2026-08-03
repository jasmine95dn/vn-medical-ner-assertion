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

QUY TẮC BẮT BUỘC:
1. Liệt kê MỌI lần xuất hiện, KỂ CẢ trùng lặp. Nếu "vàng da" xuất hiện 3 lần → xuất ra 3 entity riêng.
2. "text" phải là chuỗi con COPY Y NGUYÊN từ đoạn văn (giữ nguyên dấu, hoa/thường, dấu *). KHÔNG diễn giải, KHÔNG chuẩn hóa.
3. KHÔNG đoán tên thuốc bị che bằng dấu *. Nếu văn bản ghi "*******", giữ nguyên chuỗi dấu *.
4. Nếu không có ngữ cảnh bệnh nhân rõ ràng (văn bản kiến thức chung/FAQ), assertions để [] là hợp lệ.
5. Chỉ trả về JSON hợp lệ theo đúng định dạng, KHÔNG kèm giải thích, KHÔNG markdown.

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
        "Bây giờ trích xuất thực thể từ đoạn sau. Chỉ trả về JSON.\n"
        f"Input:\n{chunk_text}\nOutput:"
    )
    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
