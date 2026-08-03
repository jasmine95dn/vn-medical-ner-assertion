"""prompts/data_generation_prompt.py

Prompt RIÊNG — chạy 1 LẦN lúc chuẩn bị data, KHÔNG dùng lúc chạy pipeline chính.
Gọi Qwen3-8B để sinh:
  - data/few_shot_examples.json   (ví dụ NER/type/assertion đa dạng)
  - data/negative_examples.json   (câu có từ phủ định nhưng KHÔNG áp dụng isNegated)

SINH XONG BẮT BUỘC TỰ REVIEW LẠI BẰNG TAY trước khi dùng — model 8B dễ sinh sai
ý đồ, nhất là negative examples (case khó). File này chỉ tạo bản nháp.

Cách chạy (sau khi đã `ollama serve` + `ollama pull qwen3:8b`):
    python -m prompts.data_generation_prompt --out-dir data
"""

import argparse
import json
import os

FEW_SHOT_GEN_PROMPT = """\
Bạn là chuyên gia gán nhãn dữ liệu y khoa tiếng Việt.
Hãy tạo {n} ví dụ mẫu (few-shot) đa dạng cho bài toán trích xuất thực thể y khoa.

Mỗi ví dụ gồm 1 câu/đoạn văn y tế tiếng Việt TỰ NHIÊN và danh sách thực thể.
Bao phủ đủ 5 type: TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC.
Bao phủ đủ 3 assertion: isNegated, isFamily, isHistorical (và cả trường hợp assertions rỗng).
Gồm cả văn phong bệnh án ngắn LẪN văn phong FAQ/bài viết dài.
Nên có ít nhất 1 ví dụ chứa viết tắt y khoa và 1 ví dụ có entity lặp lại nhiều lần.

Quy tắc:
- "text" phải là chuỗi con xuất hiện Y NGUYÊN trong câu.
- assertions chỉ gán cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC; TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM luôn [].
- Liệt kê mọi lần xuất hiện, kể cả trùng lặp.

Chỉ trả về JSON, KHÔNG giải thích:
{{"examples": [{{"text": "...", "entities": [{{"text": "...", "type": "...", "assertions": [...]}}]}}]}}
"""

NEGATIVE_GEN_PROMPT = """\
Bạn là chuyên gia gán nhãn dữ liệu y khoa tiếng Việt.
Hãy tạo {n} ví dụ KHÓ về assertion cho bài toán trích xuất thực thể y khoa.

Mỗi ví dụ là câu CÓ CHỨA TỪ PHỦ ĐỊNH/tiền sử/người nhà NHƯNG thực thể KHÔNG nên bị
gán assertion tương ứng (bẫy false trigger). Ví dụ:
- "Không chỉ sốt mà còn ho" — "sốt" và "ho" KHÔNG bị phủ định (chữ "không" ở đây không phủ định triệu chứng).
- "Bệnh nhân lo lắng bố từng bị ung thư nhưng bản thân khỏe" — "ung thư" là isFamily+isHistorical, còn tình trạng bản thân thì không.
- "Thuốc này không gây buồn ngủ" — "buồn ngủ" bị phủ định (isNegated) đúng, KHÔNG phải bẫy; đừng tạo kiểu này.

Với mỗi ví dụ, thêm "note" giải thích ngắn tại sao KHÔNG áp dụng assertion sai.

Chỉ trả về JSON, KHÔNG giải thích:
{{"examples": [{{"text": "...", "note": "...", "entities": [{{"text": "...", "type": "...", "assertions": [...]}}]}}]}}
"""


def build_few_shot_prompt(n: int = 12) -> str:
    return FEW_SHOT_GEN_PROMPT.format(n=n)


def build_negative_prompt(n: int = 10) -> str:
    return NEGATIVE_GEN_PROMPT.format(n=n)


def _generate(prompt: str):
    """Gọi Qwen3-8B qua llm_client, parse JSON, trả về list examples."""
    from llm_client import chat_json

    data = chat_json([{"role": "user", "content": prompt}])
    return data.get("examples", []) if isinstance(data, dict) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n-few-shot", type=int, default=12)
    ap.add_argument("--n-negative", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[data-gen] Sinh few_shot_examples.json ...")
    few = _generate(build_few_shot_prompt(args.n_few_shot))
    with open(os.path.join(args.out_dir, "few_shot_examples.json"), "w", encoding="utf-8") as f:
        json.dump(few, f, ensure_ascii=False, indent=2)
    print(f"[data-gen]   -> {len(few)} ví dụ few-shot")

    print("[data-gen] Sinh negative_examples.json ...")
    neg = _generate(build_negative_prompt(args.n_negative))
    with open(os.path.join(args.out_dir, "negative_examples.json"), "w", encoding="utf-8") as f:
        json.dump(neg, f, ensure_ascii=False, indent=2)
    print(f"[data-gen]   -> {len(neg)} ví dụ negative")

    print("\n[!] BẮT BUỘC review lại 2 file bằng tay trước khi dùng cho pipeline chính.")


if __name__ == "__main__":
    main()
