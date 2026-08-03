"""main.py  — orchestrate toàn bộ pipeline cho 100 file .txt → output/*.json

Đường dẫn data ĐỊNH NGHĨA TẬP TRUNG 1 lần ở đây, truyền xuống các module.
Trên Kaggle: chỉnh DATA_DIR trỏ tới Kaggle Dataset đã add.

Pipeline mỗi file:
  [0] load_and_clean → [1] chunk → [2] NER/type/assertion (Qwen3-8B)
  → [3] position recovery → [4a/4b] candidate mapping (CHẨN_ĐOÁN/THUỐC)
  → [5] merge + validate → ghi output/<file>.json

Cờ hữu ích khi test/tiết kiệm GPU:
  --limit N          chỉ chạy N file đầu (test nhanh 2-3 file)
  --no-candidates    bỏ bước [4] (MVP: candidates rỗng) — nhanh, chạy tối 2/8
  --no-rerank        [4a] có, nhưng bỏ re-rank LLM ở [4b]
"""

import argparse
import json
import os
import sys
import time

# --------------------------------------------------------------------------- #
# PATHS — chỉnh ở đây khi chạy Kaggle. Có thể override qua CLI/env.
# --------------------------------------------------------------------------- #
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/ten-dataset-cua-ban")
INPUT_DIR = os.environ.get("INPUT_DIR", f"{DATA_DIR}/input")
ICD10_PATH = os.environ.get("ICD10_PATH", f"{DATA_DIR}/icd10_vn.xlsx")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# data JSON đi kèm repo (few-shot, trigger, negative) — cạnh file main.py
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DATA_DIR = os.path.join(_HERE, "data")

from preprocess import load_and_clean, iter_input_files  # noqa: E402
from chunker import chunk_text  # noqa: E402
from position_recovery import recover_positions  # noqa: E402
from merge_validate import merge_and_validate, write_output  # noqa: E402
from prompts.ner_assertion_prompt import build_messages  # noqa: E402


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] không đọc được {path}: {e}", file=sys.stderr)
        return default


def load_prompt_assets():
    """Nạp few-shot / trigger / negative (thiếu file → rỗng, vẫn chạy được)."""
    few = _load_json(os.path.join(REPO_DATA_DIR, "few_shot_examples.json"), [])
    triggers = _load_json(os.path.join(REPO_DATA_DIR, "assertion_triggers.json"), {})
    negatives = _load_json(os.path.join(REPO_DATA_DIR, "negative_examples.json"), [])
    if isinstance(few, dict):
        few = few.get("examples", [])
    if isinstance(negatives, dict):
        negatives = negatives.get("examples", [])
    return few, triggers, negatives


def extract_entities_for_chunk(chunk, few, triggers, negatives):
    """Bước [2]: gọi Qwen3-8B trên 1 chunk → list entity {text,type,assertions}."""
    from llm_client import chat_json

    messages = build_messages(chunk["text"], few, triggers, negatives)
    data = chat_json(messages, default={"entities": []})
    ents = data.get("entities", []) if isinstance(data, dict) else []
    # chỉ giữ bản ghi tối thiểu hợp lệ
    clean = []
    for e in ents:
        if isinstance(e, dict) and e.get("text") and e.get("type"):
            clean.append({
                "text": e.get("text"),
                "type": e.get("type"),
                "assertions": e.get("assertions", []),
            })
    return clean


def process_file(fileinfo, icd_index, assets, do_candidates, do_rerank):
    """Chạy full pipeline cho 1 file → list entity cuối (đã validate)."""
    few, triggers, negatives = assets
    canonical = fileinfo["text"]
    chunks = chunk_text(canonical)

    all_entities = []
    for chunk in chunks:
        raw_ents = extract_entities_for_chunk(chunk, few, triggers, negatives)
        located = recover_positions(raw_ents, chunk)  # gắn position toàn cục

        if do_candidates:
            from candidate_mapping import map_candidates
            for ent in located:
                if ent["type"] in ("CHẨN_ĐOÁN", "THUỐC"):
                    ent["candidates"] = map_candidates(
                        ent, chunk["text"], icd_index,
                        use_llm_rerank=do_rerank,
                    )
        all_entities.extend(located)

    return merge_and_validate(all_entities, canonical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--icd10-path", default=ICD10_PATH)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N file đầu")
    ap.add_argument("--no-candidates", action="store_true", help="bỏ bước [4] (MVP)")
    ap.add_argument("--no-rerank", action="store_true", help="bỏ re-rank LLM [4b]")
    ap.add_argument("--skip-healthcheck", action="store_true")
    args = ap.parse_args()

    do_candidates = not args.no_candidates
    do_rerank = not args.no_rerank

    # health check Ollama
    if not args.skip_healthcheck:
        from llm_client import health_check
        print("[init] kiểm tra Ollama/Qwen3-8B ...")
        if not health_check():
            print("[error] Ollama chưa sẵn sàng. Chạy `ollama serve` + "
                  "`ollama pull qwen3:8b` trước.", file=sys.stderr)
            sys.exit(1)
        print("[init]   OK")

    # build ICD index 1 lần (nếu cần candidates)
    icd_index = None
    if do_candidates:
        from icd_rxnorm_index import build_icd_index
        print(f"[init] build ICD-10 index từ {args.icd10_path} ...")
        icd_index = build_icd_index(args.icd10_path)
        print(f"[init]   {len(icd_index.rows)} mã ICD, encoded 2 model")

    assets = load_prompt_assets()
    few, _tr, neg = assets
    print(f"[init] few-shot={len(few)}  negative={len(neg)}  "
          f"candidates={'on' if do_candidates else 'OFF'}  "
          f"rerank={'on' if do_rerank else 'OFF'}")

    os.makedirs(args.output_dir, exist_ok=True)
    files = list(iter_input_files(args.input_dir))
    if args.limit:
        files = files[:args.limit]
    print(f"[run] {len(files)} file cần xử lý\n")

    t0 = time.time()
    for i, fileinfo in enumerate(files, 1):
        name = fileinfo["filename"]
        try:
            entities = process_file(fileinfo, icd_index, assets, do_candidates, do_rerank)
        except Exception as e:  # noqa: BLE001 — 1 file lỗi không được làm hỏng cả batch
            print(f"[{i}/{len(files)}] {name}  ERROR: {e}", file=sys.stderr)
            entities = []
        out_name = os.path.splitext(name)[0] + ".json"
        write_output(entities, os.path.join(args.output_dir, out_name))
        print(f"[{i}/{len(files)}] {name} → {len(entities)} entity "
              f"({time.time() - t0:.0f}s)")

    print(f"\n[done] {len(files)} file trong {time.time() - t0:.0f}s → {args.output_dir}/")


if __name__ == "__main__":
    main()
