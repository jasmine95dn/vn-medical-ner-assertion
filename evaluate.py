"""evaluate.py — chấm điểm pipeline trên validation_set.json

Hai chế độ:
  --run           chạy pipeline in-process trên từng câu validation (cần Ollama
                  Qwen3-8B; mặc định BỎ candidate cho nhanh, bật bằng --candidates)
  --pred FILE     nạp prediction có sẵn (JSON cùng cấu trúc validation_set:
                  list các {"text": ..., "entities": [...]}), khớp theo thứ tự

Chỉ số:
  - NER span+type : P / R / F1 (TP = trùng span VÀ trùng type)
  - span-only     : bỏ qua type (đo ranh giới)
  - type acc      : độ chính xác type trên các cặp trùng span
  - assertion     : exact-set-match acc + micro P/R/F1 theo từng nhãn
  - candidates    : hit@k (chỉ tính khi gold có candidates; validation seed để [])

Cách khớp: mỗi câu, greedy match pred↔gold theo (mode span). exact = trùng
[start,end]; overlap = cùng type-agnostic, chọn cặp IoU cao nhất > 0.
"""

import argparse
import json
import sys

ASSERTION_LABELS = ["isNegated", "isFamily", "isHistorical"]
TYPES_WITH_ASSERTION = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
TYPES_WITH_CANDIDATES = {"CHẨN_ĐOÁN", "THUỐC"}


# --------------------------------------------------------------------------- #
# Khớp entity
# --------------------------------------------------------------------------- #
def _iou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_entities(gold, pred, mode="exact"):
    """Trả về (matches, unmatched_gold_idx, unmatched_pred_idx).

    matches: list (gi, pi). Mỗi gold/pred dùng tối đa 1 lần (bipartite greedy).
    Khớp span (không xét type ở bước này để còn đo type acc riêng).
    """
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            gs, ge = g["position"]
            ps, pe = p["position"]
            if mode == "exact":
                if gs == ps and ge == pe:
                    pairs.append((1.0, gi, pi))
            else:  # overlap
                iou = _iou((gs, ge), (ps, pe))
                if iou > 0:
                    pairs.append((iou, gi, pi))
    pairs.sort(reverse=True)  # ưu tiên IoU cao / exact
    used_g, used_p, matches = set(), set(), []
    for _score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))
    ug = [i for i in range(len(gold)) if i not in used_g]
    up = [i for i in range(len(pred)) if i not in used_p]
    return matches, ug, up


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# --------------------------------------------------------------------------- #
# Chấm điểm
# --------------------------------------------------------------------------- #
def evaluate(gold_docs, pred_docs, mode="exact"):
    assert len(gold_docs) == len(pred_docs), \
        f"số document lệch: gold={len(gold_docs)} pred={len(pred_docs)}"

    # span+type
    tp = fp = fn = 0
    # span-only
    stp = sfp = sfn = 0
    # type accuracy trên span-match
    type_correct = type_total = 0
    # assertion
    assert_exact_ok = assert_total = 0
    a_tp = {l: 0 for l in ASSERTION_LABELS}
    a_fp = {l: 0 for l in ASSERTION_LABELS}
    a_fn = {l: 0 for l in ASSERTION_LABELS}
    # candidates
    cand_hit = cand_total = 0

    errors = []  # ghi lại vài lỗi để in ra

    for di, (gdoc, pdoc) in enumerate(zip(gold_docs, pred_docs)):
        gold = gdoc["entities"]
        pred = pdoc["entities"]
        if gdoc.get("text") and pdoc.get("text") and gdoc["text"] != pdoc["text"]:
            print(f"[warn] doc {di}: text gold/pred khác nhau (khớp theo thứ tự).",
                  file=sys.stderr)

        matches, ug, up = match_entities(gold, pred, mode=mode)

        # span-only
        stp += len(matches)
        sfn += len(ug)
        sfp += len(up)

        matched_g = {gi for gi, _ in matches}
        matched_p = {pi for _, pi in matches}

        # span+type: chỉ TP khi type cũng trùng
        for gi, pi in matches:
            g, p = gold[gi], pred[pi]
            type_total += 1
            if g["type"] == p["type"]:
                type_correct += 1
                tp += 1
            else:
                fp += 1  # span đúng, type sai → tính là FP (và FN cho gold)
                fn += 1
                errors.append((di, "TYPE", g, p))

            # assertion (chỉ khi type gold thuộc nhóm có assertion & type khớp)
            if g["type"] == p["type"] and g["type"] in TYPES_WITH_ASSERTION:
                gset = set(g.get("assertions", []))
                pset = set(p.get("assertions", []))
                assert_total += 1
                if gset == pset:
                    assert_exact_ok += 1
                elif errors is not None:
                    errors.append((di, "ASSERT", g, p))
                for l in ASSERTION_LABELS:
                    if l in gset and l in pset:
                        a_tp[l] += 1
                    elif l in pset and l not in gset:
                        a_fp[l] += 1
                    elif l in gset and l not in pset:
                        a_fn[l] += 1

            # candidates (chỉ khi gold có candidate)
            if g["type"] == p["type"] and g["type"] in TYPES_WITH_CANDIDATES:
                gcand = g.get("candidates", []) or []
                if gcand:
                    cand_total += 1
                    pcand = p.get("candidates", []) or []
                    if any(c in pcand for c in gcand):
                        cand_hit += 1

        # phần span không khớp: FP/FN cho span+type
        fp += len(up)
        fn += len(ug)
        for gi in ug:
            errors.append((di, "MISS", gold[gi], None))
        for pi in up:
            errors.append((di, "SPURIOUS", None, pred[pi]))

    return {
        "ner": prf(tp, fp, fn) + (tp, fp, fn),
        "span": prf(stp, sfp, sfn) + (stp, sfp, sfn),
        "type_acc": (type_correct / type_total if type_total else 0.0, type_correct, type_total),
        "assert_exact": (assert_exact_ok / assert_total if assert_total else 0.0,
                         assert_exact_ok, assert_total),
        "assert_labels": {l: prf(a_tp[l], a_fp[l], a_fn[l]) for l in ASSERTION_LABELS},
        "candidates": (cand_hit / cand_total if cand_total else None, cand_hit, cand_total),
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Chạy pipeline in-process để sinh prediction cho từng câu validation
# --------------------------------------------------------------------------- #
def run_pipeline_on_docs(gold_docs, do_candidates, do_rerank):
    from chunker import chunk_text
    from position_recovery import recover_positions
    from merge_validate import merge_and_validate
    from main import load_prompt_assets, extract_entities_for_chunk

    assets = load_prompt_assets()
    icd_index = None
    if do_candidates:
        raise SystemExit("--candidates trong eval cần ICD index; truyền --icd10-path "
                         "và bật thủ công nếu muốn. Mặc định eval bỏ candidate.")

    preds = []
    for doc in gold_docs:
        text = doc["text"]
        ents_all = []
        for chunk in chunk_text(text):
            raw = extract_entities_for_chunk(chunk, *assets)
            ents_all.extend(recover_positions(raw, chunk))
        final = merge_and_validate(ents_all, text)
        preds.append({"text": text, "entities": final})
    return preds


# --------------------------------------------------------------------------- #
def print_report(res, mode):
    def pct(x):
        return f"{x * 100:.1f}%"

    p, r, f, tp, fp, fn = res["ner"]
    print(f"\n=== KẾT QUẢ (match={mode}) ===")
    print(f"NER (span+type)   P={pct(p)}  R={pct(r)}  F1={pct(f)}   (TP={tp} FP={fp} FN={fn})")
    sp, sr, sf, stp, sfp, sfn = res["span"]
    print(f"Span-only         P={pct(sp)}  R={pct(sr)}  F1={pct(sf)}   (TP={stp} FP={sfp} FN={sfn})")
    ta, tc, tt = res["type_acc"]
    print(f"Type accuracy     {pct(ta)}  ({tc}/{tt} span-match)")
    aa, ao, at = res["assert_exact"]
    print(f"Assertion exact   {pct(aa)}  ({ao}/{at} entity có nhóm assertion)")
    for l, (lp, lr, lf) in res["assert_labels"].items():
        print(f"   {l:14} P={pct(lp)}  R={pct(lr)}  F1={pct(lf)}")
    cv, ch, ct = res["candidates"]
    if cv is None:
        print("Candidates        (bỏ qua — gold không có mã candidate để đối chiếu)")
    else:
        print(f"Candidates hit    {pct(cv)}  ({ch}/{ct})")


def print_errors(errors, limit=20):
    if not errors:
        return
    print(f"\n=== LỖI (tối đa {limit}) ===")
    for di, kind, g, p in errors[:limit]:
        if kind == "MISS":
            print(f"  doc{di} MISS      gold: [{g['type']}] {g['text']!r} @{g['position']}")
        elif kind == "SPURIOUS":
            print(f"  doc{di} SPURIOUS  pred: [{p['type']}] {p['text']!r} @{p['position']}")
        elif kind == "TYPE":
            print(f"  doc{di} TYPE      {g['text']!r}: gold={g['type']} pred={p['type']}")
        elif kind == "ASSERT":
            print(f"  doc{di} ASSERT    {g['text']!r}: gold={g.get('assertions')} "
                  f"pred={p.get('assertions')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="data/validation_set.json")
    ap.add_argument("--pred", default=None, help="file prediction cùng cấu trúc")
    ap.add_argument("--run", action="store_true", help="chạy pipeline sinh prediction")
    ap.add_argument("--candidates", action="store_true", help="bật candidate khi --run")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--match", choices=["exact", "overlap"], default="exact")
    ap.add_argument("--save-pred", default=None, help="lưu prediction khi --run")
    ap.add_argument("--show-errors", type=int, default=20)
    args = ap.parse_args()

    with open(args.validation, encoding="utf-8") as f:
        gold_docs = json.load(f)

    if args.run:
        pred_docs = run_pipeline_on_docs(gold_docs, args.candidates, not args.no_rerank)
        if args.save_pred:
            with open(args.save_pred, "w", encoding="utf-8") as f:
                json.dump(pred_docs, f, ensure_ascii=False, indent=2)
            print(f"[eval] đã lưu prediction → {args.save_pred}")
    elif args.pred:
        with open(args.pred, encoding="utf-8") as f:
            pred_docs = json.load(f)
    else:
        ap.error("cần --run hoặc --pred FILE")

    res = evaluate(gold_docs, pred_docs, mode=args.match)
    print_report(res, args.match)
    print_errors(res["errors"], limit=args.show_errors)


if __name__ == "__main__":
    main()
