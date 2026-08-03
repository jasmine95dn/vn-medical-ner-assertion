"""llm_client.py

Wrapper gọi Qwen3-8B qua Ollama local API. Dùng chung cho:
  - bước [2] NER + TYPE + ASSERTION
  - bước [4b] RE-RANK candidate

Đặc điểm cần xử lý:
- Qwen3 có "thinking mode" → có thể chèn khối <think>...</think> ở đầu output.
  Ta TẮT thinking (nhanh hơn, ổn định hơn cho structured extraction) và vẫn strip
  phòng trường hợp model vẫn sinh ra.
- Bật format=json của Ollama để ép output là JSON hợp lệ.
- Parse JSON chịu lỗi: cắt ```json fences, tìm object {...} đầu tiên nếu cần.
"""

import json
import os
import re
import time

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Transport: ưu tiên thư viện `ollama`, fallback sang REST qua `requests`.
# --------------------------------------------------------------------------- #
def _chat_via_ollama(messages, model, options, fmt):
    import ollama

    client = ollama.Client(host=DEFAULT_HOST)
    kwargs = dict(model=model, messages=messages, options=options)
    if fmt:
        kwargs["format"] = fmt
    # think=False: tắt reasoning trace (được hỗ trợ ở bản ollama mới; bỏ qua nếu lỗi)
    try:
        resp = client.chat(think=False, **kwargs)
    except TypeError:
        resp = client.chat(**kwargs)
    return resp["message"]["content"]


def _chat_via_requests(messages, model, options, fmt):
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "think": False,
    }
    if fmt:
        payload["format"] = fmt
    r = requests.post(f"{DEFAULT_HOST}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chat(messages, model=DEFAULT_MODEL, temperature=0.0, num_ctx=8192,
         fmt=None, max_retries=3):
    """Gọi model 1 lần, trả về text thô (đã strip <think>).

    fmt="json" để bật JSON mode của Ollama.
    """
    options = {"temperature": temperature, "num_ctx": num_ctx}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                content = _chat_via_ollama(messages, model, options, fmt)
            except ImportError:
                content = _chat_via_requests(messages, model, options, fmt)
            return _THINK_RE.sub("", content).strip()
        except Exception as e:  # noqa: BLE001 — muốn retry mọi lỗi transport
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Ollama chat thất bại sau {max_retries} lần: {last_err}")


def _extract_json(text: str):
    """Parse JSON chịu lỗi từ output model."""
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fallback: tìm object {...} cân bằng ngoặc đầu tiên.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def chat_json(messages, model=DEFAULT_MODEL, temperature=0.0, num_ctx=8192,
              max_retries=3, default=None):
    """Gọi model với JSON mode, trả về dict/list đã parse.

    Nếu parse thất bại sau các lần thử → trả về `default` (mặc định {}).
    """
    if default is None:
        default = {}
    text = chat(messages, model=model, temperature=temperature,
                num_ctx=num_ctx, fmt="json", max_retries=max_retries)
    parsed = _extract_json(text)
    return parsed if parsed is not None else default


def health_check(model=DEFAULT_MODEL):
    """Kiểm tra Ollama sống và model trả lời được. Trả về True/False."""
    try:
        out = chat([{"role": "user", "content": "Trả lời đúng 1 từ: OK"}],
                   model=model, max_retries=1)
        return bool(out)
    except Exception:  # noqa: BLE001
        return False
