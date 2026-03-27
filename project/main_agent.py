from __future__ import annotations

import json
import re

from agent.agent import build_messages
from agent.llm import call_llm, generate_response
from yolo_tool import detect_object


def _parse_tool_json(text: str) -> dict:
    s = (text or "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", s)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_tool_call(user_input: str) -> dict:
    text = (user_input or "").strip().lower()
    mapping = [
        ("人", "person"),
        ("行人", "person"),
        ("汽车", "car"),
        ("车", "car"),
        ("公交", "bus"),
        ("巴士", "bus"),
        ("卡车", "truck"),
        ("自行车", "bicycle"),
        ("摩托", "motorcycle"),
        ("摩托车", "motorcycle"),
        ("狗", "dog"),
        ("猫", "cat"),
        ("瓶", "bottle"),
        ("杯", "cup"),
        ("手机", "cell phone"),
        ("笔记本", "laptop"),
        ("电脑", "laptop"),
    ]
    for k, v in mapping:
        if k in text:
            return {"tool": "detect_object", "target": v}

    m = re.search(r"\b([a-z][a-z ]{1,30})\b", text)
    if m:
        return {"tool": "detect_object", "target": m.group(1).strip()}

    return {"tool": "detect_object", "target": "person"}


def main() -> None:
    user_input = input("请输入指令：").strip()
    if not user_input:
        print(json.dumps({"error": "empty_input"}, ensure_ascii=False))
        return

    print(f"[User] {user_input}", flush=True)
    messages = build_messages(user_input)
    try:
        llm_output = call_llm(messages)
        print(f"[LLM Output Raw] {llm_output}", flush=True)
        tool_call = _parse_tool_json(llm_output)
    except Exception as e:
        print(f"[LLM Error] {e}", flush=True)
        tool_call = _fallback_tool_call(user_input)
        tool_call["_llm_error"] = str(e)

    tool_name = str(tool_call.get("tool", "")).strip()
    target = str(tool_call.get("target", "")).strip()

    if not tool_name:
        print(json.dumps({"error": "missing_tool", "parsed": tool_call}, ensure_ascii=False))
        return

    if tool_name not in ("detect_object", "chat"):
        print(json.dumps({"error": "unknown_tool", "parsed": tool_call}, ensure_ascii=False))
        return

    if tool_name == "chat":
        resp = str(tool_call.get("response", "")).strip() or "OK"
        print(resp, flush=True)
        return

    if not target:
        print(json.dumps({"error": "invalid_tool_call", "parsed": tool_call}, ensure_ascii=False))
        return

    result = detect_object(target)
    
    try:
        final_response = generate_response(user_input, result)
        print(final_response, flush=True)
    except Exception as e:
        print(f"[Agent Error] {e}", flush=True)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
