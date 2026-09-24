from __future__ import annotations

import os
import time
from getpass import getpass

import requests


def call_llm(messages: list[dict[str, str]]) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        try:
            print("未检测到环境变量 DEEPSEEK_API_KEY，请输入 Key（输入不回显）:", flush=True)
            api_key = getpass("").strip()
        except Exception:
            api_key = input("请输入 DEEPSEEK_API_KEY: ").strip()
    if not api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY")

    url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_err = None
    for attempt in range(1, 4):
        try:
            if attempt == 1:
                print("正在请求 LLM...", flush=True)
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < 3:
                time.sleep(0.6 * attempt)
                continue
            raise RuntimeError(f"LLM 网络请求失败，请检查网络/DNS/代理设置: {e}") from e

    raise RuntimeError(f"LLM 网络请求失败: {last_err}")


def generate_response(user_input: str, tool_result: dict) -> str:
    import json

    prompt = f"""你是一个AI助手，请根据以下信息回答用户问题：

用户输入：
{user_input}

工具返回结果：
{json.dumps(tool_result, ensure_ascii=False)}

要求：
- 用自然语言回答
- 简洁
- 中文
- 不输出JSON"""

    messages = [{"role": "user", "content": prompt}]
    return call_llm(messages)
