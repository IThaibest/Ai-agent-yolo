from __future__ import annotations


def build_prompt(user_input: str) -> list[dict[str, str]]:
    system = (
        "你是一个工具选择助手，只能输出 JSON 对象，不允许输出任何其它字符、解释、标点或代码块，"
        "不能包含反引号和多余文本。严格遵守：仅输出一行 JSON。"
        "支持的工具：\n"
        '- detect_object：调用视觉检测函数，字段为 {"tool":"detect_object","target":"<英文小写类别>"}\n'
        '- chat：与用户闲聊，字段为 {"tool":"chat","response":"<文本>"}\n'
        "目标映射（中文→英文）：人→person，杯子→cup，手机→phone，瓶子→bottle。"
        "若用户提出找物体，输出 detect_object 并将中文目标映射为英文小写；"
        "若非找物体（如问候），输出 chat 并复述合适的简短回复。"
        "JSON 必须是单个对象，不要包含注释或多余字段。"
    )

    # Few-shot examples
    examples: list[dict[str, str]] = [
        {"role": "user", "content": "帮我找人"},
        {"role": "assistant", "content": '{"tool": "detect_object", "target": "person"}'},
        {"role": "user", "content": "帮我找杯子"},
        {"role": "assistant", "content": '{"tool": "detect_object", "target": "cup"}'},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": '{"tool": "chat", "response": "你好"}'},
    ]

    return [{"role": "system", "content": system}, *examples, {"role": "user", "content": user_input}]


def build_messages(user_input: str) -> list[dict[str, str]]:
    return build_prompt(user_input)
