"""
Smoke test：驗證 BaseAgent 基本對話與 Tool Calling 是否正常。
"""
from agents import BaseAgent


# ── Test 1：無 Tool 的基本對話 ─────────────────────────────────────────────
def test_simple_chat():
    agent = BaseAgent(
        agent_name="TestAgent",
        system_prompt="你是一個台灣環境法規助理，請用繁體中文簡短回答。",
    )
    reply = agent.chat("空氣污染防制法主要規範什麼？")
    print("=== Simple Chat ===")
    print(reply)
    print()


# ── Test 2：有 Tool 的 Agent ───────────────────────────────────────────────
def get_current_standard(pollutant: str) -> str:
    """模擬查詢法規標準的 tool（之後會換成真實 RAG）。"""
    mock_db = {
        "COD": "放流水標準：COD ≤ 100 mg/L（依廢水種類可能不同）",
        "pH":  "放流水標準：pH 介於 6.0 ~ 9.0",
        "SS":  "放流水標準：SS ≤ 30 mg/L",
    }
    return mock_db.get(pollutant.upper(), f"查無 {pollutant} 的標準資料")


def test_tool_chat():
    tools = [
        {
            "name": "get_current_standard",
            "description": "查詢特定污染物的放流水管制標準",
            "function": get_current_standard,
            "parameters": {
                "type": "object",
                "properties": {
                    "pollutant": {
                        "type": "string",
                        "description": "污染物名稱，例如 COD、pH、SS",
                    }
                },
                "required": ["pollutant"],
            },
        }
    ]

    agent = BaseAgent(
        agent_name="LawQueryAgent",
        system_prompt="你是台灣環境法規助理。當需要查詢管制標準時請呼叫工具，最後用繁體中文回答。",
        tools=tools,
    )

    reply = agent.chat("我們工廠的 COD 排放值是 120 mg/L，請問合規嗎？")
    print("=== Tool Chat ===")
    print(reply)
    print()


if __name__ == "__main__":
    test_simple_chat()
    test_tool_chat()
