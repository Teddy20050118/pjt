from typing import Optional, List, Dict, Callable
from models import OllamaClient

class BaseAgent:
    """
    所有 Agent 的基礎類別，提供共用的 LLM 對話與 Tool Calling 功能。

    Tool 格式（傳入 tools 參數）：
    {
        "name": "tool_name",
        "description": "說明",
        "function": callable,          # 實際執行的 Python function
        "parameters": {                # Ollama/OpenAI function schema
            "type": "object",
            "properties": { ... },
            "required": [...]
        }
    }
"""

    def __init__(self,
                 agent_name: str,
                 model: str = "llama3.1:8b",
                 host: str = "127.0.0.1:11434",
                 system_prompt: str = "",
                 tools: Optional[List[Dict]] = None,
                 max_iteration: int = 5,
                 options: Optional[dict] = None,
                 ):
        self.agent_name = agent_name
        self.model = model
        self.host = host
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_iteration = max_iteration
        self.options = options

        self.client = OllamaClient(host=host)
        self.history: List[Dict] = []

        # { tool_name: callable }
        self.tools_registry: Dict[str, Callable] = {}
        self._register_tools()

    # ------------------------------------------------------------------ #
    #  內部工具管理                                                         #
    # ------------------------------------------------------------------ #

    def _register_tools(self):
        """將 tools 列表中的 callable 註冊到 registry。"""
        for tool in self.tools:
            self.tools_registry[tool["name"]] = tool["function"]

    def _ollama_tool_schemas(self) -> List[Dict]:
        """轉換成 Ollama 接受的 tools schema（不含 function callable）。"""
        schemas = []
        for tool in self.tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        return schemas

    # ------------------------------------------------------------------ #
    #  對外主要介面                                                         #
    # ------------------------------------------------------------------ #

    def chat(self,
             user_message: str,
             reset_history: bool = False,
             ) -> str:
        """
        Args:
            user_message:  使用者訊息
            reset_history: 是否清空歷史紀錄
        Returns:
            str: LLM 最終回應
        """
        if reset_history:
            self.history = []

        if not self.tools:
            return self._simple_chat(user_message)

        return self._chat_with_tools(user_message)

    def clean_history(self):
        self.history = []

    # ------------------------------------------------------------------ #
    #  內部對話實作                                                         #
    # ------------------------------------------------------------------ #

    def _simple_chat(self, user_message: str) -> str:
        """不使用 tool 的單次對話。"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=self.options
            )
            assistant_message = response["message"]["content"]

            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            return f"[{self.agent_name}] 發生錯誤: {str(e)}"

    def _chat_with_tools(self, user_message: str) -> str:
        """
        使用 Ollama native tool calling 的對話迴圈。
        Ollama 回傳 tool_calls 時自動執行對應 function，
        並把結果以 tool role 送回，直到 LLM 不再呼叫工具為止。
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        tool_schemas = self._ollama_tool_schemas()
        iteration = 0

        while iteration < self.max_iteration:
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas,
                    options=self.options
                )

                response_message = response["message"]
                tool_calls = response_message.get("tool_calls") or []

                # LLM 不再呼叫工具 → 回傳最終答案
                if not tool_calls:
                    final_answer = response_message["content"]
                    self.history.append({"role": "user", "content": user_message})
                    self.history.append({"role": "assistant", "content": final_answer})
                    return final_answer

                # 把 assistant 的 tool_calls 加入訊息鏈
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

                # 執行每個 tool call 並把結果送回
                for tool_call in tool_calls:
                    tool_result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                    })

                iteration += 1

            except Exception as e:
                return f"[{self.agent_name}] 發生錯誤: {str(e)}"

        # 超過 max_iteration 仍未得到最終答案，強制結束
        return f"[{self.agent_name}] 已達最大迭代次數 ({self.max_iteration})，無法完成回答。"

    # ------------------------------------------------------------------ #
    #  Tool 執行                                                            #
    # ------------------------------------------------------------------ #

    def _execute_tool(self, tool_call: Dict) -> str:
        """
        執行 Ollama native tool_call 物件。
        tool_call 格式: {"function": {"name": "...", "arguments": {...}}}
        """
        func_info = tool_call.get("function", {})
        tool_name = func_info.get("name")
        arguments = func_info.get("arguments", {})

        if tool_name not in self.tools_registry:
            return f"錯誤：Tool '{tool_name}' 不存在"

        try:
            result = self.tools_registry[tool_name](**arguments)
            return str(result)
        except Exception as e:
            return f"Tool '{tool_name}' 執行失敗: {str(e)}"
