from typing import TypedDict, Optional, List, Annotated
from operator import add


class GraphState(TypedDict):
    """
    LangGraph 全域狀態，所有 Agent 共用。
    Annotated[List, add] 表示該欄位採 append 語意（不覆蓋）。
    """

    # --- 使用者輸入 ---
    user_query: str                          # 使用者原始問題
    user_provided_data: str                  # 使用者補充的數值資料（如 COD 120 mg/L）

    # --- Supervisor 路由 ---
    next_agent: Optional[str]                # Supervisor 決定下一個執行的 agent
    needs_data_input: Optional[str]          # None | 'yes' | 'no' | 'uncertain'

    # --- 流程控制 ---
    waiting_for_data_input: bool             # 是否暫停等待使用者補充資料
    data_request_hint: Optional[str]         # IntentNode 產出，告知使用者需提供何種數值
    conversation_turn: int                   # 對話輪數，上限 5
    retry_count: int                         # 當前 LawJudgmentNode 重試次數，上限 3
    retry_exhausted: bool                    # 是否已達重試上限
    total_node_calls: int                    # 全局節點執行計數，防護上限 10

    # --- 各 Agent 產出 ---
    law_search_results: List[dict]           # LawQueryAgent：法規 chunks（含 metadata）
    data_input_result: Optional[dict]        # DataInputAgent：API 回傳資料 或 解析後的使用者數值
    judgment_result: Optional[str]           # JudgmentAgent：合規判斷結論

    # --- 最終輸出 ---
    cited_articles: List[str]                # 引用的法條條號列表
    final_answer: str                        # OutputAgent 產出的最終回覆

    # --- 錯誤與訊息 ---
    error: Optional[str]                     # 錯誤訊息（API 失敗等）
    messages: Annotated[List[dict], add]     # 對話訊息串流（供 UI 顯示用，只增不改）


def initial_state(user_query: str) -> GraphState:
    """建立一個乾淨的初始 State。"""
    return GraphState(
        user_query=user_query,
        user_provided_data="",
        next_agent=None,
        needs_data_input=None,
        waiting_for_data_input=False,
        data_request_hint=None,
        conversation_turn=0,
        retry_count=0,
        retry_exhausted=False,
        total_node_calls=0,
        law_search_results=[],
        data_input_result=None,
        judgment_result=None,
        cited_articles=[],
        final_answer="",
        error=None,
        messages=[],
    )
