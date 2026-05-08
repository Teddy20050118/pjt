"""
台灣環境法規諮詢系統 — Multi-Agent Graph
=========================================
架構說明（4 Node 優化方案）：

    原計畫書 5-Agent               本檔案 4-Node
    ─────────────────────────      ─────────────────────────
    SupervisorAgent (LLM)      →   IntentNode (輕量 LLM，純分類)
    LawQueryAgent (LLM+RAG)    →   LawJudgmentNode (合併，一次推理)
    JudgmentAgent (LLM)        ↗   （同上）
    DataInputAgent (部分 LLM)  →   DataInputNode (多數純 Python)
    OutputAgent (LLM)          →   OutputNode (純 Python，無 LLM)

合併 LawQuery + Judgment 的理由：
    兩者輸入集合 100% 重疊（法條 chunks + 使用者數值），分開執行浪費一次
    完整 LLM 呼叫並增加 state 傳遞成本。對 llama3.1:8b 而言，單次 chain-of-thought
    推理品質優於拆成兩段中間結果再接續。

資源限制（llama3.1:8b，8GB VRAM）：
    單次 LLM 呼叫 prompt 上限：1500 tokens
    全流程最壞情況（含 3 次 retry）：約 3000~4500 tokens

依賴套件：
    pip install langgraph
"""

import json
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END

from agents import BaseAgent
from pollutant_catalog import (
    aliases_for_pollutant,
    infer_industry_hint,
    parse_measurements,
    retrieval_terms,
)
from rag import LawRAG
from standards import evaluate_records
from state import GraphState


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1：常數設定
# ═══════════════════════════════════════════════════════════════════════════════

# RAG 查詢結果的資源管控參數
MAX_CHUNKS = 3          # 最多傳入 LLM 的法條數量
CHUNK_CHARS = 300       # 每個法條的最大字元數（截斷用）
SCORE_THRESHOLD = 0.6   # cosine similarity 門檻，低於此值視為低相關性

# 流程防護上限
MAX_RETRY = 3           # LawJudgmentNode 最多重試次數
MAX_NODE_CALLS = 10     # 全局節點執行計數上限（防止無限迴圈）
MAX_TURNS = 5           # 等待使用者輸入的對話輪數上限

# 模型設定
MODEL_NAME = "llama3.1:8b"
OLLAMA_HOST = "127.0.0.1:11434"

# ChromaDB 設定（與 rag.py 對應）
CHROMA_DB_PATH = "./chroma_db"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2：共用工具函數（純 Python，無 LLM）
# ═══════════════════════════════════════════════════════════════════════════════

def format_chunks_for_prompt(chunks: List[dict]) -> str:
    """
    將 RAG 查詢結果截斷並格式化，組合成適合塞入 LLM prompt 的字串。

    控制策略：
        1. 過濾 score < SCORE_THRESHOLD 的低相關結果
        2. 若全部低於門檻，保留最高分前 2 筆（保底）
        3. 最多保留 MAX_CHUNKS 筆
        4. 每筆文字截斷至 CHUNK_CHARS 字元

    Args:
        chunks: LawRAG.retrieve() 的回傳值
                每筆包含 text, law_name, article, score

    Returns:
        str: 格式化後的法條節錄字串
    """
    if not chunks:
        return "（查無相關法條）"

    # 步驟一：依 score 門檻過濾
    filtered = [c for c in chunks if c.get("score", 0) >= SCORE_THRESHOLD]

    # 保底：若全部低於門檻，取最高分前 2 筆
    if not filtered:
        filtered = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:2]

    # 步驟二：取前 MAX_CHUNKS 筆（已按 retrieve 回傳順序排序，通常是 score 降序）
    top_chunks = filtered[:MAX_CHUNKS]

    lines = []
    for chunk in top_chunks:
        law_name = chunk.get("law_name", "")
        article = chunk.get("article", "")
        text = chunk.get("text", "")

        # 截斷文字，避免過長的條文佔滿 context window
        text_snippet = text[:CHUNK_CHARS]
        if len(text) > CHUNK_CHARS:
            text_snippet += "..."

        lines.append(f"[{law_name} {article}]\n{text_snippet}")

    return "\n\n".join(lines)


def parse_json_safe(text: str) -> Optional[dict]:
    """
    從 LLM 回應文字中安全解析 JSON。
    LLM 有時會在 JSON 前後加入說明文字或 Markdown 程式碼區塊，
    此函數使用正規表達式找出第一個完整的 JSON 物件。

    Args:
        text: LLM 回應的原始字串

    Returns:
        dict: 解析成功時回傳 Python dict；失敗時回傳 None
    """
    if not text:
        return None

    # 嘗試直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 嘗試從文字中找出 JSON 物件（處理 LLM 在 JSON 前後加說明的情況）
    # 逐字掃描配對括號，支援任意深度巢狀
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def build_final_answer(state: GraphState) -> str:
    """
    純 Python 格式化最終回答，不呼叫 LLM。
    根據 state 中的 judgment_result、cited_articles、error 組合輸出字串。

    Args:
        state: 當前 GraphState

    Returns:
        str: 格式化後的最終回答字串
    """
    # 情況一：已達重試上限或發生無法恢復的錯誤
    if state.get("retry_exhausted") or (state.get("error") and not state.get("judgment_result")):
        error_msg = state.get("error", "未知錯誤")
        return (
            "很抱歉，系統無法完成本次查詢。\n"
            f"錯誤原因：{error_msg}\n"
            "建議您換個方式描述問題，或提供更具體的數值資料後重新查詢。"
        )

    # 情況二：正常輸出（judgment_result 為 dict）
    result = state.get("judgment_result")
    if isinstance(result, dict):
        judgment = result.get("judgment", "資料不足")
        reason = result.get("reason", "")
        cited = result.get("cited_articles", [])
    elif isinstance(result, str):
        # 備援：judgment_result 為字串時直接使用
        return result
    else:
        return "系統未能產生有效的判斷結果，請重新查詢。"

    # 組合引用條號（來自 judgment_result 或 state.cited_articles）
    all_cited = cited or state.get("cited_articles", [])
    cited_str = "、".join(all_cited) if all_cited else "（無具體引用條號）"

    # 組合來源法規名稱
    law_sources = list({
        c.get("law_name", "") for c in state.get("law_search_results", []) if c.get("law_name")
    })
    sources_str = "、".join(law_sources) if law_sources else "（未知來源）"

    return (
        f"【合規判斷結果】\n{judgment}\n\n"
        f"【判斷說明】\n{reason}\n\n"
        f"【引用條號】\n{cited_str}\n\n"
        f"【參考法規來源】\n{sources_str}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3：Node 函數定義
# ═══════════════════════════════════════════════════════════════════════════════
#
# 每個 Node 函數簽名：(state: GraphState) -> dict
# 回傳 dict 只含「本次要更新的欄位」，LangGraph 會自動 merge 進 state。
# 回傳空 dict {} 表示 state 不變（用於跳過邏輯）。
#
# ═══════════════════════════════════════════════════════════════════════════════

def entry_router_node(state: GraphState) -> dict:
    """
    進入點路由節點（純 Python，不呼叫 LLM）。

    功能：
        處理 re-entry 情境——當使用者提供數值後重新呼叫 graph.invoke() 時，
        此節點依據 state 中已有的欄位決定跳過哪些 Node，
        避免重複執行 IntentNode 的 LLM 分類呼叫。

    State 更新：
        total_node_calls += 1（計數，供防護條件邊使用）
    """
    return {"total_node_calls": state.get("total_node_calls", 0) + 1}


def intent_node(state: GraphState) -> dict:
    """
    意圖分類節點（輕量 LLM，1 次呼叫）。

    功能：
        判斷使用者問題是否需要具體數值（如 COD mg/L、pH 值）才能進行合規判斷。
        設定 needs_data_input 與 data_request_hint。

    Prompt 設計重點：
        - 角色定義（環境法規意圖分類器）
        - 分類標準說明
        - JSON 輸出格式
        - 不傳入：法規列表、歷史對話、法條文字

    State 更新：
        needs_data_input, data_request_hint, total_node_calls, error（失敗時）
    """
    agent = BaseAgent(
        agent_name="意圖分類器",
        model=MODEL_NAME,
        host=OLLAMA_HOST,
        system_prompt=(
            "你是台灣環境法規諮詢系統的意圖分類器。\n"
            "你的唯一任務是判斷使用者的問題，是否需要使用者額外補充具體排放數值，才能進行合規判斷。\n\n"
            "【問題類型與判斷規則】\n\n"
            "類型一：合規性判斷問題，且問題中沒有任何具體數值\n"
            "  定義：使用者詢問是否合規、是否超標、是否會被罰款、是否違法等，"
            "但問題中完全沒有提供任何具體的排放數值。\n"
            "  規則：回傳 needs_data_input=\"yes\"，並在 data_hint 說明需要提供哪些數值。\n\n"
            "類型二：合規性判斷問題，且問題中已包含具體數值\n"
            "  定義：使用者詢問合規性，且問題中已包含具體的排放數值（如 COD 120、pH 7.5）。\n"
            "  規則：回傳 needs_data_input=\"no\"，data_hint 填入空字串。\n"
            "  原因：後續節點會自動從問題中萃取數值，無需再次向使用者索取。\n\n"
            "類型三：純法規內容查詢\n"
            "  定義：使用者只詢問法條內容、罰則規定、法規說明等，不涉及自身排放數值的合規判斷。\n"
            "  規則：回傳 needs_data_input=\"no\"，data_hint 填入空字串。\n\n"
            "【重要判斷提示】\n"
            "只要問題含有『合規嗎』、『會罰款嗎』、『超標嗎』、『違法嗎』、『是否符合規定』"
            "等合規性判斷詞語，且問題中沒有任何具體數值，就必須回傳 needs_data_input=\"yes\"。\n"
            "若無法確定，請優先偏向 needs_data_input=\"yes\"，以免遺漏必要的數值收集步驟。\n\n"
            "【範例】\n"
            "使用者：『我們工廠廢水合規嗎？』\n"
            "-> {\"needs_data_input\": \"yes\", "
            "\"data_hint\": \"請提供具體的廢水檢測數值，例如 COD、BOD、SS、pH、氨氮等及其單位。\"}\n\n"
            "使用者：『COD 120 mg/L 會被罰嗎？』\n"
            "-> {\"needs_data_input\": \"no\", \"data_hint\": \"\"}\n\n"
            "使用者：『水污法第7條規定什麼？』\n"
            "-> {\"needs_data_input\": \"no\", \"data_hint\": \"\"}\n\n"
            "【輸出格式】\n"
            "只回傳以下 JSON，不得包含任何其他文字：\n"
            "{\"needs_data_input\": \"yes 或 no\", "
            "\"data_hint\": \"需要提供哪些排放數值（若 needs_data_input 為 no 則填入空字串）\"}"
        ),
    )

    prompt = f"使用者問題：{state['user_query']}"
    raw_response = agent.chat(prompt, reset_history=True)

    parsed = parse_json_safe(raw_response)

    if parsed is None:
        # 解析失敗時保守設為 yes，避免靜默跳過數據收集
        return {
            "needs_data_input": "yes",
            "data_request_hint": "請提供相關污染物的排放數值（如 COD、BOD、SS、pH 等）及其單位。",
            "total_node_calls": state.get("total_node_calls", 0) + 1,
            "error": f"IntentNode JSON 解析失敗，已保守設為 yes。原始回應：{raw_response[:100]}",
        }

    needs_data = parsed.get("needs_data_input", "yes")
    hint = parsed.get("data_hint", "") or None

    return {
        "needs_data_input": needs_data,
        "data_request_hint": hint,
        "total_node_calls": state.get("total_node_calls", 0) + 1,
        "error": None,
    }


def data_input_node(state: GraphState) -> dict:
    """
    數據輸入節點（多數情況為純 Python，LLM 僅作為最後手段）。

    解析來源優先順序：
        1. user_provided_data（使用者後續補充的數值，優先使用）
        2. user_query（user_provided_data 為空時，從原始問題中嘗試提取，
           處理「COD 120 mg/L，合規嗎？」此類問題自帶數值的情境）

    解析方法優先順序：
        1. Regex（_parse_numeric_data）：速度快、不耗 VRAM
        2. LLM 解析：處理 regex 無法辨識的自由文字（如「大概一百多 mg/L」）

    只有在「Regex 與 LLM 均無法從兩個來源中萃取有效數值」時，
    才設定 waiting_for_data_input = True，暫停流程等待使用者輸入。

    【預留介面】：此節點未來可在解析流程前加入環保署 API 呼叫，
                 若 API 回傳有效數據，直接填入 data_input_result。

    State 更新：
        data_input_result, waiting_for_data_input, total_node_calls, error
    """
    total = state.get("total_node_calls", 0) + 1

    # ── 預留介面：環保署 API 查詢（目前為模擬，後續介接真實 API）──────────────
    # api_result = fetch_moenv_api(state["user_query"])
    # if api_result:
    #     return {"data_input_result": api_result, "waiting_for_data_input": False,
    #             "total_node_calls": total}
    # ─────────────────────────────────────────────────────────────────────────────

    # 決定解析來源：優先使用 user_provided_data，為空則退回 user_query
    user_provided = state.get("user_provided_data", "").strip()
    user_query = state.get("user_query", "").strip()

    text_to_parse = user_provided if user_provided else user_query

    # 步驟一：Regex 解析（不耗 VRAM，優先嘗試）
    parsed_data = _parse_numeric_data(text_to_parse)
    if parsed_data:
        return {
            "data_input_result": parsed_data,
            "waiting_for_data_input": False,
            "total_node_calls": total,
        }

    # 步驟二：LLM 解析（處理 regex 無法辨識的模糊描述）
    agent = BaseAgent(
        agent_name="數值解析器",
        model=MODEL_NAME,
        host=OLLAMA_HOST,
        system_prompt=(
            "你是一個數值萃取工具，負責從文字中提取污染物排放數值資料。\n"
            "請只回傳 JSON，格式：\n"
            "{\"pollutant\": \"污染物名稱\", \"value\": 數值（數字）, \"unit\": \"單位\"}\n"
            "若包含多個污染物，回傳陣列：[{...}, {...}]\n"
            "若文字中完全沒有任何數值資料，回傳 {\"error\": \"無法解析\"}"
        ),
    )
    raw_response = agent.chat(f"請從以下文字中萃取數值：{text_to_parse}", reset_history=True)
    parsed_llm = parse_json_safe(raw_response)

    if parsed_llm and "error" not in parsed_llm:
        return {
            "data_input_result": parsed_llm,
            "waiting_for_data_input": False,
            "total_node_calls": total,
        }

    # 步驟三：Regex 與 LLM 均失敗，確認真的缺乏數值 → 暫停等待使用者輸入
    # 只有到達此步驟，才判定需要使用者補充資料
    return {
        "waiting_for_data_input": True,
        "conversation_turn": state.get("conversation_turn", 0) + 1,
        "total_node_calls": total,
    }


def law_judgment_node(state: GraphState) -> dict:
    """
    法規查詢與合規判斷節點（核心推理節點，1 次 LLM 呼叫）。

    功能：
        1. 【Python 層】呼叫 LawRAG.retrieve() 查詢相關法條（不透過 tool calling）
        2. 【Python 層】截斷並格式化 chunks（控制 token 用量）
        3. 【LLM 層】一次推理完成：法條解讀 + 合規判斷 + 條號引用
        4. 解析 LLM 的 JSON 回應，更新 state

    LLM 不負責 RAG 查詢的理由：
        直接 Python 呼叫 rag.retrieve() 比 tool calling 更穩定、更省 token，
        且可在 Python 層精確控制截斷邏輯，不依賴 LLM 判斷查詢策略。

    重試邏輯：
        若 LLM 回傳無法解析的非 JSON，retry_count += 1，最多 MAX_RETRY 次。
        超過上限設 retry_exhausted = True，進入 OutputNode 輸出錯誤。

    State 更新：
        law_search_results, judgment_result, cited_articles,
        retry_count, retry_exhausted, total_node_calls, error
    """
    total = state.get("total_node_calls", 0) + 1
    retry_count = state.get("retry_count", 0)

    data_result = state.get("data_input_result")
    has_error = isinstance(data_result, dict) and data_result.get("error")

    # 防呆：若路由漏掉數值解析，仍嘗試從原始問題補解析，避免把
    # 「COD 120 mg/L」誤判成「使用者未提供具體數值」。
    if not data_result:
        data_result = _parse_numeric_data(state.get("user_query", ""))
        has_error = isinstance(data_result, dict) and data_result.get("error")

    # 步驟一：呼叫 RAG 查詢（純 Python，不透過 LLM tool calling）
    try:
        rag = LawRAG(db_path=CHROMA_DB_PATH)
        law_query = _build_law_retrieval_query(state["user_query"], data_result)
        raw_chunks = rag.retrieve(query=law_query, n_results=10)
    except Exception as e:
        # RAG 失敗時不重試，直接帶錯誤進 OutputNode
        return {
            "law_search_results": [],
            "error": f"RAG 查詢失敗：{str(e)}",
            "retry_exhausted": True,
            "total_node_calls": total,
        }

    # 步驟二：格式化 chunks（截斷 + 過濾低相關性結果）
    chunks_for_prompt = format_chunks_for_prompt(raw_chunks)

    # 步驟三：組合 user message（控制 token 用量）
    if data_result and not has_error:
        data_str = _format_data_result(data_result)
    else:
        data_str = "（使用者未提供具體數值）"

    structured_judgment = _build_structured_judgment(data_result, state["user_query"])
    if structured_judgment:
        return {
            "law_search_results": raw_chunks,
            "judgment_result": structured_judgment,
            "cited_articles": structured_judgment.get("cited_articles", []),
            "retry_count": retry_count,
            "error": None,
            "total_node_calls": total,
        }

    user_message = (
        f"【相關法條節錄】\n{chunks_for_prompt}\n\n"
        f"【使用者問題】\n{state['user_query']}\n\n"
        f"【排放數值資料】\n{data_str}\n\n"
        "請根據以上法條與問題類型，依照系統規則回答。"
    )

    # 步驟四：LLM 推理
    agent = BaseAgent(
        agent_name="法規判斷師",
        model=MODEL_NAME,
        host=OLLAMA_HOST,
        system_prompt=(
            "你是台灣環境法規專家，能夠回答合規判斷與法規內容查詢兩類問題。\n\n"
            "【回答規則】\n\n"
            "情況一：合規性判斷（使用者提供了具體排放數值）\n"
            "  - 將使用者數值與法條規定標準逐一比較。\n"
            "  - 明確判斷合規或違規，並說明依據的條號與標準數值。\n"
            "  - 若【排放數值資料】已有 COD 120 mg/L 這類資料，不得回答「本案例無相關數值」。\n"
            "  - 若已有使用者數值但相關法條節錄沒有明確限值，judgment 填入「資料不足」，reason 必須說明「已有使用者數值，但未檢索到可比對的法定限值」。\n"
            "  - judgment 填入「合規」或「違規」。\n\n"
            "情況二：純法規內容查詢（使用者詢問法條內容、罰則或法規說明）\n"
            "  - 整理並說明檢索到的相關法條內容，直接回答使用者問題。\n"
            "  - judgment 填入「純法規查詢」。\n"
            "  - reason 欄位詳細整理法條內容，不得回答「資料不足」。\n\n"
            "情況三：合規性判斷但無具體數值，且法條亦無法提供答案\n"
            "  - judgment 填入「資料不足」，並在 reason 說明需要哪些數值才能判斷。\n\n"
            "【輸出格式】\n"
            "必須只回傳 JSON，不得包含任何其他文字：\n"
            "{\"judgment\": \"合規/違規/純法規查詢/資料不足\", "
            "\"reason\": \"說明內容（300字以內）\", "
            "\"cited_articles\": [\"第X條\", ...]}"
        ),
    )

    raw_response = agent.chat(user_message, reset_history=True)
    parsed = parse_json_safe(raw_response)

    if parsed is None:
        # JSON 解析失敗，計入重試
        new_retry = retry_count + 1
        is_exhausted = new_retry >= MAX_RETRY
        return {
            "law_search_results": raw_chunks,
            "retry_count": new_retry,
            "retry_exhausted": is_exhausted,
            "total_node_calls": total,
            "error": f"LawJudgmentNode 第 {new_retry} 次 JSON 解析失敗。原始回應：{raw_response[:150]}",
        }

    # 成功：更新 state
    cited = parsed.get("cited_articles", [])
    return {
        "law_search_results": raw_chunks,
        "judgment_result": parsed,
        "cited_articles": cited,
        "retry_count": retry_count,    # 成功時不累加
        "error": None,
        "total_node_calls": total,
    }


def output_node(state: GraphState) -> dict:
    """
    最終輸出節點（純 Python，不呼叫 LLM）。

    功能：
        呼叫 build_final_answer() 將 state 中的結構化結果
        格式化為可讀的繁體中文字串，寫入 final_answer 欄位。

    不使用 LLM 的理由：
        LawJudgmentNode 已產出結構化 JSON，格式固定，
        Python 字串模板即可完成組合，且避免 LLM 幻覺改變判斷內容。

    State 更新：
        final_answer, messages（append 最終訊息）
    """
    answer = build_final_answer(state)

    return {
        "final_answer": answer,
        "messages": [{"role": "assistant", "content": answer}],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4：路由函數（條件邊）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 所有路由函數均為純 Python，不呼叫 LLM。
# 全局防護：total_node_calls > MAX_NODE_CALLS 時一律跳至 output_node。
#
# ═══════════════════════════════════════════════════════════════════════════════

def route_entry(state: GraphState) -> str:
    """
    entry_router_node 執行後的路由決策。

    re-entry 跳過邏輯（使用者提供數值後重新呼叫 graph.invoke() 的情況）：
        1. 若 data_input_result 已有值  → 直接進 law_judgment_node
        2. 若 needs_data_input == "yes" 且 user_provided_data 非空
                                        → 進 data_input_node（解析數值）
        3. 若 needs_data_input 已設定   → 依其值路由（跳過 IntentNode）
        4. 否則                         → 從 intent_node 開始正常流程

    全局防護：
        total_node_calls 超限時強制進 output_node。
    """
    # 全局防護：超過節點呼叫上限，強制終止
    if state.get("total_node_calls", 0) > MAX_NODE_CALLS:
        return "output"

    # re-entry：數值已解析，直接進入主推理
    if state.get("data_input_result") is not None:
        return "law_judgment"

    # re-entry：已知需要數值且使用者已輸入
    if state.get("needs_data_input") == "yes" and state.get("user_provided_data", ""):
        return "data_input"

    # re-entry：已完成意圖分類，不重複 LLM 呼叫
    if state.get("needs_data_input") is not None:
        if state.get("needs_data_input") == "yes":
            return "data_input"
        if _query_has_numeric_data(state.get("user_query", "")):
            return "data_input"
        return "law_judgment"

    # 正常首次進入
    return "intent"


def route_after_intent(state: GraphState) -> str:
    """
    intent_node 執行後的路由決策。

    全局防護同上。
    """
    if state.get("total_node_calls", 0) > MAX_NODE_CALLS:
        return "output"

    if state.get("needs_data_input") == "yes":
        return "data_input"

    if _query_has_numeric_data(state.get("user_query", "")):
        return "data_input"

    # "no" 或 "uncertain" 皆直接進入主推理
    return "law_judgment"


def route_after_data_input(state: GraphState) -> str:
    """
    data_input_node 執行後的路由決策。

    全局防護同上。
    waiting_for_data_input == True 時回傳 END，讓主程式等待使用者輸入。
    若已等待超過 MAX_TURNS 輪，放棄等待，帶空數據繼續推理。
    """
    if state.get("total_node_calls", 0) > MAX_NODE_CALLS:
        return "output"

    if state.get("waiting_for_data_input"):
        if state.get("conversation_turn", 0) >= MAX_TURNS:
            # 超過等待輪數上限，放棄等待，帶空數據繼續
            return "law_judgment"
        # 正常暫停，等待使用者輸入
        return END

    return "law_judgment"


def route_after_judgment(state: GraphState) -> str:
    """
    law_judgment_node 執行後的路由決策。

    重試邏輯：
        僅在 LLM 輸出格式錯誤（JSON 解析失敗）時重試，
        而非「找不到法條」（後者是有效結果，不應重試）。
    """
    if state.get("total_node_calls", 0) > MAX_NODE_CALLS:
        return "output"

    # 已達重試上限，強制輸出
    if state.get("retry_exhausted"):
        return "output"

    # 有錯誤且未超過重試上限，重新執行 law_judgment_node
    if state.get("error") and state.get("retry_count", 0) < MAX_RETRY:
        # 確認是「JSON 解析失敗」型的錯誤才重試（RAG 失敗不重試）
        error_msg = state.get("error", "")
        if "JSON 解析失敗" in error_msg:
            return "law_judgment"

    # 正常輸出
    return "output"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5：Graph 建構與對外介面
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    建構並編譯 LangGraph StateGraph。

    Graph 結構（4 Node + 1 路由節點）：
        entry_router_node
          ├─[route_entry]─→ intent_node
          │                   └─[route_after_intent]─→ data_input_node 或 law_judgment_node
          ├─[route_entry]─→ data_input_node
          │                   └─[route_after_data_input]─→ law_judgment_node 或 END
          └─[route_entry]─→ law_judgment_node
                              └─[route_after_judgment]─→ law_judgment_node（retry）或 output_node

    Returns:
        編譯後的 LangGraph CompiledGraph 物件
    """
    builder = StateGraph(GraphState)

    # 新增所有 Node
    builder.add_node("entry_router", entry_router_node)
    builder.add_node("intent", intent_node)
    builder.add_node("data_input", data_input_node)
    builder.add_node("law_judgment", law_judgment_node)
    builder.add_node("output", output_node)

    # 設定進入點
    builder.set_entry_point("entry_router")

    # entry_router 的條件邊
    builder.add_conditional_edges(
        "entry_router",
        route_entry,
        {
            "intent": "intent",
            "data_input": "data_input",
            "law_judgment": "law_judgment",
            "output": "output",
        },
    )

    # intent_node 的條件邊
    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "data_input": "data_input",
            "law_judgment": "law_judgment",
            "output": "output",
        },
    )

    # data_input_node 的條件邊
    builder.add_conditional_edges(
        "data_input",
        route_after_data_input,
        {
            "law_judgment": "law_judgment",
            "output": "output",
            END: END,
        },
    )

    # law_judgment_node 的條件邊（含重試迴路）
    builder.add_conditional_edges(
        "law_judgment",
        route_after_judgment,
        {
            "law_judgment": "law_judgment",
            "output": "output",
        },
    )

    # output_node 固定進入 END
    builder.add_edge("output", END)

    return builder.compile()


def run_query(user_query: str, user_provided_data: str = "") -> dict:
    """
    對外介面：執行一次完整的法規諮詢查詢。

    若使用者問題需要數值但未提供，回傳的 state["waiting_for_data_input"] 為 True，
    呼叫方應取得 state["data_request_hint"] 提示使用者輸入，
    再以更新後的 state 呼叫 run_query_with_state() 繼續執行。

    Args:
        user_query:         使用者問題字串
        user_provided_data: 使用者已知的數值資料（可為空字串）

    Returns:
        dict: 最終的 GraphState，呼叫方可取 state["final_answer"] 顯示結果
    """
    from state import initial_state

    state = initial_state(user_query)
    state["user_provided_data"] = user_provided_data

    graph = build_graph()
    result = graph.invoke(state)
    return result


def run_query_with_state(state: dict) -> dict:
    """
    對外介面：以現有 state 繼續執行（re-entry 用途）。

    使用情境：
        主程式偵測到 state["waiting_for_data_input"] == True，
        收集使用者輸入後，更新 state["user_provided_data"] 與
        state["waiting_for_data_input"] = False，再呼叫此函數繼續。

    Args:
        state: 包含 user_provided_data 已更新的 GraphState dict

    Returns:
        dict: 更新後的最終 GraphState
    """
    graph = build_graph()
    return graph.invoke(state)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 附錄：私有工具函數
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_numeric_data(text: str) -> Optional[dict]:
    """
    以正規表達式解析常見污染物數值格式。
    格式範例：「COD 120 mg/L」、「pH 7.5」、「SS 30mg/L」

    Args:
        text: 使用者輸入的數值描述字串

    Returns:
        dict: {"pollutant": ..., "value": ..., "unit": ...}
              若包含多筆，回傳 list of dict；
              若解析失敗，回傳 None
    """
    return parse_measurements(text)


def _query_has_numeric_data(text: str) -> bool:
    """判斷使用者問題是否已包含可解析的污染物數值。"""
    return _parse_numeric_data(text) is not None


def _format_data_result(data_result) -> str:
    """將資料解析結果格式化給判斷節點使用。"""
    if isinstance(data_result, list):
        return "、".join(
            f"{d.get('pollutant')} {d.get('value')} {d.get('unit')}"
            for d in data_result if isinstance(d, dict)
        )
    if isinstance(data_result, dict) and "pollutant" in data_result:
        return (f"{data_result.get('pollutant')} "
                f"{data_result.get('value')} {data_result.get('unit')}")
    if isinstance(data_result, dict) and "raw" in data_result:
        return data_result.get("raw", "")
    return str(data_result)


def _build_law_retrieval_query(user_query: str, data_result) -> str:
    """
    建立較適合 RAG 的法規查詢字串。

    使用者問「COD 120 mg/L 合規嗎」時，原句容易只命中泛用水污法條。
    加入污染物別名、放流水標準、限值等關鍵詞，提升命中標準/附表的機率。
    """
    pollutants = []
    records = data_result if isinstance(data_result, list) else [data_result]
    for record in records:
        if isinstance(record, dict) and record.get("pollutant"):
            pollutants.append(str(record["pollutant"]))

    aliases = []
    for pollutant in pollutants:
        aliases.extend(aliases_for_pollutant(pollutant))

    query_parts = [user_query, *retrieval_terms(), "mg/L"]
    query_parts.extend(aliases)
    return " ".join(part for part in query_parts if part)


def _build_structured_judgment(data_result, user_query: str = "") -> Optional[dict]:
    """
    使用結構化標準資料做合規初判。

    有標準時直接用通用比較器；沒有標準時交回 None，讓 RAG + LLM 處理。
    """
    if not data_result:
        return None

    evaluations = evaluate_records(data_result, industry_hint=_infer_industry_hint(user_query))
    matched = [item for item in evaluations if item["status"] in {"passed", "failed"}]
    missing_condition_items = [
        item for item in evaluations if item["status"] == "missing_conditions"
    ]

    if missing_condition_items:
        reason_lines = []
        cited_articles = []
        for item in missing_condition_items:
            record = item["record"]
            standard = item["standard"]
            missing_labels = [
                condition.get("label") or condition.get("field")
                for condition in item.get("missing_conditions", [])
            ]
            reason_lines.append(
                f"{record.get('pollutant')} 已找到可比對標準，但仍缺少必要條件："
                f"{'、'.join(missing_labels)}。請補充後才能判斷合規。"
            )
            source = standard.get("source_article") or standard.get("source_name")
            if source:
                cited_articles.append(source)

        return {
            "judgment": "資料不足",
            "reason": "\n".join(reason_lines),
            "cited_articles": list(dict.fromkeys(cited_articles)),
        }

    if not matched:
        return None

    failed = [item for item in matched if item["status"] == "failed"]
    judgment = "違規" if failed else "合規"

    reason_lines = []
    cited_articles = []
    for item in matched:
        record = item["record"]
        standard = item["standard"]
        comparison = item["comparison"]
        pollutant = record.get("pollutant")
        value = record.get("value")
        unit = record.get("unit", standard.get("unit", ""))
        limit_value = standard.get("limit_value")
        source = standard.get("source_article") or standard.get("source_name")

        if standard.get("limit_type") == "range":
            limit_text = f"{standard.get('min_value')}~{standard.get('max_value')} {standard.get('unit', '')}".strip()
        else:
            limit_text = f"{limit_value} {standard.get('unit', '')}".strip()

        status_text = "符合" if comparison["passed"] else "超過"
        reason_lines.append(
            f"{pollutant} 檢測值為 {value:g} {unit}，"
            f"{standard.get('industry')}標準限值為 {limit_text}，"
            f"比對結果：{comparison['detail']}，判定為{status_text}標準。"
        )
        if source:
            cited_articles.append(source)

    return {
        "judgment": judgment,
        "reason": "\n".join(reason_lines),
        "cited_articles": list(dict.fromkeys(cited_articles)),
    }


def _infer_industry_hint(user_query: str) -> Optional[str]:
    """從使用者問題抓取可套用結構化標準的產業提示。"""
    return infer_industry_hint(user_query)


if __name__ == "__main__":
    # 這是你要丟給系統測試的真實情境問題
    test_query = "	工廠廢水 COD 測到 120 mg/L，這樣合規嗎？" 
    #我們工廠廢水 COD 測到 120 mg/L，這樣合規嗎？
    print(f"使用者提問: {test_query}\n")
    # 準備初始狀態 (State)
    from state import initial_state # 這裡需要從 state.py 引入初始狀態的設定
    
    # 使用 Claude 寫好的 initial_state 函數來初始化
    current_state = initial_state(test_query)

    # 【關鍵修正】：呼叫 build_graph() 取得編譯好的圖表物件
    app = build_graph()

    # 執行 Graph 流程
    try:
        # 這裡改成 app.stream
        for output in app.stream(current_state):
            for node_name, state_update in output.items():
                print(f"[節點執行完畢]: {node_name}")
                print(f"狀態變化: {state_update}\n")
                print("-" * 40)
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
