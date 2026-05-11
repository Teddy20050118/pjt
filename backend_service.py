from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from law_query_categories import build_category_query, normalize_category
from pollutant_catalog import infer_industry_hint, parse_measurements
from standards import evaluate_records, load_standards


COMPLIANCE_KEYWORDS = (
    "合規",
    "符合",
    "超標",
    "標準",
    "限值",
    "放流水",
    "可不可以排",
    "是否合法",
)

SUMMARY_KEYWORDS = (
    "主要法規",
    "有哪些法規",
    "法規條目",
    "適用法規",
    "相關法規",
    "哪些規定",
    "有哪些規定",
    "這類",
    "類別",
)

FOLLOWUP_KEYWORDS = (
    "那",
    "如果",
    "上一題",
    "剛剛",
    "前面",
    "它",
    "這個",
    "該",
    "呢",
)

CATEGORY_LAW_FILTERS = {
    "permit_plan": ["水污染防治措施計畫及許可申請審查管理辦法"],
    "monitoring_reporting": ["水污染防治措施及檢測申報管理辦法"],
    "penalty": ["違反水污染防治法罰鍰額度裁罰準則", "水污染防治法"],
}


@dataclass
class ConversationMemory:
    conversation_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    last_measurements: Any = None
    last_industry_hint: str | None = None
    last_category: str = "auto"
    last_query: str = ""


class ConversationMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, ConversationMemory] = {}

    def get(self, conversation_id: str | None) -> ConversationMemory:
        key = conversation_id or str(uuid4())
        if key not in self._items:
            self._items[key] = ConversationMemory(conversation_id=key)
        return self._items[key]

    def clear(self) -> None:
        self._items.clear()

    def count(self) -> int:
        return len(self._items)


memory_store = ConversationMemoryStore()


def process_query(
    *,
    query: str,
    conversation_id: str | None,
    category: str = "auto",
    run_graph: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Query is required.")

    normalized_category = normalize_category(category)
    memory = memory_store.get(conversation_id)
    effective_query = build_effective_query(clean_query, memory)
    measurements = parse_measurements(effective_query) or memory.last_measurements
    industry_hint = infer_industry_hint(effective_query) or memory.last_industry_hint
    if should_use_category_summary(effective_query, normalized_category, measurements):
        payload = build_category_summary_payload(
            conversation_id=memory.conversation_id,
            category=normalized_category,
        )
        remember(
            memory,
            clean_query,
            payload.get("final_answer", ""),
            measurements,
            industry_hint,
            normalized_category,
        )
        return payload

    need_data_hint = build_missing_data_hint(
        query=effective_query,
        category=normalized_category,
        measurements=measurements,
        industry_hint=industry_hint,
    )

    if need_data_hint:
        payload = empty_payload(
            conversation_id=memory.conversation_id,
            category=normalized_category,
        )
        payload.update({
            "waiting_for_data_input": True,
            "data_request_hint": need_data_hint,
            "final_answer": need_data_hint,
            "structured_judgment": None,
        })
        remember(memory, clean_query, payload["final_answer"], measurements, industry_hint, normalized_category)
        return payload

    structured_judgment = build_structured_judgment(
        measurements=measurements,
        industry_hint=industry_hint,
        category=normalized_category,
    )

    graph_query = build_graph_query(effective_query, normalized_category, memory)
    try:
        state = run_graph(graph_query)
    except Exception as exc:
        if not structured_judgment:
            raise
        state = {
            "final_answer": build_structured_answer(structured_judgment),
            "law_search_results": [],
            "cited_articles": [],
            "error": f"Graph fallback used: {exc}",
        }
    payload = compact_state(
        state,
        conversation_id=memory.conversation_id,
        category=normalized_category,
        structured_judgment=structured_judgment,
    )

    if structured_judgment and should_prefer_structured_answer(structured_judgment):
        payload["final_answer"] = build_structured_answer(structured_judgment)

    payload["citations"] = merge_citations(
        structured_judgment=structured_judgment,
        state=state,
        category=normalized_category,
        query=effective_query,
    )
    payload["cited_articles"] = [item["title"] for item in payload["citations"]]

    remember(
        memory,
        clean_query,
        payload.get("final_answer", ""),
        measurements,
        industry_hint,
        normalized_category,
    )
    return payload


def build_effective_query(query: str, memory: ConversationMemory) -> str:
    if not memory.last_query:
        return query
    is_followup = len(query) <= 32 or any(keyword in query for keyword in FOLLOWUP_KEYWORDS)
    if not is_followup:
        return query

    context_parts = [f"前一題問題：{memory.last_query}"]
    if memory.last_measurements:
        context_parts.append(f"前一題檢測資料：{memory.last_measurements}")
    if memory.last_industry_hint:
        context_parts.append(f"前一題適用對象：{memory.last_industry_hint}")
    context_parts.append(f"本題追問：{query}")
    return "\n".join(context_parts)


def build_graph_query(query: str, category: str, memory: ConversationMemory) -> str:
    remembered = []
    if memory.last_query:
        remembered.append(f"對話記憶：上一題是「{memory.last_query}」。")
    if memory.last_industry_hint:
        remembered.append(f"已知適用對象：{memory.last_industry_hint}。")
    if memory.last_measurements:
        remembered.append(f"已知檢測資料：{memory.last_measurements}。")
    remembered.append(build_category_query(query, category))
    return "\n".join(part for part in remembered if part)


def should_use_category_summary(query: str, category: str, measurements: Any) -> bool:
    if category == "auto" or measurements:
        return False
    if category not in {"industry_scope", "facility_special", "sewer_system"}:
        return False
    return any(keyword in query for keyword in SUMMARY_KEYWORDS)


def build_category_summary_payload(*, conversation_id: str, category: str) -> dict[str, Any]:
    answer, citations = build_category_summary(category)
    payload = empty_payload(conversation_id=conversation_id, category=category)
    payload.update({
        "final_answer": answer,
        "citations": citations,
        "cited_articles": [citation["title"] for citation in citations],
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
            "used_category_summary": True,
        },
    })
    return payload


def build_category_summary(category: str) -> tuple[str, list[dict[str, Any]]]:
    if category == "industry_scope":
        answer = (
            "【工廠/事業類別主要法規】\n"
            "這個面向主要看《放流水標準》第2條的事業類別附表，也就是附表一到附表八。\n\n"
            "主要類別包括：\n"
            "1. 附表一：晶圓製造及半導體製造業。\n"
            "2. 附表二：光電材料及元件製造業。\n"
            "3. 附表三：石油化學業。\n"
            "4. 附表四：化工業。\n"
            "5. 附表五：金屬基本工業、金屬表面處理業、電鍍業、印刷電路板製造業。\n"
            "6. 附表六：發電廠。\n"
            "7. 附表七：海水淡化廠。\n"
            "8. 附表八：上述類別以外之其他事業。\n\n"
            "如果你要判斷某個工廠的 COD、BOD、SS、pH 等是否合規，需要再提供事業類別與檢測值。"
        )
        return answer, category_summary_citations([
            ("放流水標準第2條附表一", "晶圓製造及半導體製造業放流水水質項目及限值"),
            ("放流水標準第2條附表二", "光電材料及元件製造業放流水水質項目及限值"),
            ("放流水標準第2條附表三", "石油化學業放流水水質項目及限值"),
            ("放流水標準第2條附表四", "化工業放流水水質項目及限值"),
            ("放流水標準第2條附表五", "金屬基本工業、金屬表面處理業、電鍍業和印刷電路板製造業放流水水質項目及限值"),
            ("放流水標準第2條附表六", "發電廠放流水水質項目及限值"),
            ("放流水標準第2條附表七", "海水淡化廠放流水水質項目及限值"),
            ("放流水標準第2條附表八", "其他事業放流水水質項目及限值"),
        ])

    if category == "facility_special":
        answer = (
            "【建築物/特殊類別主要法規】\n"
            "這個面向主要看《放流水標準》第2條附表十五與附表十六。\n\n"
            "主要類別包括：\n"
            "1. 附表十五：建築物污水處理設施放流水水質項目及限值。\n"
            "2. 附表十六：總量管制區、特殊保護水體或其他特定條件下的放流水限值。\n\n"
            "如果問題是一般建築物污水處理設施，優先看附表十五；如果涉及總量管制區或特殊保護水體，再確認附表十六的適用條件。"
        )
        return answer, category_summary_citations([
            ("放流水標準第2條附表十五", "建築物污水處理設施放流水水質項目及限值"),
            ("放流水標準第2條附表十六", "總量管制區及特殊保護水體相關放流水限值"),
        ])

    if category == "sewer_system":
        answer = (
            "【污水下水道系統主要法規】\n"
            "這個面向主要看《放流水標準》第2條的污水下水道系統附表，也就是附表九到附表十四。\n\n"
            "主要類別包括：\n"
            "1. 附表九：科學工業園區專用污水下水道系統。\n"
            "2. 附表十：石油化學專業區專用污水下水道系統。\n"
            "3. 附表十一：其他工業區專用污水下水道系統。\n"
            "4. 附表十二：社區專用污水下水道系統。\n"
            "5. 附表十三：其他指定地區或場所專用污水下水道系統。\n"
            "6. 附表十四：公共污水下水道系統。\n\n"
            "若要判斷排放限值，需要先確認是哪一種下水道系統，再提供污染物與檢測值。"
        )
        return answer, category_summary_citations([
            ("放流水標準第2條附表九", "科學工業園區專用污水下水道系統放流水水質項目及限值"),
            ("放流水標準第2條附表十", "石油化學專業區專用污水下水道系統放流水水質項目及限值"),
            ("放流水標準第2條附表十一", "其他工業區專用污水下水道系統放流水水質項目及限值"),
            ("放流水標準第2條附表十二", "社區專用污水下水道系統放流水水質項目及限值"),
            ("放流水標準第2條附表十三", "其他指定地區或場所專用污水下水道系統放流水水質項目及限值"),
            ("放流水標準第2條附表十四", "公共污水下水道系統放流水水質項目及限值"),
        ])

    return "", []


def category_summary_citations(items: list[tuple[str, str]]) -> list[dict[str, Any]]:
    standards = load_standards()
    citations = []
    for article, description in items:
        source_file = ""
        for standard in standards:
            if standard.get("source_article") == article:
                source_file = standard.get("source_file", "")
                break
        citations.append({
            "title": article,
            "text": description,
            "source_file": source_file,
            "type": "category_summary",
        })
    return citations


def build_missing_data_hint(
    *,
    query: str,
    category: str,
    measurements: Any,
    industry_hint: str | None,
) -> str:
    if category not in {"auto", "effluent_standard", "industry_scope", "sewer_system", "facility_special"}:
        return ""
    if not measurements:
        return ""
    if not any(keyword in query for keyword in COMPLIANCE_KEYWORDS):
        return ""
    if industry_hint:
        return ""
    return (
        "目前已有檢測數值，但還缺少適用對象，無法可靠判斷限值。"
        "請補充事業或系統類別，例如化工業、晶圓製造及半導體製造業、電鍍業、"
        "工業區專用污水下水道系統、社區專用污水下水道系統等。"
    )


def filter_standards_for_category(category: str) -> list[dict[str, Any]]:
    standards = load_standards()
    if category == "sewer_system":
        return [item for item in standards if item.get("scope_type") == "sewer_system"]
    if category == "industry_scope":
        return [item for item in standards if item.get("scope_type") == "industry"]
    if category == "facility_special":
        return [
            item for item in standards
            if "建築物" in str(item.get("industry", ""))
            or "附表十六" in str(item.get("source_name", ""))
            or "總量管制" in str(item.get("source_name", ""))
        ]
    return standards


def build_structured_judgment(
    *,
    measurements: Any,
    industry_hint: str | None,
    category: str,
) -> dict[str, Any] | None:
    if not measurements:
        return None

    evaluations = evaluate_records(
        measurements,
        industry_hint=industry_hint,
        standards=filter_standards_for_category(category),
    )
    if not evaluations:
        return None

    items = [format_evaluation(item) for item in evaluations]
    statuses = {item["status"] for item in items}
    if "failed" in statuses:
        overall = "failed"
        is_compliant = False
    elif statuses == {"passed"}:
        overall = "passed"
        is_compliant = True
    elif "missing_conditions" in statuses:
        overall = "missing_data"
        is_compliant = None
    else:
        overall = "no_standard"
        is_compliant = None

    return {
        "overall_status": overall,
        "is_compliant": is_compliant,
        "industry_hint": industry_hint,
        "items": items,
    }


def format_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    record = evaluation.get("record") or {}
    standard = evaluation.get("standard") or {}
    comparison = evaluation.get("comparison") or {}
    status = evaluation.get("status")
    limit_value = standard.get("limit_value")
    min_value = standard.get("min_value")
    max_value = standard.get("max_value")

    return {
        "status": status,
        "is_compliant": comparison.get("passed") if comparison else None,
        "pollutant": record.get("pollutant"),
        "measured_value": record.get("value"),
        "unit": record.get("unit") or standard.get("unit") or "",
        "limit_type": standard.get("limit_type"),
        "limit_value": limit_value,
        "min_value": min_value,
        "max_value": max_value,
        "limit_unit": standard.get("unit") or "",
        "industry": standard.get("industry"),
        "scope_type": standard.get("scope_type"),
        "source_name": standard.get("source_name"),
        "source_article": standard.get("source_article"),
        "source_file": standard.get("source_file"),
        "comparison": comparison.get("detail"),
        "missing_conditions": [
            condition.get("label") or condition.get("field")
            for condition in evaluation.get("missing_conditions", [])
        ],
        "reason": build_item_reason(record, standard, comparison, status),
    }


def build_item_reason(
    record: dict[str, Any],
    standard: dict[str, Any],
    comparison: dict[str, Any],
    status: str,
) -> str:
    pollutant = record.get("pollutant", "污染物")
    value = record.get("value")
    unit = record.get("unit") or standard.get("unit") or ""
    if status == "no_standard":
        return f"{pollutant} {value} {unit} 尚未找到可直接比對的結構化限值。".strip()
    if status == "missing_conditions":
        return f"{pollutant} 已找到可能標準，但仍缺少適用條件。"
    if standard.get("limit_type") == "range":
        limit = f"{standard.get('min_value')}~{standard.get('max_value')} {standard.get('unit', '')}".strip()
    else:
        limit = f"{standard.get('limit_value')} {standard.get('unit', '')}".strip()
    result = "符合" if comparison.get("passed") else "超過或不符合"
    return f"{pollutant} 檢測值 {value:g} {unit}，標準限值 {limit}，比對式 {comparison.get('detail')}，判斷為{result}。"


def should_prefer_structured_answer(structured_judgment: dict[str, Any]) -> bool:
    return structured_judgment.get("overall_status") in {"passed", "failed", "missing_data"}


def build_structured_answer(structured_judgment: dict[str, Any]) -> str:
    status = structured_judgment.get("overall_status")
    if status == "passed":
        headline = "【合規判斷結果】\n符合目前找到的結構化放流水標準。"
    elif status == "failed":
        headline = "【合規判斷結果】\n不符合目前找到的結構化放流水標準。"
    elif status == "missing_data":
        headline = "【合規判斷結果】\n需要補充適用條件後才能完成判斷。"
    else:
        headline = "【合規判斷結果】\n尚未找到可直接比對的結構化標準。"

    lines = [headline, "", "【判斷說明】"]
    for item in structured_judgment.get("items", []):
        lines.append(f"- {item.get('reason')}")

    cited = [
        item.get("source_article") or item.get("source_name")
        for item in structured_judgment.get("items", [])
        if item.get("source_article") or item.get("source_name")
    ]
    if cited:
        lines.extend(["", "【引用條號】", "、".join(dict.fromkeys(cited))])
    return "\n".join(lines)


def compact_state(
    state: dict[str, Any],
    *,
    conversation_id: str,
    category: str,
    structured_judgment: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "category": category,
        "final_answer": state.get("final_answer") or "",
        "judgment_result": state.get("judgment_result"),
        "structured_judgment": structured_judgment,
        "cited_articles": state.get("cited_articles") or [],
        "citations": [],
        "waiting_for_data_input": bool(state.get("waiting_for_data_input")),
        "data_request_hint": state.get("data_request_hint"),
        "error": state.get("error"),
        "state": {
            "has_law_search_results": bool(state.get("law_search_results")),
            "has_structured_judgment": bool(structured_judgment),
        },
    }


def empty_payload(*, conversation_id: str, category: str) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "category": category,
        "final_answer": "",
        "judgment_result": None,
        "structured_judgment": None,
        "cited_articles": [],
        "citations": [],
        "waiting_for_data_input": False,
        "data_request_hint": None,
        "error": None,
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
        },
    }


def merge_citations(
    *,
    structured_judgment: dict[str, Any] | None,
    state: dict[str, Any],
    category: str,
    query: str,
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    if structured_judgment:
        for item in structured_judgment.get("items", []):
            title = item.get("source_article") or item.get("source_name")
            if not title:
                continue
            citations.append({
                "title": title,
                "text": item.get("reason") or "",
                "law_name": item.get("source_name"),
                "article": item.get("source_article"),
                "source_file": item.get("source_file"),
                "type": "structured_standard",
            })

    for item in state.get("law_search_results") or []:
        title = " ".join(
            part for part in [
                str(item.get("law_name", "")).strip(),
                str(item.get("article", "")).strip(),
            ] if part
        )
        citations.append({
            "title": title or str(item.get("source_file", "")),
            "text": str(item.get("text", ""))[:480],
            "law_name": item.get("law_name"),
            "article": item.get("article"),
            "source_file": item.get("source_file"),
            "score": item.get("score"),
            "type": "rag_chunk",
        })

    citations.extend(retrieve_category_citations(query, category))
    return dedupe_citations(citations)[:6]


def retrieve_category_citations(query: str, category: str) -> list[dict[str, Any]]:
    law_filters = CATEGORY_LAW_FILTERS.get(category, [])
    if not law_filters:
        return []
    try:
        from rag import LawRAG

        rag = LawRAG()
        results = []
        for law_filter in law_filters:
            results.extend(rag.retrieve(query=query, n_results=2, law_filter=law_filter))
    except Exception:
        return []

    return [
        {
            "title": " ".join(
                part for part in [
                    str(item.get("law_name", "")).strip(),
                    str(item.get("article", "")).strip(),
                ] if part
            ),
            "text": str(item.get("text", ""))[:480],
            "law_name": item.get("law_name"),
            "article": item.get("article"),
            "source_file": item.get("source_file"),
            "score": item.get("score"),
            "type": "category_filtered_rag",
        }
        for item in results
    ]


def dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for citation in citations:
        key = (
            citation.get("title"),
            citation.get("source_file"),
            citation.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(citation)
    return output


def remember(
    memory: ConversationMemory,
    query: str,
    answer: str,
    measurements: Any,
    industry_hint: str | None,
    category: str,
) -> None:
    memory.messages.extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ])
    memory.messages = memory.messages[-12:]
    if measurements:
        memory.last_measurements = measurements
    if industry_hint:
        memory.last_industry_hint = industry_hint
    memory.last_category = category
    memory.last_query = query
