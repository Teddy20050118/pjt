from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from law_query_categories import build_category_query, normalize_category
from pollutant_catalog import canonical_pollutant, detect_pollutants, infer_industry_hint, parse_measurements
from standards import evaluate_records, is_excluded_other_industry_match, is_other_industry_standard, load_standards


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

LIMIT_QUERY_KEYWORDS = (
    "限值",
    "標準",
    "標準值",
    "範圍",
    "是多少",
    "多少",
    "為何",
    "應介於",
    "介於",
    "上限",
    "下限",
    "管制值",
    "容許值",
    "規定",
    "排放標準",
    "水質項目及限值",
)

ITEM_LIST_QUERY_KEYWORDS = (
    "哪些水質項目",
    "水質項目",
    "要看哪些",
    "要檢測哪些",
    "檢測哪些",
    "常見項目",
    "含重金屬",
    "重金屬廢水",
)

SCOPE_QUERY_KEYWORDS = (
    "適用",
    "應該看哪",
    "看哪一個附表",
    "哪個附表",
    "哪個放流水標準",
    "哪一個放流水標準",
    "屬於",
    "不屬於",
    "不同",
    "差別",
)

SEWER_VS_INDUSTRY_KEYWORDS = (
    "排入園區污水下水道",
    "排放到園區污水下水道",
    "排入污水下水道",
    "專用污水下水道系統",
    "事業放流水標準還是",
    "和一般",
    "有什麼不同",
    "差在哪",
)

WASTE_MANAGEMENT_KEYWORDS = (
    "廢溶劑",
    "廢油",
    "事業廢棄物",
    "有害事業廢棄物",
    "清除處理機構",
    "合法清運",
    "合法清除",
    "清除處理",
    "清運",
    "委託",
    "少量",
    "偶爾產生",
    "自行處理",
    "貯存",
    "暫存",
)

HAZARDOUS_WASTE_CLASSIFICATION_KEYWORDS = (
    "是不是有害",
    "是否有害",
    "算有害",
    "屬於有害",
    "判定有害",
    "認定為有害",
)

PERMIT_PLAN_KEYWORDS = (
    "水污染防治許可",
    "排放許可",
    "許可證",
    "許可文件",
    "水污染防治措施計畫",
    "水措計畫",
    "新增廢水處理設備",
    "排放水量增加",
    "變更水污染防治許可",
    "變更許可",
)

MONITORING_REPORTING_KEYWORDS = (
    "操作紀錄",
    "保存",
    "記錄哪些",
    "檢測申報",
    "水質檢測申報",
    "申報通常",
    "沒有依規定做檢測申報",
    "未申報",
    "紀錄保存",
)

PENALTY_VIOLATION_KEYWORDS = (
    "裁罰",
    "罰鍰",
    "超過放流水標準",
    "未取得許可",
    "繞流",
    "暗管",
    "偷排",
    "沒有正常操作",
    "尚未超標",
    "裁罰準則",
    "違規嚴重",
)

COMPLIANCE_CHECKLIST_KEYWORDS = (
    "合規檢查清單",
    "檢查清單",
)

ANTI_HALLUCINATION_KEYWORDS = (
    "不要只給結論",
    "依據的法規名稱",
    "附表名稱",
    "行業別不明確",
    "先問我",
    "不要直接套用標準",
    "請區分",
    "事業放流水標準",
    "專用污水下水道系統放流水標準",
    "資料庫沒有",
    "查無資料",
    "不要自行推測",
)

BARE_ARTICLE_PATTERN = re.compile(r"^\s*第\s*([0-9一二三四五六七八九十百]+)\s*條\s*$")
ARTICLE_PATTERN = re.compile(r"第\s*([0-9一二三四五六七八九十百]+)\s*條")
BARE_TABLE_PATTERN = re.compile(r"^\s*附表\s*([0-9一二三四五六七八九十百]+)\s*$")
TABLE_PATTERN = re.compile(r"附表\s*([0-9一二三四五六七八九十百]+)")

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

KNOWN_UNSTRUCTURED_POLLUTANTS = (
    "甲苯",
)


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
    effective_query, is_followup = build_effective_query(clean_query, memory)
    parsed_measurements = parse_measurements(effective_query)
    measurements = parsed_measurements or (memory.last_measurements if is_followup else None)
    industry_hint = infer_industry_hint(effective_query) or (memory.last_industry_hint if is_followup else None)
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

    waste_payload = build_waste_management_payload(
        query=effective_query,
        conversation_id=memory.conversation_id,
        category=normalized_category,
    )
    if waste_payload:
        remember(
            memory,
            clean_query,
            waste_payload.get("final_answer", ""),
            measurements,
            industry_hint,
            "waste_management",
        )
        return waste_payload

    water_obligation_payload = build_water_obligation_payload(
        query=effective_query,
        conversation_id=memory.conversation_id,
        category=normalized_category,
    )
    if water_obligation_payload:
        remember(
            memory,
            clean_query,
            water_obligation_payload.get("final_answer", ""),
            measurements,
            industry_hint,
            water_obligation_payload.get("category", normalized_category),
        )
        return water_obligation_payload

    anti_hallucination_payload = build_anti_hallucination_payload(
        query=effective_query,
        conversation_id=memory.conversation_id,
        category=normalized_category,
    )
    if anti_hallucination_payload:
        remember(
            memory,
            clean_query,
            anti_hallucination_payload.get("final_answer", ""),
            measurements,
            industry_hint,
            anti_hallucination_payload.get("category", normalized_category),
        )
        return anti_hallucination_payload

    scope_payload = build_scope_query_payload(
        query=effective_query,
        conversation_id=memory.conversation_id,
        category=normalized_category,
        industry_hint=industry_hint,
    )
    if scope_payload:
        remember(
            memory,
            clean_query,
            scope_payload.get("final_answer", ""),
            measurements,
            industry_hint,
            normalized_category,
        )
        return scope_payload

    item_list_payload = build_item_list_query_payload(
        query=effective_query,
        conversation_id=memory.conversation_id,
        category=normalized_category,
        industry_hint=industry_hint,
    )
    if item_list_payload:
        remember(
            memory,
            clean_query,
            item_list_payload.get("final_answer", ""),
            measurements,
            industry_hint,
            normalized_category,
        )
        return item_list_payload

    multi_industry_limit_payload = build_multi_industry_limit_table_payload(
        query=effective_query,
        conversation_id=memory.conversation_id,
        category=normalized_category,
    )
    if multi_industry_limit_payload:
        remember(
            memory,
            clean_query,
            multi_industry_limit_payload.get("final_answer", ""),
            measurements,
            industry_hint,
            normalized_category,
        )
        return multi_industry_limit_payload

    limit_query_payload = build_limit_query_payload(
        query=effective_query,
        conversation_id=memory.conversation_id,
        category=normalized_category,
        measurements=parsed_measurements,
        industry_hint=industry_hint,
    )
    if limit_query_payload:
        remember(
            memory,
            clean_query,
            limit_query_payload.get("final_answer", ""),
            measurements,
            industry_hint,
            normalized_category,
        )
        return limit_query_payload

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

    graph_query = build_graph_query(effective_query, normalized_category, memory, include_memory=is_followup)
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


def build_effective_query(query: str, memory: ConversationMemory) -> tuple[str, bool]:
    if not memory.last_query:
        return query, False
    has_new_industry = infer_industry_hint(query) is not None
    has_new_measurements = parse_measurements(query) is not None
    has_new_pollutants = bool(detect_pollutants(query))
    has_new_waste_topic = is_waste_management_query(query)
    is_explicit_followup = any(keyword in query for keyword in FOLLOWUP_KEYWORDS)
    is_short_contextual_followup = (
        len(query) <= 16
        and not has_new_industry
        and not has_new_measurements
        and not has_new_pollutants
        and not has_new_waste_topic
    )
    is_followup = (is_explicit_followup and not has_new_waste_topic) or is_short_contextual_followup
    if not is_followup:
        return query, False

    context_parts = [f"前一題問題：{memory.last_query}"]
    if memory.last_measurements:
        context_parts.append(f"前一題檢測資料：{memory.last_measurements}")
    if memory.last_industry_hint:
        context_parts.append(f"前一題適用對象：{memory.last_industry_hint}")
    context_parts.append(f"本題追問：{query}")
    return "\n".join(context_parts), True


def build_graph_query(query: str, category: str, memory: ConversationMemory, *, include_memory: bool = True) -> str:
    remembered = []
    if include_memory and memory.last_query:
        remembered.append(f"對話記憶：上一題是「{memory.last_query}」。")
    if include_memory and memory.last_industry_hint:
        remembered.append(f"已知適用對象：{memory.last_industry_hint}。")
    if include_memory and memory.last_measurements:
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


def build_waste_management_payload(
    *,
    query: str,
    conversation_id: str,
    category: str,
) -> dict[str, Any] | None:
    if category not in {"auto", "waste_management"} and not is_waste_management_query(query):
        return None
    if category == "waste_management" or is_waste_management_query(query):
        payload = empty_payload(conversation_id=conversation_id, category="waste_management")
        if is_hazardous_waste_classification_query(query):
            answer = build_waste_classification_hint()
        else:
            answer = build_waste_obligation_answer()
        citations = waste_management_citations()
        payload.update({
            "final_answer": answer,
            "citations": citations,
            "cited_articles": [citation["title"] for citation in citations],
            "state": {
                "has_law_search_results": False,
                "has_structured_judgment": False,
                "used_waste_management": True,
            },
        })
        return payload
    return None


def is_waste_management_query(query: str) -> bool:
    return any(keyword in query for keyword in WASTE_MANAGEMENT_KEYWORDS)


def is_hazardous_waste_classification_query(query: str) -> bool:
    return any(keyword in query for keyword in HAZARDOUS_WASTE_CLASSIFICATION_KEYWORDS)


def build_waste_obligation_answer() -> str:
    return (
        "通常仍需依法處理。只要廢溶劑屬於事業廢棄物，尤其可能具易燃性、毒性或揮發性的有害事業廢棄物，"
        "即使只是偶爾產生、數量很少，也不代表可以自行倒掉、焚燒、混入一般垃圾或交給無照業者。\n\n"
        "實務上應先分類收集並妥善暫存，容器要密封、標示內容物與貯存資訊，避免洩漏、揮發或火災風險。"
        "後續通常應委託合法清除處理機構清運處理，或依合法再利用方式辦理，並保存清除處理聯單或相關紀錄。\n\n"
        "少量產生可能影響申報、貯存或管理強度，但不是免除合法清運與妥善處理義務的理由。"
    )


def build_waste_classification_hint() -> str:
    return (
        "是否屬於有害事業廢棄物，需要看廢溶劑的成分與有害特性，不能只靠「量少」判斷。"
        "請補充 SDS、安全資料表、主要成分、閃火點、是否含毒性或腐蝕性物質，以及可能的廢棄物代碼。\n\n"
        "在尚未確認前，建議先按較嚴格方式分類、密封標示、獨立暫存，並洽合法清除處理機構或主管機關確認。"
    )


def waste_management_citations() -> list[dict[str, Any]]:
    return [
        {
            "title": "廢棄物清理法第28條",
            "text": "事業廢棄物之清理，應依中央主管機關規定方式辦理，包含自行、共同、委託或其他合法清理方式。",
            "law_name": "廢棄物清理法",
            "article": "第28條",
            "source_file": "廢棄物清理法",
            "type": "structured_waste_management",
        },
        {
            "title": "廢棄物清理法第30條",
            "text": "事業委託清除、處理其事業廢棄物，仍應注意受託者資格與清理流向管理。",
            "law_name": "廢棄物清理法",
            "article": "第30條",
            "source_file": "廢棄物清理法",
            "type": "structured_waste_management",
        },
        {
            "title": "事業廢棄物貯存清除處理方法及設施標準",
            "text": "事業廢棄物應依規定分類、貯存、清除及處理，避免污染環境或危害安全。",
            "law_name": "事業廢棄物貯存清除處理方法及設施標準",
            "article": "",
            "source_file": "事業廢棄物貯存清除處理方法及設施標準",
            "type": "structured_waste_management",
        },
        {
            "title": "公民營廢棄物清除處理機構許可管理辦法",
            "text": "廢棄物清除處理機構應取得相應許可後，始得從事清除、處理業務。",
            "law_name": "公民營廢棄物清除處理機構許可管理辦法",
            "article": "",
            "source_file": "公民營廢棄物清除處理機構許可管理辦法",
            "type": "structured_waste_management",
        },
    ]


def build_water_obligation_payload(
    *,
    query: str,
    conversation_id: str,
    category: str,
) -> dict[str, Any] | None:
    route = classify_water_obligation_query(query, category)
    if not route:
        return None

    if route == "permit_plan":
        answer = build_permit_plan_answer(query)
        citations = water_permit_citations()
        response_category = "permit_plan"
    elif route == "monitoring_reporting":
        answer = build_monitoring_reporting_answer(query)
        citations = monitoring_reporting_citations()
        response_category = "monitoring_reporting"
    elif route == "penalty_violation":
        answer = build_penalty_violation_answer(query)
        citations = penalty_violation_citations()
        response_category = "penalty"
    else:
        answer = build_water_compliance_checklist_answer()
        citations = water_permit_citations() + monitoring_reporting_citations() + penalty_violation_citations()
        response_category = "auto"

    payload = empty_payload(conversation_id=conversation_id, category=response_category)
    payload.update({
        "final_answer": answer,
        "citations": dedupe_citations(citations),
        "cited_articles": [citation["title"] for citation in dedupe_citations(citations)],
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
            "route": route,
        },
    })
    return payload


def classify_water_obligation_query(query: str, category: str) -> str | None:
    if any(keyword in query for keyword in COMPLIANCE_CHECKLIST_KEYWORDS) and "水污染" in query:
        return "water_compliance_checklist"
    if category == "permit_plan" or any(keyword in query for keyword in PERMIT_PLAN_KEYWORDS):
        return "permit_plan"
    if category == "monitoring_reporting" or any(keyword in query for keyword in MONITORING_REPORTING_KEYWORDS):
        return "monitoring_reporting"
    if category == "penalty" or any(keyword in query for keyword in PENALTY_VIOLATION_KEYWORDS):
        return "penalty_violation"
    return None


def build_permit_plan_answer(query: str) -> str:
    if "差別" in query or "有什麼差別" in query:
        return (
            "水污染防治措施計畫和排放許可證是不同階段的管理文件。\n\n"
            "| 項目 | 水污染防治措施計畫 | 排放許可證/許可文件 |\n"
            "|---|---|---|\n"
            "| 核心用途 | 說明廢水如何收集、處理、回用或排放 | 核准實際排放條件 |\n"
            "| 時點 | 設置、變更或建置處理設施前 | 正式排放廢（污）水前 |\n"
            "| 重點 | 製程、用水量、廢水來源、處理流程、設計處理量、應變措施 | 放流口、排放水量、水質、檢測資料、排放方式 |\n\n"
            "簡化理解是：水措計畫先審處理系統設計，排放許可證再確認實際排放是否可被允許。"
        )
    if "新增" in query or "變更" in query or "排放水量增加" in query:
        return (
            "可能需要辦理水污染防治措施計畫或排放許可證（文件）變更。判斷重點不是設備名稱本身，"
            "而是是否影響原核准內容。\n\n"
            "通常需要檢討變更的情況包括：新增或改變廢水處理單元、改變處理流程、增加處理量、"
            "新增製程廢水來源、排放水量增加、改變放流口、改變排放水質或藥劑操作條件。\n\n"
            "如果變更後的實際系統與原許可文件不一致，即使是為了改善污染，也應先確認是否需申請變更。"
        )
    return (
        "通常需要。列管事業或污水下水道系統在排放廢（污）水前，通常應先取得水污染防治相關核准文件，"
        "例如水污染防治措施計畫、排放許可證或簡易排放許可文件。\n\n"
        "實務上要先確認：是否屬列管事業、廢水來源與製程、排放去向、排放水量、處理流程、放流口位置，"
        "以及是否排入專用污水下水道或委託處理。少量排放不當然免除許可或核准義務。\n\n"
        "可把流程理解為：先審查廢水處理與管理方式，再核准是否可以實際排放。"
    )


def build_monitoring_reporting_answer(query: str) -> str:
    if "檢測申報" in query or "申報" in query:
        if "沒有依規定" in query or "未申報" in query:
            return (
                "沒有依規定辦理檢測、申報或紀錄保存，可能違反水污染防治相關檢測申報與管理義務，"
                "並可能面臨罰鍰、限期改善、按次處分或其他行政處分。\n\n"
                "常見違規包括：未定期檢測、逾期申報、申報資料不完整、不實申報、未保存紀錄、採樣或檢測程序不符規定，"
                "或未委託合格檢測機構。核心風險是主管機關無法追蹤排放狀態與處理設施運作紀錄。"
            )
        return (
            "水質檢測申報通常不只是填一個檢測值，而是要能描述採樣當時的排放狀態。\n\n"
            "常見資料包括：基本資料、採樣日期與時間、採樣點、放流水量或操作流量、檢測項目、檢測結果、"
            "檢測方法、檢測單位、處理設施操作狀況、異常紀錄、申報日期與負責人資料。\n\n"
            "檢測項目會依事業類別、許可文件與適用放流水標準不同而不同，例如 pH、COD、BOD、SS、氨氮與重金屬等。"
        )
    return (
        "需要。廢水處理設施操作紀錄通常應定期製作並保存，以供主管機關查核。\n\n"
        "常見紀錄內容包括：操作時間、進流水量與放流水量、水質數據、加藥種類與用量、設備運轉狀態、"
        "污泥產生與清運、異常或故障、停機、維修保養與緊急應變處理。\n\n"
        "這類紀錄的目的，是確認處理設施是否持續正常操作，而不是只看某一次採樣是否合格。"
    )


def build_penalty_violation_answer(query: str) -> str:
    if "未取得許可" in query:
        return (
            "未取得水污染防治相關許可或核准文件就排放廢（污）水，本身即可能構成重大違規；"
            "即使單次水質未必超標，也代表排放行為未納入許可管理。\n\n"
            "可能處分包括罰鍰、限期改善、命停止排放、按次或按日連續處罰；情節重大時，可能涉及停工、停業、廢止許可或刑事風險。"
        )
    if "繞流" in query or "暗管" in query or "偷排" in query:
        return (
            "繞流排放、暗管或偷排通常比一般水質超標更嚴重，因為它代表排放行為可能刻意避開處理設施或監測管理。\n\n"
            "主管機關通常會看是否故意、是否規避許可條件、污染程度、影響水體、是否重複違規與是否立即改善。"
            "這類情節可能導致較高罰鍰、連續處罰、停工停業、廢止許可或刑事風險。"
        )
    if "沒有正常操作" in query or "尚未超標" in query:
        return (
            "仍可能違規。水污染法規不只看最後放流水檢測值，也要求廢水處理設施依核准內容與正常操作條件運轉。\n\n"
            "即使當次檢測值尚未超標，若處理設施未正常操作、未依許可文件操作、未記錄或故障未妥善處理，仍可能違反管理義務。"
        )
    if "裁罰準則" in query or "考量哪些因素" in query:
        return (
            "裁罰通常會考量違規類型、污染物項目、超標程度、排放量、違規次數、是否故意或過失、是否繞流或不實申報、"
            "污染影響範圍、改善配合程度及是否屬累犯等因素。\n\n"
            "因此同樣是超標，若伴隨暗管、繞流、偽造資料或多次違規，風險通常會比單純偶發超標更高。"
        )
    return (
        "事業排放廢水超過放流水標準，通常可能依水污染防治法受到罰鍰、限期改善、複查及按次或按日連續處罰。\n\n"
        "主管機關通常會依採樣檢測確認超標項目與程度，再要求改善；若持續超標、故意繞流、偷排或不實申報，"
        "可能提高裁罰風險，並可能涉及停工、停業、廢止許可或刑事責任。"
    )


def build_water_compliance_checklist_answer() -> str:
    return (
        "水污染法規合規檢查清單可先用下表盤點：\n\n"
        "| 面向 | 檢查重點 | 常見風險 |\n"
        "|---|---|---|\n"
        "| 許可文件 | 是否有水措計畫、排放許可證或簡易許可文件；實際製程、水量、放流口是否一致 | 未許可排放、未辦變更 |\n"
        "| 操作紀錄 | 是否記錄操作時間、水量、水質、加藥、設備運轉、污泥、異常與維修 | 設施未正常操作、紀錄不足 |\n"
        "| 檢測申報 | 是否依期程採樣、檢測、申報；採樣點、檢測方法、檢測單位是否正確 | 逾期申報、不實申報、漏測 |\n"
        "| 放流水標準 | 是否依事業別或下水道系統附表確認 pH、COD、SS、氨氮、重金屬等限值 | 超標、引用錯誤附表 |\n"
        "| 異常應變 | 故障、停電、藥劑不足或水質異常時是否通報、貯存或停止排放 | 異常仍排放、繞流排放 |\n"
        "| 裁罰風險 | 是否有重複違規、繞流、暗管、未許可、不實資料 | 罰鍰、限期改善、連續處罰、停工停業 |\n\n"
        "實務上應以許可文件、主管機關核定內容與實際排放去向為主，並定期比對現場與文件是否一致。"
    )


def water_permit_citations() -> list[dict[str, Any]]:
    return [
        {
            "title": "水污染防治法",
            "text": "事業或污水下水道系統排放廢（污）水前，應依水污染防治相關規定辦理許可或核准事項。",
            "law_name": "水污染防治法",
            "article": "",
            "source_file": "水污染防治法",
            "type": "structured_water_obligation",
        },
        {
            "title": "水污染防治措施計畫及許可申請審查管理辦法",
            "text": "水污染防治措施計畫、排放許可證與變更申請之審查管理依本辦法辦理。",
            "law_name": "水污染防治措施計畫及許可申請審查管理辦法",
            "article": "",
            "source_file": "水污染防治措施計畫及許可申請審查管理辦法",
            "type": "structured_water_obligation",
        },
    ]


def monitoring_reporting_citations() -> list[dict[str, Any]]:
    return [
        {
            "title": "水污染防治措施及檢測申報管理辦法",
            "text": "事業或污水下水道系統之檢測申報、紀錄保存、操作維護與異常管理依相關管理辦法辦理。",
            "law_name": "水污染防治措施及檢測申報管理辦法",
            "article": "",
            "source_file": "水污染防治措施及檢測申報管理辦法",
            "type": "structured_water_obligation",
        }
    ]


def penalty_violation_citations() -> list[dict[str, Any]]:
    return [
        {
            "title": "水污染防治法",
            "text": "違反水污染防治法之排放、許可、申報或管理義務，可能依規定裁處罰鍰、限期改善或其他處分。",
            "law_name": "水污染防治法",
            "article": "",
            "source_file": "水污染防治法",
            "type": "structured_water_obligation",
        },
        {
            "title": "違反水污染防治法罰鍰額度裁罰準則",
            "text": "裁罰額度會依違規類型、污染程度、違規次數、改善情形等因素裁量。",
            "law_name": "違反水污染防治法罰鍰額度裁罰準則",
            "article": "",
            "source_file": "違反水污染防治法罰鍰額度裁罰準則",
            "type": "structured_water_obligation",
        },
    ]


def build_anti_hallucination_payload(
    *,
    query: str,
    conversation_id: str,
    category: str,
) -> dict[str, Any] | None:
    route = classify_anti_hallucination_query(query)
    if not route:
        return None

    if route == "anti_reference_requirements":
        answer = (
            "可以。若要列出依據，我會先確認適用的事業或污水下水道系統類別、污染物項目與必要條件，"
            "再列法規名稱、附表名稱、污染物限值、單位與條件，不只給結論。\n\n"
            "標準型回答會包含：適用對象、法規來源、附表名稱、項目、限值、單位，以及是否有保護區、流量級距、新設/既設或排放去向等條件。"
            "如果資料庫中沒有該污染物或該業別的結構化限值，會明確說明查無資料，不自行推測數值。"
        )
        citations = effluent_standard_guardrail_citations()
    elif route == "missing_industry_guardrail":
        answer = (
            "如果行業別或排放情境不明確，不能直接套用某一個放流水附表。需要先補充：\n\n"
            "| 需補充資訊 | 用途 |\n"
            "|---|---|\n"
            "| 事業或系統類別 | 判斷是半導體、光電、石化、化工、金屬、發電、海淡、其他事業或污水下水道系統 |\n"
            "| 製程或廢水來源 | 區分專用附表、以外之事業或特殊設施 |\n"
            "| 排放去向 | 判斷是直接排放、排入專用污水下水道或其他處理方式 |\n"
            "| 污染物項目與檢測值 | 查限值或做合規判斷時使用 |\n"
            "| 其他條件 | 保護區、流量級距、新設/既設、建照日期等條件可能影響限值 |\n\n"
            "在資訊不足時，正確做法是要求補充條件或列出可能適用路徑，而不是直接套用單一標準。"
        )
        citations = effluent_standard_guardrail_citations()
    elif route == "effluent_vs_sewer_standard":
        answer = (
            "「事業放流水標準」和「專用污水下水道系統放流水標準」要分開看，核心差異是排放主體與排放去向。\n\n"
            "| 類型 | 適用重點 | 常見附表 |\n"
            "|---|---|---|\n"
            "| 事業放流水標準 | 事業本身直接排放廢水到承受水體時，依其業別或設施類型查限值 | 半導體、光電、石化、化工、金屬、發電、海淡、其他事業等附表 |\n"
            "| 專用污水下水道系統放流水標準 | 園區、工業區、社區或公共污水下水道系統處理後，由系統排放時查限值 | 科學工業園區、石化專業區、其他工業區、社區、公共污水下水道系統等附表 |\n\n"
            "因此，若廠商排入園區或工業區專用污水下水道，通常還要確認納管規定；若是系統最終排放，則看該污水下水道系統的放流水標準。"
            "不能把事業直接排放標準和下水道系統放流水標準混用。"
        )
        citations = effluent_standard_guardrail_citations()
    else:
        answer = (
            "如果本機結構化資料庫沒有某個污染物、業別或附表的限值，應明確回覆「目前資料庫查無該污染物限值」，"
            "不自行推測，也不使用相近污染物、相近行業或一般常識補數值。\n\n"
            "可再補充污染物別名、化學式、事業別、排放去向或附表名稱後重新查詢；若仍查無，才回到法規全文檢索或請使用者確認是否有地方加嚴、納管標準或許可文件特別條件。"
        )
        citations = effluent_standard_guardrail_citations()

    payload = empty_payload(conversation_id=conversation_id, category=category)
    deduped = dedupe_citations(citations)
    payload.update({
        "final_answer": answer,
        "citations": deduped,
        "cited_articles": [citation["title"] for citation in deduped],
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
            "route": route,
        },
    })
    return payload


def classify_anti_hallucination_query(query: str) -> str | None:
    if not any(keyword in query for keyword in ANTI_HALLUCINATION_KEYWORDS):
        return None
    if "行業別不明確" in query or "不要直接套用標準" in query or "先問我" in query:
        return "missing_industry_guardrail"
    if "請區分" in query and "事業放流水標準" in query and "專用污水下水道" in query:
        return "effluent_vs_sewer_standard"
    if "資料庫沒有" in query or "查無資料" in query or "不要自行推測" in query:
        return "no_standard_guardrail"
    if "不要只給結論" in query or "依據的法規名稱" in query or "附表名稱" in query:
        return "anti_reference_requirements"
    return None


def effluent_standard_guardrail_citations() -> list[dict[str, Any]]:
    return [
        {
            "title": "放流水標準第2條",
            "text": "放流水標準依事業類別、污水下水道系統類別及附表所列水質項目與限值適用。",
            "law_name": "放流水標準",
            "article": "2",
            "source_file": "放流水標準",
            "type": "structured_guardrail",
        },
        {
            "title": "放流水標準第2條附表一至附表十四",
            "text": "事業與污水下水道系統應依其適用附表查詢污染物項目、限值、單位與條件。",
            "law_name": "放流水標準",
            "article": "2",
            "source_file": "放流水標準",
            "type": "structured_guardrail",
        },
    ]


def build_scope_query_payload(
    *,
    query: str,
    conversation_id: str,
    category: str,
    industry_hint: str | None,
) -> dict[str, Any] | None:
    if category not in {"auto", "industry_scope", "sewer_system", "facility_special"}:
        return None
    if not any(keyword in query for keyword in SCOPE_QUERY_KEYWORDS + SEWER_VS_INDUSTRY_KEYWORDS):
        return None

    answer = ""
    citation_items: list[tuple[str, str]] = []
    route = "scope_query"

    if is_sewer_vs_industry_query(query):
        route = "sewer_vs_industry_query"
        if "科學" in query and "園區" in query and "下水道" in query:
            answer = (
                "若廠商廢水是排入科學工業園區專用污水下水道系統，放流水端通常應先看"
                "《放流水標準》第2條附表九的科學工業園區專用污水下水道系統標準；"
                "事業本身仍需確認園區納管標準、許可文件與前處理要求。\n\n"
                "也就是說，不能只用半導體業附表一直接取代園區專用污水下水道系統標準。"
            )
            citation_items = [
                ("放流水標準第2條附表九", "科學工業園區專用污水下水道系統放流水水質項目及限值"),
                ("放流水標準第2條附表一", "晶圓製造及半導體製造業放流水水質項目及限值"),
            ]
        elif "石油化學專業區" in query:
            answer = (
                "石油化學專業區專用污水下水道系統與一般石油化學業不是同一個適用對象。"
                "前者是污水下水道系統放流水，主要看附表十；後者是石油化學業事業放流水，主要看附表三。\n\n"
                "實務判斷時要先確認排放主體與排放去向：事業直接排放看事業別附表，專用污水下水道系統排放看下水道系統附表。"
            )
            citation_items = [
                ("放流水標準第2條附表十", "石油化學專業區專用污水下水道系統放流水水質項目及限值"),
                ("放流水標準第2條附表三", "石油化學業放流水水質項目及限值"),
            ]
        elif "建築物污水處理設施" in query and "社區專用污水下水道" in query:
            answer = (
                "建築物污水處理設施與社區專用污水下水道系統是不同適用對象。"
                "建築物污水處理設施主要看附表十五；社區專用污水下水道系統主要看附表十二。\n\n"
                "兩者可能在 BOD、COD、SS、大腸桿菌群等項目與流量或設置時間條件上不同，不能混用同一附表。"
            )
            citation_items = [
                ("放流水標準第2條附表十五", "建築物污水處理設施放流水水質項目及限值"),
                ("放流水標準第2條附表十二", "社區專用污水下水道系統放流水水質項目及限值"),
            ]

    if not answer and "半導體封裝測試" in query:
        answer = (
            "半導體封裝測試廠不能只因名稱含有「半導體」就直接套用晶圓製造及半導體製造業附表一。"
            "應先確認實際製程是否屬晶圓製造或半導體製造；若只是封裝、測試或後段加工，還要確認是否另屬其他事業類別、"
            "是否位於園區並排入專用污水下水道，以及許可文件核定的適用標準。\n\n"
            "需要補充的資訊包括：主要製程、廢水來源、排放去向，以及是否納入園區或工業區污水下水道系統。"
        )
        citation_items = [
            ("放流水標準第2條附表一", "晶圓製造及半導體製造業放流水水質項目及限值"),
            ("放流水標準第2條附表八", "上述專用事業以外之其他事業放流水水質項目及限值"),
        ]
    elif not answer and ("酸洗" in query or "電鍍" in query or "金屬零件" in query):
        answer = (
            "金屬零件酸洗及電鍍加工，放流水標準通常應優先看《放流水標準》第2條附表五，"
            "也就是金屬基本工業、金屬表面處理業、電鍍業和印刷電路板製造業的放流水水質項目及限值。\n\n"
            "若實際排入工業區或其他專用污水下水道，仍需另確認納管標準與許可文件。"
        )
        citation_items = [
            ("放流水標準第2條附表五", "金屬基本工業、金屬表面處理業、電鍍業和印刷電路板製造業放流水水質項目及限值"),
        ]
    elif not answer and ("食品" in query or "一般製造業" in query or "一般食品工廠" in query) and "不屬於" in query:
        answer = (
            "若一般食品工廠或一般製造業確定不屬於半導體、光電、石化、化工、金屬、發電廠或海水淡化廠等專用附表類別，"
            "事業放流水通常應看《放流水標準》第2條附表八，也就是上述類別以外之事業。\n\n"
            "仍需確認實際製程、主管機關核定類別、排放去向與許可文件；若排入專用污水下水道，還要看納管或下水道系統相關標準。"
        )
        citation_items = [
            ("放流水標準第2條附表八", "上述專用事業以外之其他事業放流水水質項目及限值"),
        ]
    elif not answer and "公共污水下水道系統" in query and ("哪個附表" in query or "流量" in query or "250" in query or "二五" in query):
        flow_note = (
            "若問題涉及每日流量大於 250 立方公尺，仍是在附表十四內確認對應項目與限值；"
            if any(token in query for token in ("流量", "250", "二五")) else ""
        )
        answer = (
            "公共污水下水道系統的放流水標準主要看《放流水標準》第2條附表十四。"
            f"{flow_note}不能改用社區專用污水下水道系統或事業別附表。\n\n"
            "若要判斷特定污染物是否超標，還需要提供污染物項目與實測值。"
        )
        citation_items = [
            ("放流水標準第2條附表十四", "公共污水下水道系統放流水水質項目及限值"),
        ]
    elif not answer and "建築物污水處理設施" in query and ("98 年 1 月 1 日" in query or "九十八年一月一日" in query or "較早申請" in query):
        answer = (
            "會不同。建築物污水處理設施在附表十五中，會依建造執照申請時間區分適用情境，"
            "例如中華民國 98 年 1 月 1 日以後申請建造執照者，與 97 年 12 月 31 日以前申請建造執照者，"
            "可能適用不同項目限值或流量條件。\n\n"
            "實務上要再確認申請建造執照日期、每日流量及排放地點，才能逐項判斷 BOD、COD、SS、大腸桿菌群等限值。"
        )
        citation_items = [
            ("放流水標準第2條附表十五", "建築物污水處理設施放流水水質項目及限值"),
        ]

    if not answer:
        return None

    citations = category_summary_citations(citation_items)
    payload = empty_payload(conversation_id=conversation_id, category=category)
    payload.update({
        "final_answer": answer,
        "citations": citations,
        "cited_articles": [citation["title"] for citation in citations],
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
            "route": route,
        },
    })
    return payload


def is_sewer_vs_industry_query(query: str) -> bool:
    has_sewer = "下水道" in query or "專用污水下水道系統" in query
    has_comparison_or_discharge_context = any(
        keyword in query
        for keyword in ("排入", "排放到", "還是", "有什麼不同", "差在哪", "和一般", "區分")
    )
    return has_sewer and has_comparison_or_discharge_context


def build_item_list_query_payload(
    *,
    query: str,
    conversation_id: str,
    category: str,
    industry_hint: str | None,
) -> dict[str, Any] | None:
    if category not in {"auto", "effluent_standard", "industry_scope", "sewer_system", "facility_special"}:
        return None
    if not industry_hint or not any(keyword in query for keyword in ITEM_LIST_QUERY_KEYWORDS):
        return None
    if detect_pollutants(query) and any(keyword in query for keyword in LIMIT_QUERY_KEYWORDS):
        return None

    standards = [
        standard for standard in filter_standards_for_category(category)
        if industry_matches_standard(industry_hint, standard)
    ]
    if not standards:
        return None

    standards = dedupe_limit_standards(standards)
    pollutants = sorted({str(standard.get("pollutant")) for standard in standards if standard.get("pollutant")})
    if not pollutants:
        return None

    heavy_metals = [item for item in pollutants if item in {"鎘", "鉛", "總鉻", "六價鉻", "總汞", "甲基汞", "銅", "鋅", "銀", "鎳", "硒", "砷"}]
    common = [item for item in pollutants if item in {"pH", "BOD", "COD", "SS", "NH3-N", "油脂", "氰化物", "氟鹽"}]
    others = [item for item in pollutants if item not in set(heavy_metals + common)]

    lines = [
        f"{industry_hint}排放廢水時，應依《放流水標準》第2條相關附表確認水質項目及限值。",
        "",
    ]
    source_titles = sorted({
        str(standard.get("source_article") or standard.get("source_name"))
        for standard in standards
        if standard.get("source_article") or standard.get("source_name")
    })
    if source_titles:
        lines.append(f"主要依據：{'、'.join(source_titles)}。")
        lines.append("")
    if common:
        lines.append(f"常見基本項目：{'、'.join(common)}。")
    if heavy_metals:
        lines.append(f"重金屬相關項目：{'、'.join(heavy_metals)}。")
    if others:
        lines.append(f"其他已結構化項目：{'、'.join(others)}。")
    lines.extend([
        "",
        "實際仍應以許可文件、排放去向及主管機關核定的適用附表為準；若排入專用污水下水道，另需確認納管標準。",
    ])

    citations = citations_for_limit_standards(standards)
    payload = empty_payload(conversation_id=conversation_id, category=category)
    payload.update({
        "final_answer": "\n".join(lines),
        "citations": citations,
        "cited_articles": [citation["title"] for citation in citations],
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
            "route": "item_list_query",
        },
    })
    return payload


def build_multi_industry_limit_table_payload(
    *,
    query: str,
    conversation_id: str,
    category: str,
) -> dict[str, Any] | None:
    if category not in {"auto", "effluent_standard", "industry_scope"}:
        return None
    if "表格" not in query and "整理" not in query:
        return None

    industry_candidates = [
        ("半導體業", "晶圓製造及半導體製造業", ("半導體", "半導體業", "晶圓")),
        ("光電業", "光電材料及元件製造業", ("光電", "光電業")),
        ("石化業", "石油化學業", ("石化", "石化業", "石油化學")),
        ("化工業", "化工業", ("化工", "化工業")),
    ]
    selected_industries = [
        (label, canonical)
        for label, canonical, aliases in industry_candidates
        if any(alias in query for alias in aliases)
    ]
    pollutants = detect_pollutants(query)
    if len(selected_industries) < 2 or len(pollutants) < 2:
        return None

    lines = [
        "依目前結構化資料整理如下；條件型標準會在欄位內列出適用情境：",
        "",
        "| 行業 | " + " | ".join(pollutants) + " | 依據 |",
        "|---|" + "|".join("---" for _ in pollutants) + "|---|",
    ]
    citation_standards: list[dict[str, Any]] = []
    for label, industry in selected_industries:
        values = []
        row_sources = []
        for pollutant in pollutants:
            matches = find_limit_standards(pollutant=pollutant, industry_hint=industry, category=category)
            citation_standards.extend(matches)
            if not matches:
                values.append("查無結構化限值")
                continue
            unconditional = [standard for standard in matches if not standard.get("conditions")]
            selected = unconditional or matches
            if len(selected) == 1 and not selected[0].get("conditions"):
                values.append(format_limit_value(selected[0]))
            else:
                values.append("；".join(
                    f"{format_conditions(standard.get('conditions', []))}：{format_limit_value(standard)}"
                    for standard in selected[:4]
                ))
            row_sources.append(selected[0].get("source_article") or selected[0].get("source_name"))
        lines.append(f"| {label} | " + " | ".join(values) + f" | {'、'.join(dict.fromkeys(row_sources))} |")

    lines.append("")
    lines.append("若表中顯示查無結構化限值，表示目前資料庫未命中該污染物與行業組合，不自行推測數值。")
    citations = citations_for_limit_standards(citation_standards)
    payload = empty_payload(conversation_id=conversation_id, category=category)
    payload.update({
        "final_answer": "\n".join(lines),
        "citations": citations,
        "cited_articles": [citation["title"] for citation in citations],
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
            "route": "multi_industry_limit_table",
        },
    })
    return payload


def build_limit_query_payload(
    *,
    query: str,
    conversation_id: str,
    category: str,
    measurements: Any,
    industry_hint: str | None,
) -> dict[str, Any] | None:
    if not should_use_structured_limit_query(query, category, measurements):
        return None

    pollutants = detect_pollutants(query)
    for pollutant in KNOWN_UNSTRUCTURED_POLLUTANTS:
        if pollutant in query and pollutant not in pollutants:
            pollutants.append(pollutant)
    if not pollutants or not industry_hint:
        return None

    grouped_matches = []
    missing_pollutants = []
    for pollutant in pollutants:
        matches = find_limit_standards(
            pollutant=pollutant,
            industry_hint=industry_hint,
            category=category,
        )
        if matches:
            grouped_matches.append((pollutant, matches))
        else:
            missing_pollutants.append(pollutant)

    if not grouped_matches and not missing_pollutants:
        return None

    answer = build_limit_query_answer(
        grouped_standards=grouped_matches,
        industry_hint=industry_hint,
        missing_pollutants=missing_pollutants,
    )
    citations = citations_for_limit_standards([
        standard
        for _, matches in grouped_matches
        for standard in matches
    ])
    payload = empty_payload(conversation_id=conversation_id, category=category)
    payload.update({
        "final_answer": answer,
        "citations": citations,
        "cited_articles": [citation["title"] for citation in citations],
        "state": {
            "has_law_search_results": False,
            "has_structured_judgment": False,
            "used_structured_limit_query": True,
            "route": "limit_query",
        },
    })
    return payload


def should_use_structured_limit_query(query: str, category: str, measurements: Any) -> bool:
    if measurements:
        return False
    if category not in {"auto", "effluent_standard", "industry_scope", "sewer_system", "facility_special"}:
        return False
    return any(keyword in query for keyword in LIMIT_QUERY_KEYWORDS)


def find_limit_standards(
    *,
    pollutant: str,
    industry_hint: str,
    category: str,
) -> list[dict[str, Any]]:
    standards = filter_standards_for_category(category)
    candidates = [
        standard for standard in standards
        if pollutant_matches_standard(pollutant, standard)
        and industry_matches_standard(industry_hint, standard)
    ]
    if not candidates:
        return []

    scored = sorted(
        ((limit_standard_score(standard, industry_hint), index, standard) for index, standard in enumerate(candidates)),
        key=lambda item: (-item[0], item[1]),
    )
    best_score = scored[0][0]
    best = [standard for score, _, standard in scored if score == best_score]
    return dedupe_limit_standards(best)


def pollutant_matches_standard(pollutant: str, standard: dict[str, Any]) -> bool:
    target = canonical_pollutant(pollutant).upper()
    names = [standard.get("pollutant", ""), *standard.get("aliases", [])]
    return target in {canonical_pollutant(str(name)).upper() for name in names}


def industry_matches_standard(industry_hint: str, standard: dict[str, Any]) -> bool:
    if is_excluded_other_industry_match(industry_hint, standard):
        return False
    industry = str(standard.get("industry", ""))
    source = str(standard.get("source_name", ""))
    normalized_hint = normalize_industry_text(industry_hint)
    return (
        industry_hint in industry
        or industry_hint in source
        or normalized_hint in normalize_industry_text(industry)
        or normalized_hint in normalize_industry_text(source)
    )


def limit_standard_score(standard: dict[str, Any], industry_hint: str) -> int:
    industry = str(standard.get("industry", ""))
    source = str(standard.get("source_name", ""))
    article = str(standard.get("source_article", ""))
    normalized_industry = normalize_industry_text(industry)
    normalized_hint = normalize_industry_text(industry_hint)
    score = 0

    if industry == industry_hint:
        score += 120
    elif normalized_industry == normalized_hint:
        score += 115
    elif industry.startswith(industry_hint):
        score += 90
    elif normalized_industry.startswith(normalized_hint):
        score += 85
    elif industry_hint in industry:
        score += 60
    elif normalized_hint in normalized_industry:
        score += 55
    elif industry_hint in source:
        score += 40
    elif normalized_hint in normalize_industry_text(source):
        score += 35

    if "以外之事業" in industry or "以外之事業" in source:
        score += 25 if is_other_industry_standard(standard) and "以外之事業" in industry_hint else -200
    if "附表五" in article or "附表五" in source:
        score += 45
    if " / " in industry:
        score -= 10
    if article:
        score += 5
    if not standard.get("conditions"):
        score += 3
    return score


def normalize_industry_text(value: str) -> str:
    return (
        value.replace("和", "、")
        .replace("及", "、")
        .replace("/", "、")
        .replace(" ", "")
    )


def dedupe_limit_standards(standards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for standard in standards:
        key = (
            standard.get("source_article"),
            standard.get("source_name"),
            standard.get("pollutant"),
            standard.get("limit_type"),
            standard.get("limit_value"),
            standard.get("min_value"),
            standard.get("max_value"),
            standard.get("unit"),
            tuple((condition.get("field"), condition.get("label"), condition.get("value")) for condition in standard.get("conditions", [])),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(standard)
    return result


def build_limit_query_answer(
    *,
    grouped_standards: list[tuple[str, list[dict[str, Any]]]],
    industry_hint: str,
    missing_pollutants: list[str] | None = None,
) -> str:
    missing_pollutants = missing_pollutants or []
    if not grouped_standards:
        return (
            f"目前結構化資料庫查無 {industry_hint} 對 "
            f"{'、'.join(missing_pollutants)} 的放流水限值。"
            "不自行推測數值；建議再確認完整法規表、許可文件或主管機關核定內容。"
        )

    if len(grouped_standards) > 1:
        lines = [
            f"{industry_hint}的放流水限值如下：",
            "",
            "| 項目 | 限值 | 依據 |",
            "|---|---:|---|",
        ]
        extra_notes = []
        for pollutant, standards in grouped_standards:
            unconditional = [standard for standard in standards if not standard.get("conditions")]
            selected = unconditional or standards
            if len(selected) == 1 and not selected[0].get("conditions"):
                standard = selected[0]
                source = standard.get("source_article") or standard.get("source_name")
                lines.append(f"| {pollutant} | {format_limit_value(standard)} | {source} |")
                continue

            detail = "；".join(
                f"{format_conditions(standard.get('conditions', []))}：{format_limit_value(standard)}"
                for standard in selected[:8]
            )
            source = selected[0].get("source_article") or selected[0].get("source_name")
            lines.append(f"| {pollutant} | {detail} | {source} |")
            extra_notes.append(f"{pollutant} 的限值需依適用情境判斷。")

        lines.extend([
            "",
            "依據為《放流水標準》第2條相關附表。",
        ])
        if extra_notes:
            lines.extend(["", *extra_notes])
        controlled_notes = build_controlled_limit_notes(grouped_standards)
        if controlled_notes:
            lines.extend(["", *controlled_notes])
        if missing_pollutants:
            lines.extend([
                "",
                f"目前結構化資料庫查無 {industry_hint} 對 {'、'.join(missing_pollutants)} 的放流水限值；不自行推測數值。",
            ])
        return "\n".join(lines)

    pollutant, standards = grouped_standards[0]
    unconditional = [standard for standard in standards if not standard.get("conditions")]
    conditional = [standard for standard in standards if standard.get("conditions")]
    selected = unconditional or conditional

    if len(selected) == 1 and not conditional:
        standard = selected[0]
        answer = (
            f"{industry_hint}的放流水 {pollutant} 限值為 {format_limit_value(standard)}。\n\n"
            f"{format_unit_note(standard)}依據為《放流水標準》第2條相關附表"
            f"（{standard.get('source_article') or standard.get('source_name')}）。"
        )
        if missing_pollutants:
            answer += f"\n\n目前結構化資料庫查無 {industry_hint} 對 {'、'.join(missing_pollutants)} 的放流水限值；不自行推測數值。"
        return answer

    lines = [f"{industry_hint}的放流水 {pollutant} 限值依適用情境不同："]
    for standard in selected[:8]:
        condition_text = format_conditions(standard.get("conditions", []))
        source = standard.get("source_article") or standard.get("source_name")
        lines.append(f"- {condition_text}：{format_limit_value(standard)}（{source}）")
    controlled_notes = build_controlled_limit_notes(grouped_standards)
    if controlled_notes:
        lines.extend(["", *controlled_notes])
    if missing_pollutants:
        lines.extend(["", f"目前結構化資料庫查無 {industry_hint} 對 {'、'.join(missing_pollutants)} 的放流水限值；不自行推測數值。"])
    return "\n".join(lines)


def build_controlled_limit_notes(grouped_standards: list[tuple[str, list[dict[str, Any]]]]) -> list[str]:
    """Add law-context notes only when they are directly supported by structured standards."""
    all_standards = [standard for _, standards in grouped_standards for standard in standards]
    if not all_standards:
        return []

    sources = {str(standard.get("source_article") or standard.get("source_name") or "") for standard in all_standards}
    pollutants = {canonical_pollutant(pollutant) for pollutant, _ in grouped_standards}
    notes = []

    if any("附表五" in source for source in sources):
        if "總鉻" in pollutants and "六價鉻" not in pollutants:
            hex_chrome = find_limit_standards(
                pollutant="六價鉻",
                industry_hint="金屬基本工業、金屬表面處理業、電鍍業和印刷電路板製造業",
                category="effluent_standard",
            )
            if hex_chrome and any("附表五" in str(item.get("source_article") or item.get("source_name")) for item in hex_chrome):
                notes.append(f"另同一附表也對六價鉻訂有限值：{format_limit_value(hex_chrome[0])}。")
        notes.append("若廢水排入工業區專用污水下水道，或所在地有地方加嚴標準、總量管制要求，仍應另行確認納管標準或地方規範。")

    return notes


def format_limit_value(standard: dict[str, Any]) -> str:
    unit = str(standard.get("unit") or "").strip()
    is_coliform = standard.get("pollutant") == "大腸桿菌群"
    if standard.get("pollutant") == "大腸桿菌群" and unit == "mg/L":
        unit = "CFU/100mL"
        standard = {**standard}
        for key in ("limit_value", "min_value", "max_value"):
            if standard.get(key) is not None:
                standard[key] = float(standard[key]) * 1000
    suffix = f" {unit}" if unit else ""
    if standard.get("limit_type") == "range":
        keep_decimal = not unit
        min_value = format_limit_number(standard.get("min_value"), keep_decimal=keep_decimal)
        max_value = format_limit_number(standard.get("max_value"), keep_decimal=keep_decimal)
        return f"{min_value} 至 {max_value}{suffix}"
    if standard.get("limit_type") == "max":
        if is_coliform:
            return f"{float(standard.get('limit_value')):,.0f}{suffix}"
        return f"{format_limit_number(standard.get('limit_value'))}{suffix}"
    if standard.get("limit_type") == "min":
        return f"不得低於 {format_limit_number(standard.get('limit_value'))}{suffix}"
    if standard.get("limit_type") == "equals":
        return f"{format_limit_number(standard.get('limit_value'))}{suffix}"
    return "目前結構化資料未標示可直接呈現的限值"


def format_limit_number(value: Any, *, keep_decimal: bool = False) -> str:
    number = float(value)
    if keep_decimal and number.is_integer():
        return f"{number:.1f}"
    return f"{number:g}"


def format_unit_note(standard: dict[str, Any]) -> str:
    if str(standard.get("unit") or "").strip():
        return ""
    return "此項目為無單位指標。"


def format_conditions(conditions: list[dict[str, Any]]) -> str:
    if not conditions:
        return "一般適用條件"
    labels = [
        clean_condition_label(str(condition.get("label") or condition.get("field") or "特定條件"))
        for condition in conditions
    ]
    return "、".join(labels)


def clean_condition_label(label: str) -> str:
    compact = re.sub(r"\s+", "", label)
    if compact == "排放於自來水水質水量保護區內者":
        return "排放於自來水水質水量保護區內"
    if "排放於自來水水質水量保護區外者" in compact and "完成建造" in compact:
        return "保護區外，既設或101年12月12日前已完成建造、建造中或完成工程招標者"
    if "尚未完成工程招標者" in compact:
        return "保護區外，101年12月12日前尚未完成工程招標者"
    if compact.startswith("排放於自來水水質水量保護區外者"):
        return compact.replace("排放於自來水水質水量保護區外者", "保護區外")
    return compact


def citations_for_limit_standards(standards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for standard in standards:
        title = standard.get("source_article") or standard.get("source_name")
        if not title:
            continue
        citations.append({
            "title": title,
            "text": f"{standard.get('industry')} {standard.get('pollutant')}：{format_limit_value(standard)}",
            "law_name": standard.get("source_name"),
            "article": standard.get("source_article"),
            "source_file": standard.get("source_file"),
            "type": "structured_limit_query",
        })
    return dedupe_citations(citations)


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
            "route": "numeric_compliance" if structured_judgment else "rag",
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
        if citations:
            return dedupe_citations(citations)[:6]

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
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []

    for raw_citation in citations:
        citation = normalize_citation(raw_citation)
        canonical = canonicalize_citation(citation)
        key = canonical["key"]
        existing = groups.get(key)

        if existing is None:
            groups[key] = citation
            order.append(key)
            continue

        existing_meta = canonicalize_citation(existing)
        if canonical["specificity_score"] > existing_meta["specificity_score"]:
            citation = merge_citation_fields(primary=citation, secondary=existing)
            groups[key] = citation
        else:
            groups[key] = merge_citation_fields(primary=existing, secondary=citation)

    groups = absorb_bare_references(groups)
    return [groups[key] for key in order if key in groups]


def absorb_bare_references(
    groups: dict[tuple[str, str, str], dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    bare_keys = [
        key for key, citation in groups.items()
        if (
            canonicalize_citation(citation)["kind"] == "article"
            and is_bare_article_title(citation.get("title", ""))
        )
        or (
            canonicalize_citation(citation)["kind"] == "table"
            and is_bare_table_title(citation.get("title", ""))
        )
    ]

    for bare_key in bare_keys:
        bare = groups.get(bare_key)
        if not bare:
            continue
        bare_meta = canonicalize_citation(bare)
        matches = [
            key for key, candidate in groups.items()
            if key != bare_key
            and canonicalize_citation(candidate)["kind"] == bare_meta["kind"]
            and canonicalize_citation(candidate).get("article_no") == bare_meta.get("article_no")
            and canonicalize_citation(candidate).get("table_no") == bare_meta.get("table_no")
            and canonicalize_citation(candidate)["specificity_score"] > bare_meta["specificity_score"]
        ]
        if len(matches) == 1:
            groups[matches[0]] = merge_citation_fields(primary=groups[matches[0]], secondary=bare)
            groups.pop(bare_key, None)
        elif len(matches) > 1:
            groups.pop(bare_key, None)

    return groups


def canonicalize_citation(citation: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_citation(citation)
    title = normalized.get("title", "")
    law_name = normalized.get("law_name", "")
    article_no = extract_article_number(title) or extract_article_number(normalized.get("article", ""))
    table_no = extract_table_number(title) or extract_table_number(str(normalized.get("source_file") or ""))

    if article_no:
        has_law_name = bool(law_name) or not is_bare_article_title(title)
        key_law = law_name if law_name else ("__unknown_law__" if is_bare_article_title(title) else title_without_article(title))
        specificity = 30 if has_law_name else 10
        return {
            "kind": "article",
            "law_name": law_name,
            "article_no": article_no,
            "table_no": "",
            "display_title": title,
            "specificity_score": specificity,
            "key": ("article", key_law, article_no),
        }

    if table_no:
        is_bare = is_bare_table_title(title)
        specificity = 30 if not is_bare else 10
        return {
            "kind": "table",
            "law_name": law_name,
            "article_no": "",
            "table_no": table_no,
            "display_title": title,
            "specificity_score": specificity,
            "key": ("table", "", table_no),
        }

    if law_name or title:
        display = title or law_name
        return {
            "kind": "law",
            "law_name": law_name or display,
            "article_no": "",
            "table_no": "",
            "display_title": display,
            "specificity_score": 20,
            "key": ("law", law_name or display, ""),
        }

    return {
        "kind": "unknown",
        "law_name": "",
        "article_no": "",
        "table_no": "",
        "display_title": "",
        "specificity_score": 0,
        "key": ("unknown", str(citation), ""),
    }


def normalize_citation(citation: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(citation)
    title = normalize_title(str(normalized.get("title", "")))
    law_name = normalize_title(str(normalized.get("law_name", "")))
    article = normalize_title(str(normalized.get("article", "")))

    if is_bare_article_title(title) and law_name:
        title = " ".join(part for part in [law_name, title] if part)
    elif not title and law_name:
        title = " ".join(part for part in [law_name, article] if part)

    normalized["title"] = title
    if law_name:
        normalized["law_name"] = law_name
    if article:
        normalized["article"] = article
    return normalized


def merge_citation_fields(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for field in ("text", "source_file", "law_name", "article", "score", "type"):
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]
    return merged


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


def is_bare_article_title(title: str) -> bool:
    return bool(BARE_ARTICLE_PATTERN.match(normalize_title(title)))


def is_bare_table_title(title: str) -> bool:
    return bool(BARE_TABLE_PATTERN.match(normalize_title(title)))


def extract_article_number(title: str) -> str | None:
    match = ARTICLE_PATTERN.search(normalize_title(title))
    if not match:
        return None
    return match.group(1)


def extract_table_number(title: str) -> str | None:
    match = TABLE_PATTERN.search(normalize_title(title))
    if not match:
        return None
    return match.group(1)


def title_without_article(title: str) -> str:
    return ARTICLE_PATTERN.sub("", normalize_title(title)).strip()


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
