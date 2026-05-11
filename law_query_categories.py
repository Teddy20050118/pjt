QUERY_CATEGORY_CONTEXTS = {
    "auto": "",
    "effluent_standard": "本次查詢聚焦於放流水標準、水質項目限值、是否超標與合規判斷。請優先依放流水標準及相關附表回答。",
    "industry_scope": "本次查詢聚焦於事業或工廠適用類別，例如化工、半導體、光電、石化、金屬表面處理、電鍍、印刷電路板等行業別。",
    "sewer_system": "本次查詢聚焦於污水下水道系統，例如科學園區、工業區、石油化學專業區、社區或公共污水下水道系統。",
    "facility_special": "本次查詢聚焦於建築物污水處理設施、總量管制區、特殊保護水體或其他特殊適用條件。",
    "permit_plan": "本次查詢聚焦於水污染防治措施計畫、許可申請、變更、展延與審查程序。",
    "monitoring_reporting": "本次查詢聚焦於檢測、監測、申報、紀錄保存、操作維護與管理義務。",
    "penalty": "本次查詢聚焦於違反水污染防治法的罰則、裁罰額度、改善期限與處分依據。",
}


def normalize_category(category: str | None) -> str:
    if category in QUERY_CATEGORY_CONTEXTS:
        return str(category)
    return "auto"


def build_category_query(query: str, category: str) -> str:
    context = QUERY_CATEGORY_CONTEXTS.get(category, "")
    if not context:
        return query
    return f"{context}\n\n使用者問題：{query}"
