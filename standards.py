"""
結構化放流水標準查詢與通用比較器。

設計原則：
    1. 法規限值放在 standards.json，不寫死在流程程式。
    2. 比較器只處理 max/min/range/equals 等通用比較型態。
    3. 新增法規時新增資料列；只有新增全新比較語意時才改程式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pollutant_catalog import canonical_pollutant


STANDARD_PATH = Path(__file__).with_name("standards.json")

METAL_SURFACE_INDUSTRY = "金屬基本工業、金屬表面處理業、電鍍業和印刷電路板製造業"
SEMICONDUCTOR_INDUSTRY = "晶圓製造及半導體製造業"
CHEMICAL_INDUSTRY = "化工業"
OTHER_INDUSTRY_MARKER = "以外之事業"
OTHER_INDUSTRY_HINTS = ("其他事業", "以外之事業", "其他產業", "其他行業")
INDUSTRIES_EXCLUDED_FROM_OTHER_TABLE = (
    "晶圓製造及半導體製造業",
    "光電材料及元件製造業",
    "石油化學業",
    "化工業",
    METAL_SURFACE_INDUSTRY,
    "金屬基本工業",
    "金屬表面處理業",
    "電鍍業",
    "印刷電路板製造業",
    "發電廠",
    "海水淡化廠",
)

SUPPLEMENTAL_TABLE_LIMITS = (
    (
        SEMICONDUCTOR_INDUSTRY,
        "附表一晶圓製造及半導體製造業放流水水質項目及限值",
        "放流水標準第2條附表一",
        "附表一晶圓製造及半導體製造業放流水水質項目及限值.pdf",
        (("BOD", 30.0), ("COD", 100.0), ("SS", 30.0)),
    ),
    (
        CHEMICAL_INDUSTRY,
        "附表四化工業放流水水質項目及限值",
        "放流水標準第2條附表四",
        "附表四化工業放流水水質項目及限值.pdf",
        (("BOD", 30.0), ("COD", 100.0), ("SS", 30.0)),
    ),
    (
        METAL_SURFACE_INDUSTRY,
        "附表五金屬基本工業、金屬表面處理業、電鍍業和印刷電路板製造業放流水水質項目及限值",
        "放流水標準第2條附表五",
        "附表五金屬基本工業、金屬表面處理業、電鍍業和印刷電路板製造業放流水水質項目及限值.pdf",
        (
            ("BOD", 30.0),
            ("COD", 100.0),
            ("SS", 30.0),
            ("銅", 3.0),
            ("鎳", 1.0),
            ("鋅", 5.0),
            ("總鉻", 2.0),
            ("六價鉻", 0.5),
        ),
    ),
)

SUPPLEMENTAL_STANDARDS = [
    {
        "id": f"supplemental_{source_article}_{pollutant}",
        "domain": "wastewater",
        "industry": industry,
        "pollutant": pollutant,
        "aliases": [],
        "requires_industry": True,
        "source_name": source_name,
        "source_article": source_article,
        "source_file": source_file,
        "notes": "補齊專用附表抽表漏列的通用限值，供結構化標準查詢與合規比對共用。",
        "scope_type": "industry",
        "conditions": [],
        "limit_type": "max",
        "limit_value": limit,
        "unit": "mg/L",
    }
    for industry, source_name, source_article, source_file, limits in SUPPLEMENTAL_TABLE_LIMITS
    for pollutant, limit in limits
]


def load_standards(path: Path = STANDARD_PATH) -> List[Dict[str, Any]]:
    """載入結構化法規限值資料。"""
    with path.open("r", encoding="utf-8") as f:
        return with_supplemental_standards(json.load(f))


def with_supplemental_standards(standards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """補齊已知抽表漏列，但仍保持法規來源 metadata 的結構化標準。"""
    result = list(standards)
    existing = {
        (
            standard.get("industry"),
            standard.get("pollutant"),
            standard.get("source_article"),
            standard.get("limit_type"),
        )
        for standard in result
    }
    for standard in SUPPLEMENTAL_STANDARDS:
        key = (
            standard.get("industry"),
            standard.get("pollutant"),
            standard.get("source_article"),
            standard.get("limit_type"),
        )
        if key not in existing:
            result.append(dict(standard))
            existing.add(key)
    return result


def normalize_industry_key(value: str) -> str:
    return (
        value.replace("和", "、")
        .replace("及", "、")
        .replace("/", "、")
        .replace(" ", "")
    )


def is_other_industry_standard(standard: Dict[str, Any]) -> bool:
    text = f"{standard.get('industry', '')} {standard.get('source_name', '')}"
    return OTHER_INDUSTRY_MARKER in text


def is_other_industry_hint(industry_hint: Optional[str]) -> bool:
    if not industry_hint:
        return False
    return any(hint in industry_hint for hint in OTHER_INDUSTRY_HINTS)


def is_excluded_other_industry_match(industry_hint: Optional[str], standard: Dict[str, Any]) -> bool:
    """附表八列出的是被排除的專用產業；這些字樣不能當作精準命中。"""
    if not industry_hint or not is_other_industry_standard(standard) or is_other_industry_hint(industry_hint):
        return False
    normalized_hint = normalize_industry_key(industry_hint)
    return any(
        normalized_hint == normalize_industry_key(industry)
        or normalized_hint in normalize_industry_key(industry)
        or normalize_industry_key(industry) in normalized_hint
        for industry in INDUSTRIES_EXCLUDED_FROM_OTHER_TABLE
    )


def normalize_pollutant(value: str) -> str:
    """統一污染物名稱格式，便於別名比對。"""
    return canonical_pollutant(value).upper()


def iter_data_records(data_result: Any) -> Iterable[Dict[str, Any]]:
    """將單筆或多筆 data_input_result 統一轉為 iterable。"""
    if isinstance(data_result, list):
        for item in data_result:
            if isinstance(item, dict):
                yield item
    elif isinstance(data_result, dict):
        yield data_result


def standard_matches(record: Dict[str, Any],
                     standard: Dict[str, Any],
                     industry_hint: Optional[str] = None) -> bool:
    """判斷一筆使用者數值是否可套用某筆標準。"""
    pollutant = normalize_pollutant(str(record.get("pollutant", "")))
    standard_names = [standard.get("pollutant", ""), *standard.get("aliases", [])]
    normalized_names = {normalize_pollutant(str(name)) for name in standard_names}

    if pollutant not in normalized_names:
        return False

    if industry_hint:
        if is_excluded_other_industry_match(industry_hint, standard):
            return False
        return industry_hint in str(standard.get("industry", ""))

    if standard.get("requires_industry", False):
        return False

    return True


def missing_required_conditions(standard: Dict[str, Any],
                                context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """回傳尚未由使用者或系統提供的必要條件。"""
    context = context or {}
    missing = []
    for condition in standard.get("conditions", []):
        field = condition.get("field")
        if condition.get("required", True) and field and field not in context:
            missing.append(condition)
    return missing


def conditions_match(standard: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> bool:
    """檢查已提供的條件是否符合標準適用範圍。"""
    context = context or {}
    for condition in standard.get("conditions", []):
        field = condition.get("field")
        if not field or field not in context:
            continue

        actual = context[field]
        expected = condition.get("value")
        operator = condition.get("operator", "equals")

        match operator:
            case "equals":
                if actual != expected:
                    return False
            case "in":
                if actual not in condition.get("values", []):
                    return False
            case "gte":
                if float(actual) < float(expected):
                    return False
            case "lte":
                if float(actual) > float(expected):
                    return False
            case _:
                raise ValueError(f"不支援的 condition operator: {operator}")

    return True


def find_standard(record: Dict[str, Any],
                  industry_hint: Optional[str] = None,
                  context: Optional[Dict[str, Any]] = None,
                  standards: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """依污染物與可選產業提示查找最適合的標準。"""
    standards = standards if standards is not None else load_standards()
    fallback = None

    for standard in standards:
        if not standard_matches(record, standard, industry_hint=industry_hint):
            continue
        if not conditions_match(standard, context=context):
            continue
        if industry_hint and industry_hint in str(standard.get("industry", "")):
            return standard
        if fallback is None:
            fallback = standard

    return fallback


def matching_standards(record: Dict[str, Any],
                       industry_hint: Optional[str] = None,
                       standards: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    standards = standards if standards is not None else load_standards()
    return [
        standard for standard in standards
        if standard_matches(record, standard, industry_hint=industry_hint)
    ]


def compare_value(value: float, standard: Dict[str, Any]) -> Dict[str, Any]:
    """
    通用比較器。

    分支超過 3 種時使用 match-case，避免長串 if-else。
    """
    limit_type = standard.get("limit_type")

    match limit_type:
        case "max":
            limit = float(standard["limit_value"])
            passed = value <= limit
            detail = f"{value:g} <= {limit:g}"
        case "min":
            limit = float(standard["limit_value"])
            passed = value >= limit
            detail = f"{value:g} >= {limit:g}"
        case "range":
            min_value = float(standard["min_value"])
            max_value = float(standard["max_value"])
            passed = min_value <= value <= max_value
            detail = f"{min_value:g} <= {value:g} <= {max_value:g}"
        case "equals":
            limit = float(standard["limit_value"])
            passed = value == limit
            detail = f"{value:g} == {limit:g}"
        case _:
            raise ValueError(f"不支援的 limit_type: {limit_type}")

    return {
        "passed": passed,
        "detail": detail,
        "limit_type": limit_type,
    }


def comparison_determinable_without_conditions(record: Dict[str, Any],
                                               candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """When every conditional candidate leads to the same outcome, no follow-up is needed."""
    if len(candidates) < 2:
        return None
    value = float(record["value"])
    if all(candidate.get("limit_type") == "max" for candidate in candidates):
        limit_values = [float(candidate["limit_value"]) for candidate in candidates if candidate.get("limit_value") is not None]
        if len(limit_values) != len(candidates):
            return None
        if value > max(limit_values):
            standard = max(candidates, key=lambda item: float(item["limit_value"]))
            return {"standard": standard, "comparison": compare_value(value, standard), "status": "failed"}
        if value <= min(limit_values):
            standard = min(candidates, key=lambda item: float(item["limit_value"]))
            return {"standard": standard, "comparison": compare_value(value, standard), "status": "passed"}
    return None


def evaluate_records(data_result: Any,
                     industry_hint: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None,
                     standards: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """查找每筆使用者數值的標準並回傳比較結果。"""
    results = []
    standards = standards if standards is not None else load_standards()

    for record in iter_data_records(data_result):
        if "pollutant" not in record or "value" not in record:
            continue

        standard = find_standard(
            record,
            industry_hint=industry_hint,
            context=context,
            standards=standards,
        )
        if not standard:
            results.append({
                "record": record,
                "standard": None,
                "comparison": None,
                "status": "no_standard",
            })
            continue

        missing_conditions = missing_required_conditions(standard, context=context)
        if missing_conditions:
            deterministic = comparison_determinable_without_conditions(
                record,
                matching_standards(record, industry_hint=industry_hint, standards=standards),
            )
            if deterministic:
                results.append({
                    "record": record,
                    "standard": deterministic["standard"],
                    "comparison": deterministic["comparison"],
                    "status": deterministic["status"],
                })
                continue
            results.append({
                "record": record,
                "standard": standard,
                "comparison": None,
                "status": "missing_conditions",
                "missing_conditions": missing_conditions,
            })
            continue

        comparison = compare_value(float(record["value"]), standard)
        results.append({
            "record": record,
            "standard": standard,
            "comparison": comparison,
            "status": "passed" if comparison["passed"] else "failed",
        })

    return results
