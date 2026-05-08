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


def load_standards(path: Path = STANDARD_PATH) -> List[Dict[str, Any]]:
    """載入結構化法規限值資料。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
