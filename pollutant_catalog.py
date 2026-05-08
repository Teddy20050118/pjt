"""
污染物、單位與產業分類 catalog。

這個模組負責可泛化的文字感知：
    - 只接受 catalog 內的污染物名稱與別名
    - 單位採白名單，避免把「合規嗎」誤當單位
    - 產業提示也由 catalog 管理，新增產業時不改流程程式
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CATALOG_PATH = Path(__file__).with_name("pollutants.json")


def load_catalog(path: Path = CATALOG_PATH) -> Dict[str, Any]:
    """載入污染物與產業 catalog。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_pollutants(catalog: Optional[Dict[str, Any]] = None) -> Iterable[Dict[str, Any]]:
    catalog = catalog if catalog is not None else load_catalog()
    return catalog.get("pollutants", [])


def _alias_entries(catalog: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """展開污染物別名，長別名優先，避免短字串先吃掉長字串。"""
    entries = []
    for pollutant in _iter_pollutants(catalog):
        for alias in pollutant.get("aliases", []):
            entries.append({
                "alias": alias,
                "canonical": pollutant["canonical"],
                "default_unit": pollutant.get("default_unit", ""),
                "allowed_units": pollutant.get("allowed_units", []),
            })
    return sorted(entries, key=lambda item: len(item["alias"]), reverse=True)


def canonical_pollutant(name: str, catalog: Optional[Dict[str, Any]] = None) -> str:
    """將污染物名稱或別名正規化為 canonical 名稱。"""
    target = name.strip().upper()
    for pollutant in _iter_pollutants(catalog):
        names = [pollutant["canonical"], *pollutant.get("aliases", [])]
        if target in {str(item).strip().upper() for item in names}:
            return pollutant["canonical"]
    return name.strip().upper()


def aliases_for_pollutant(name: str, catalog: Optional[Dict[str, Any]] = None) -> List[str]:
    """取得污染物 canonical 名稱與所有別名。"""
    canonical = canonical_pollutant(name, catalog)
    for pollutant in _iter_pollutants(catalog):
        if pollutant["canonical"] == canonical:
            return list(dict.fromkeys([pollutant["canonical"], *pollutant.get("aliases", [])]))
    return [name]


def retrieval_terms(catalog: Optional[Dict[str, Any]] = None) -> List[str]:
    """取得 RAG 查詢強化用的通用詞。"""
    catalog = catalog if catalog is not None else load_catalog()
    return catalog.get("retrieval_terms", [])


def infer_industry_hint(text: str, catalog: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """從使用者文字推斷產業提示。"""
    catalog = catalog if catalog is not None else load_catalog()
    for industry in catalog.get("industries", []):
        if any(alias in text for alias in industry.get("aliases", [])):
            return industry["canonical"]
    return None


def parse_measurements(text: str, catalog: Optional[Dict[str, Any]] = None) -> Optional[dict | list]:
    """
    從文字解析污染物檢測值。

    泛化策略：
        1. 先找 catalog 中的污染物別名
        2. 在污染物名稱後方近距離尋找數值與白名單單位
        3. 若找不到，再支援「120 mg/L COD」這類反向寫法
    """
    if not text:
        return None

    catalog = catalog if catalog is not None else load_catalog()
    results = []
    seen = set()

    for entry in _alias_entries(catalog):
        alias = entry["alias"]
        for match in re.finditer(_alias_regex(alias), text, flags=re.IGNORECASE):
            parsed = _parse_value_after_alias(text, match.end(), entry)
            if parsed is None:
                parsed = _parse_value_before_alias(text, match.start(), entry)
            if parsed is None:
                continue

            key = (parsed["pollutant"], parsed["value"], parsed["unit"])
            if key in seen:
                continue
            seen.add(key)
            results.append(parsed)

    if not results:
        return None
    return results[0] if len(results) == 1 else results


def _alias_regex(alias: str) -> str:
    """
    建立別名匹配 regex。

    英文/數字別名必須使用字詞邊界，避免 Co 誤命中 COD。
    中文別名不使用 \\b，因為中文詞邊界在 Python regex 中不穩定。
    """
    escaped = re.escape(alias)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\\-]*", alias):
        return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return escaped


def _parse_value_after_alias(text: str, start: int, entry: Dict[str, Any]) -> Optional[dict]:
    """解析「COD 測到 120 mg/L」這類污染物在前的寫法。"""
    window = text[start:start + 32]
    match = re.search(
        r"(?:\s|[:：=為是到測得測到濃度約大約達]){0,12}"
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*([A-Za-z/%／\u4e00-\u9fff]*)",
        window,
    )
    if not match:
        return None
    return _build_record(entry, match.group(1), match.group(2))


def _parse_value_before_alias(text: str, alias_start: int, entry: Dict[str, Any]) -> Optional[dict]:
    """解析「120 mg/L COD」這類數值在前的寫法。"""
    window = text[max(0, alias_start - 32):alias_start]
    matches = list(re.finditer(
        r"([0-9]+(?:\.[0-9]+)?)"
        r"\s*([A-Za-z/%／\u4e00-\u9fff]*)"
        r"(?:\s|[:：=為是到測得測到濃度約大約達]){0,12}$",
        window,
    ))
    if not matches:
        return None
    match = matches[-1]
    return _build_record(entry, match.group(1), match.group(2))


def _build_record(entry: Dict[str, Any], value: str, unit: str) -> Optional[dict]:
    unit = normalize_unit(unit, entry)
    if unit is None:
        return None
    return {
        "pollutant": entry["canonical"],
        "value": float(value),
        "unit": unit,
    }


def normalize_unit(raw_unit: str, entry: Dict[str, Any]) -> Optional[str]:
    """白名單化單位；不允許的文字視為沒有單位或解析失敗。"""
    unit = raw_unit.strip()
    allowed_units = entry.get("allowed_units", [])
    default_unit = entry.get("default_unit", "")

    if unit in allowed_units:
        return _canonical_unit(unit)

    if not unit:
        return default_unit

    # pH 等無單位項目若後方接中文問句，不把問句當單位。
    if allowed_units == [""]:
        return default_unit

    normalized = unit.replace("／", "/")
    for allowed in allowed_units:
        if normalized.lower() == str(allowed).replace("／", "/").lower():
            return _canonical_unit(allowed)

    return None


def _canonical_unit(unit: str) -> str:
    normalized = unit.replace("／", "/")
    match normalized.lower():
        case "mg/l" | "毫克/公升":
            return "mg/L"
        case _:
            return normalized
