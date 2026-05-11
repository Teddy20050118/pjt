from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz


LAW_DIR = next(
    p for p in Path(".").iterdir()
    if p.is_dir() and any(child.suffix.lower() == ".pdf" for child in p.iterdir())
)
INVENTORY_PATH = Path("law_inventory.json")
STANDARDS_PATH = Path("standards.json")
POLLUTANTS_PATH = Path("pollutants.json")


NUMERAL_MAP = {
    "零": "0",
    "○": "0",
    "〇": "0",
    "ㄧ": "1",
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
}

CANONICAL_POLLUTANTS = {
    "氫離子濃度指數": ("pH", ["pH", "PH", "氫離子濃度指數"]),
    "化學需氧量": ("COD", ["COD", "化學需氧量", "化學需氧量(COD)", "化學需氧量（COD）"]),
    "生化需氧量": ("BOD", ["BOD", "BOD5", "生化需氧量", "生化需氧量(BOD)", "生化需氧量（BOD）"]),
    "懸浮固體": ("SS", ["SS", "懸浮固體", "懸浮固體物"]),
    "氨氮": ("NH3-N", ["NH3-N", "氨氮", "氨態氮"]),
    "總毒性有機物": ("TTO", ["TTO", "總毒性有機物"]),
    "2-甲氧基-1-丙醇": ("2-METHOXY-1-PROPANOL", ["2-甲氧基-1-丙醇", "2-Methoxy-1-propanol"]),
    "二甲基乙醯胺": ("DMAC", ["DMAC", "二甲基乙醯胺", "二甲基乙酰胺", "Dimethylacetamide"]),
    "鈷": ("COBALT", ["鈷", "Cobalt", "Co"]),
    "銻": ("ANTIMONY", ["銻", "Antimony", "Sb"]),
}

SOURCE_ARTICLES = {
    "附表一": "放流水標準第2條附表一",
    "附表二": "放流水標準第2條附表二",
    "附表三": "放流水標準第2條附表三",
    "附表四": "放流水標準第2條附表四",
    "附表五": "放流水標準第2條附表五",
    "附表六": "放流水標準第2條附表六",
    "附表七": "放流水標準第2條附表七",
    "附表八": "放流水標準第2條附表八",
    "附表九": "放流水標準第2條附表九",
    "附表十": "放流水標準第2條附表十",
    "附表十一": "放流水標準第2條附表十一",
    "附表十二": "放流水標準第2條附表十二",
    "附表十三": "放流水標準第2條附表十三",
    "附表十四": "放流水標準第2條附表十四",
    "附表十五": "放流水標準第2條附表十五",
    "附表十六": "放流水標準第2條附表十六",
}

SKIP_LIMIT_TEXT = (
    "攝氏",
    "水溫",
    "不得",
    "以下",
    "放流口",
)


def compact(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "")


def readable(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def slug(value: str) -> str:
    text = value.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = text.strip("_").lower()
    if text:
        return text[:60]
    codepoints = "_".join(f"{ord(ch):x}" for ch in value[:12])
    return f"u_{codepoints}"


def normalize_item(value: str) -> str:
    value = compact(value)
    value = re.sub(r"（.*?）", "", value)
    value = re.sub(r"\(.*?\)", "", value)
    return value


def canonical_pollutant(item: str) -> tuple[str, list[str]]:
    normalized = normalize_item(item)
    if normalized in CANONICAL_POLLUTANTS:
        return CANONICAL_POLLUTANTS[normalized]
    return normalized, [normalized, item]


def source_article(source_name: str) -> str:
    for prefix, article in sorted(SOURCE_ARTICLES.items(), key=lambda pair: len(pair[0]), reverse=True):
        if source_name.startswith(prefix):
            return article
    return "放流水標準附表"


def source_scope(source_name: str, inventory_scope: str) -> str:
    scope = source_name.replace("放流水水質項目及限值", "")
    scope = re.sub(r"^附表[一二三四五六七八九十]+", "", scope)
    return scope or source_name


def parse_number(value: str) -> float | None:
    value = compact(value).replace("．", ".").replace("。", ".")
    value = value.replace("＜", "").replace("<", "")
    value = value.replace("⎯", "-").replace("\uf0be", "-")
    if any(token in value for token in SKIP_LIMIT_TEXT):
        return None
    converted = "".join(NUMERAL_MAP.get(ch, ch) for ch in value)
    match = re.search(r"\d+(?:\.\d+)?", converted)
    if not match:
        return None
    return float(match.group(0))


def parse_limit(limit: str) -> dict[str, Any] | None:
    text = compact(limit)
    if not text:
        return None
    if any(token in text for token in SKIP_LIMIT_TEXT):
        return None

    range_text = text.replace("．", ".").replace("⎯", "-").replace("\uf0be", "-")
    range_text = "".join(NUMERAL_MAP.get(ch, ch) for ch in range_text)
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-~－]\s*(\d+(?:\.\d+)?)", range_text)
    if range_match:
        return {
            "limit_type": "range",
            "min_value": float(range_match.group(1)),
            "max_value": float(range_match.group(2)),
            "unit": "",
        }

    number = parse_number(text)
    if number is None:
        return None

    return {
        "limit_type": "max",
        "limit_value": number,
        "unit": "mg/L",
    }


def extract_tables(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with fitz.open(path) as doc:
        for page in doc:
            for table in page.find_tables().tables:
                rows.extend(table.extract())
    return [[readable(cell) for cell in row] for row in rows if any(readable(cell) for cell in row)]


def header_indexes(row: list[str]) -> tuple[int, int, int | None, int | None]:
    item_idx = next(i for i, cell in enumerate(row) if compact(cell) == "項目")
    limit_idx = next(i for i, cell in enumerate(row) if compact(cell) == "限值")
    note_idx = next((i for i, cell in enumerate(row) if compact(cell) == "備註"), None)
    scope_idx = next((i for i, cell in enumerate(row) if compact(cell) == "適用範圍"), None)
    return item_idx, limit_idx, note_idx, scope_idx


def generate_standards(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    standards: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for item in inventory:
        if item.get("type") != "standard_attachment":
            continue

        path = LAW_DIR / item["file"]
        rows = extract_tables(path)
        current_item = ""
        current_scope = ""
        item_idx = limit_idx = 0
        note_idx: int | None = None
        scope_idx: int | None = None

        for row in rows:
            try:
                if any(compact(cell) == "項目" for cell in row) and any(compact(cell) == "限值" for cell in row):
                    item_idx, limit_idx, note_idx, scope_idx = header_indexes(row)
                    current_item = ""
                    current_scope = ""
                    continue
            except StopIteration:
                continue

            if limit_idx >= len(row) or item_idx >= len(row):
                continue

            if scope_idx is not None and scope_idx < len(row) and row[scope_idx]:
                current_scope = readable(row[scope_idx])

            raw_item = readable(row[item_idx])
            if raw_item:
                current_item = raw_item
            if not current_item:
                continue

            limit = readable(row[limit_idx])
            parsed_limit = parse_limit(limit)
            if not parsed_limit:
                continue

            canonical, aliases = canonical_pollutant(current_item)
            condition_cells = []
            for idx, cell in enumerate(row):
                if idx in {item_idx, limit_idx}:
                    continue
                if note_idx is not None and idx == note_idx:
                    continue
                if scope_idx is not None and idx == scope_idx:
                    continue
                if readable(cell):
                    condition_cells.append(readable(cell))

            conditions = []
            condition_label = "；".join(dict.fromkeys(condition_cells))
            if condition_label:
                conditions.append({
                    "field": "applicability_context",
                    "label": condition_label,
                    "operator": "equals",
                    "value": condition_label,
                    "required": True,
                })

            note = readable(row[note_idx]) if note_idx is not None and note_idx < len(row) else ""
            scope = source_scope(item["name"], item.get("scope", ""))
            if current_scope and current_scope != "共同適用":
                scope = f"{scope} / {current_scope}"

            key = (
                item["file"],
                scope,
                canonical,
                parsed_limit.get("limit_type"),
                parsed_limit.get("limit_value"),
                parsed_limit.get("min_value"),
                parsed_limit.get("max_value"),
                condition_label,
            )
            if key in seen:
                continue
            seen.add(key)

            standard = {
                "id": f"{slug(item['name'])}_{slug(scope)}_{slug(canonical)}_{len(standards) + 1}",
                "domain": "wastewater",
                "industry": scope,
                "pollutant": canonical,
                "aliases": list(dict.fromkeys(alias for alias in aliases if alias != canonical)),
                "requires_industry": True,
                "source_name": item["name"],
                "source_article": source_article(item["name"]),
                "source_file": item["file"],
                "notes": note or "由放流水標準附表表格自動結構化；複雜適用條件保留於 conditions。",
                "scope_type": item.get("scope", "industry"),
                "conditions": conditions,
            }
            standard.update(parsed_limit)
            standards.append(standard)

    return standards


def generate_pollutant_catalog(standards: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    pollutant_aliases: dict[str, set[str]] = {}
    for standard in standards:
        pollutant_aliases.setdefault(standard["pollutant"], set()).add(standard["pollutant"])
        pollutant_aliases[standard["pollutant"]].update(standard.get("aliases", []))

    pollutants = []
    for canonical in sorted(pollutant_aliases):
        unit = "" if canonical == "pH" else "mg/L"
        pollutants.append({
            "canonical": canonical,
            "aliases": sorted(pollutant_aliases[canonical], key=lambda text: (len(text), text)),
            "default_unit": unit,
            "allowed_units": [""] if unit == "" else ["mg/L", "毫克/公升", "毫克／公升", "mg/l"],
        })

    industries = []
    for item in inventory:
        if item.get("type") != "standard_attachment":
            continue
        canonical = source_scope(item["name"], item.get("scope", ""))
        aliases = [alias for alias in (canonical, item["name"]) if alias]
        for token in re.split(r"[、/，,及和]+", canonical):
            token = token.strip()
            if len(token) >= 2:
                aliases.append(token)
                for suffix in ("製造業", "專用污水下水道系統", "污水下水道系統", "污水下水道", "處理設施", "專業區", "系統"):
                    if token.endswith(suffix):
                        short = token[:-len(suffix)]
                        if len(short) >= 2:
                            aliases.append(short)
                if token.endswith("業") and len(token) >= 3:
                    aliases.append(token[:-1])
        industries.append({
            "canonical": canonical,
            "aliases": [alias for alias in dict.fromkeys(aliases) if alias],
        })

    industries = sorted(
        industries,
        key=lambda item: (len(item["canonical"]), item["canonical"]),
    )

    return {
        "pollutants": pollutants,
        "industries": industries,
        "retrieval_terms": ["放流水標準", "水質項目", "限值", "標準值"],
    }


def update_inventory(inventory: list[dict[str, Any]], standards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_file: dict[str, list[str]] = {}
    for standard in standards:
        by_file.setdefault(standard["source_file"], []).append(standard["pollutant"])

    for item in inventory:
        structured_items = sorted(set(by_file.get(item["file"], [])))
        if structured_items:
            item["structured"] = True
            item["structured_items"] = structured_items
            item["notes"] = "已由 PDF 表格自動結構化為 standards.json"
        elif item.get("type") == "law_text":
            item["structured"] = False
            item["structured_items"] = []
            item["notes"] = "全文進 RAG；非水質限值表，維持 RAG 查詢"
    return inventory


def main() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    standards = generate_standards(inventory)
    catalog = generate_pollutant_catalog(standards, inventory)
    inventory = update_inventory(inventory, standards)

    STANDARDS_PATH.write_text(json.dumps(standards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    POLLUTANTS_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    INVENTORY_PATH.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"standards: {len(standards)}")
    print(f"pollutants: {len(catalog['pollutants'])}")
    print(f"industries: {len(catalog['industries'])}")


if __name__ == "__main__":
    main()
