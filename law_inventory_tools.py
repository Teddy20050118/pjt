"""
法規清單與 RAG metadata 檢查工具。

用途：
    1. 掃描廢水法規資料夾產生 law_inventory.json。
    2. 讀取 ChromaDB sqlite metadata，標記哪些檔案已進 RAG。

這個工具不需要 chromadb 套件，僅讀取 chroma.sqlite3 的 metadata。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Set


LAW_DIR = Path("廢水法規")
CHROMA_SQLITE = Path("chroma_db") / "chroma.sqlite3"
INVENTORY_PATH = Path("law_inventory.json")


def indexed_source_files(sqlite_path: Path = CHROMA_SQLITE) -> Set[str]:
    """讀取 ChromaDB metadata 中已索引的 source_file。"""
    if not sqlite_path.exists():
        return set()

    with sqlite3.connect(sqlite_path) as con:
        rows = con.execute(
            """
            SELECT DISTINCT string_value
            FROM embedding_metadata
            WHERE key = 'source_file' AND string_value IS NOT NULL
            """
        ).fetchall()

    return {row[0] for row in rows}


def classify_file(path: Path) -> Dict[str, str]:
    """依檔名分類法規類型與範圍。"""
    name = path.stem.strip()

    if name.startswith("附表"):
        entry_type = "standard_attachment"
    elif "放流水標準" in name:
        entry_type = "standard_main_text"
    else:
        entry_type = "law_text"

    if any(token in name for token in ("專用污水下水道", "下水道系統")):
        scope = "sewer_system"
    elif any(token in name for token in ("指定地區", "場所", "社區", "公共污水")):
        scope = "region"
    elif name.startswith("附表"):
        scope = "industry"
    else:
        scope = "general"

    return {"type": entry_type, "scope": scope}


def known_structured_items(file_name: str) -> List[str]:
    """目前已人工結構化的項目。"""
    if file_name == "附表一晶圓製造及半導體製造業放流水水質項目及限值.pdf":
        return [
            "pH",
            "COD",
            "SS",
            "TTO",
            "2-METHOXY-1-PROPANOL",
            "DMAC",
            "COBALT",
            "ANTIMONY",
        ]
    return []


def build_inventory(law_dir: Path = LAW_DIR,
                    sqlite_path: Path = CHROMA_SQLITE) -> List[Dict]:
    """建立法規清單。"""
    indexed = indexed_source_files(sqlite_path)
    inventory = []

    for path in sorted(law_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        structured_items = known_structured_items(path.name)
        classification = classify_file(path)
        inventory.append({
            "name": path.stem.strip(),
            "file": path.name,
            "type": classification["type"],
            "scope": classification["scope"],
            "rag_indexed": path.name in indexed,
            "structured": bool(structured_items),
            "structured_items": structured_items,
            "notes": "已結構化核心 demo 項目" if structured_items else "全文進 RAG；尚未結構化比對",
        })

    return inventory


def write_inventory(path: Path = INVENTORY_PATH) -> List[Dict]:
    """寫出 law_inventory.json 並回傳清單。"""
    inventory = build_inventory()
    path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    return inventory


def summarize_inventory(inventory: List[Dict]) -> Dict[str, int]:
    """產生簡短統計。"""
    return {
        "total": len(inventory),
        "rag_indexed": sum(1 for item in inventory if item["rag_indexed"]),
        "structured": sum(1 for item in inventory if item["structured"]),
    }


if __name__ == "__main__":
    result = write_inventory()
    print(json.dumps(summarize_inventory(result), ensure_ascii=False, indent=2))
