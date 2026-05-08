"""
法規 RAG（Retrieval-Augmented Generation）Pipeline
====================================================
功能：
    1. 遍歷指定資料夾，自動讀取 .txt 與 .pdf 法規檔案
    2. 以正規表達式依「第 X 條」切割為獨立 Chunk
    3. 附加 metadata（法規名稱、條號、來源檔名）後存入 ChromaDB
    4. 提供語意查詢介面供 Agent 呼叫

依賴套件：
    pip install pymupdf chromadb
    ollama pull nomic-embed-text
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import chromadb


def _has_chinese(text: str) -> bool:
    """判斷字串中是否含有中文字元（CJK 統一漢字區段 U+4E00~U+9FFF）。
    用於診斷 PDF 擷取結果是否有效。"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# ── 常數設定 ─────────────────────────────────────────────────────────────────

# 匹配行首的「第 X 條」，支援阿拉伯數字、全形數字、中文數字及空格變體
# 使用 re.MULTILINE 確保 ^ 對每行行首生效，避免條文內引用（如「依第5條」）被誤切
ARTICLE_PATTERN = re.compile(
    r'^第\s*[0-9０-９零一二三四五六七八九十百千]+\s*條',
    re.MULTILINE
)

# fallback 固定長度切割的預設字元數（用於無條文結構的附表類檔案）
FALLBACK_CHUNK_SIZE = 500

# 每批 upsert 的 chunk 數量上限
# 台灣法規單一檔案可能有 100+ 條文，單次全部送 Ollama embedding 可能超時或耗盡記憶體
UPSERT_BATCH_SIZE = 50

# 單一條文超過此字元數時，自動呼叫 _split_by_length 進一步切碎
# nomic-embed-text context window 約 2048 tokens，保守設 1500 字元
MAX_CHUNK_CHARS = 1500


# ── 主要類別 ─────────────────────────────────────────────────────────────────

class LawRAG:
    """
    法規向量資料庫。

    基本用法：
        rag = LawRAG()
        rag.build_from_folder(r"C:\\path\\to\\廢水法規")
        results = rag.retrieve("COD 放流水標準")
    """

    def __init__(self,
                 db_path: str = "./chroma_db",
                 collection_name: str = "law_chunks",
                 embed_model: str = "nomic-embed-text",
                 ollama_host: str = "http://127.0.0.1:11434",
                 ):
        """
        初始化向量資料庫連線與 Embedding 函數。

        Args:
            db_path:         ChromaDB 持久化儲存路徑
            collection_name: ChromaDB collection 名稱
            embed_model:     Ollama embedding 模型名稱
            ollama_host:     Ollama 服務位址
        """
        # 建立 Ollama Embedding 函數
        self._embed_fn = OllamaEmbeddingFunction(
            model_name=embed_model,
            url=f"{ollama_host}/api/embeddings",
        )

        # 建立 ChromaDB 持久化客戶端
        self._client = chromadb.PersistentClient(path=db_path)

        # 取得或建立 collection，使用 cosine 相似度
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── 公開介面 ─────────────────────────────────────────────────────────────

    def build_from_folder(self, folder_path: str) -> int:
        """
        遍歷資料夾，將所有 .txt 與 .pdf 法規檔案建立向量索引。

        Args:
            folder_path: 法規資料夾路徑

        Returns:
            int: 本次新增或更新的 chunk 總數
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"資料夾不存在：{folder_path}")

        # 啟動前驗證：先送一個測試字串給 Ollama embedding，
        # 確認 nomic-embed-text 模型可用，避免等到 upsert 才爆出錯誤
        print("[RAG] 驗證 Ollama embedding 連線...")
        try:
            self._embed_fn(["連線測試"])
            print("[RAG] Ollama embedding 連線正常")
        except Exception as e:
            raise RuntimeError(
                f"[RAG] Ollama embedding 失敗，請確認 Ollama 服務已啟動且 nomic-embed-text 模型已下載。"
                f"\n原始錯誤：{str(e)}"
            )

        total_chunks = 0

        for file_path in sorted(folder.iterdir()):
            suffix = file_path.suffix.lower()

            # 略過非目標檔案類型
            if suffix not in (".txt", ".pdf"):
                continue

            law_name = self._extract_law_name(file_path)
            print(f"[RAG] 處理：{file_path.name}")

            try:
                # 依副檔名分流讀取文字
                if suffix == ".txt":
                    text = self._read_txt(file_path)
                else:
                    text = self._read_pdf(file_path)

                # 切割成 chunks
                chunks = self._split_by_article(text, law_name, file_path.name)

                if not chunks:
                    print(f"[RAG]   找不到條文結構，改用固定長度切割")
                    chunks = self._split_by_length(text, law_name, file_path.name)

                # 寫入 ChromaDB
                self._upsert_chunks(chunks)
                print(f"[RAG]   完成，{len(chunks)} 個 chunks")
                total_chunks += len(chunks)

            except Exception as e:
                print(f"[RAG]   錯誤，跳過此檔案：{str(e)}")
                continue

        print(f"[RAG] 建置完成，本次共處理 {total_chunks} 個 chunks，資料庫總計 {self.count()} 個 chunks")
        return total_chunks

    def retrieve(self,
                 query: str,
                 n_results: int = 5,
                 law_filter: Optional[str] = None,
                 ) -> List[Dict]:
        """
        以自然語言查詢法規資料庫。

        Args:
            query:      查詢字串（自然語言）
            n_results:  回傳筆數上限
            law_filter: 限制特定法規名稱（None 表示搜尋全部法規）

        Returns:
            List[Dict]: 每筆結果包含 text、law_name、article、source_file、score
        """
        where = {"law_name": {"$eq": law_filter}} if law_filter else None

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[RAG] 查詢失敗：{str(e)}")
            return []

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text":        doc,
                "law_name":    meta.get("law_name", ""),
                "article":     meta.get("article", ""),
                "source_file": meta.get("source_file", ""),
                "score":       round(1 - dist, 4),  # cosine distance → similarity
            })

        return output

    def count(self) -> int:
        """回傳資料庫中目前的 chunk 總數。"""
        return self._collection.count()

    # ── 私有方法：檔案讀取 ───────────────────────────────────────────────────

    def _read_txt(self, file_path: Path) -> str:
        """
        讀取 UTF-8 純文字法規檔案。

        Args:
            file_path: 檔案路徑

        Returns:
            str: 檔案全文
        """
        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def _read_pdf(self, file_path: Path) -> str:
        """
        使用 PyMuPDF 萃取 PDF 全文。

        台灣政府 PDF 常見問題：部分檔案使用 CIDFont + 自訂 ToUnicode CMap，
        直接用 get_text("text") 會把中文字輸出為數字或空白。
        本方法採用三階段嘗試策略，從最可靠到最後備援依序嘗試：

        第一階段：get_text("blocks") — 以文字區塊回傳，
                  對 CIDFont 編碼的 PDF 相容性較好。
        第二階段：get_text("text", flags=...) — 保留連字符與空白的純文字模式。
        第三階段：如果前兩者仍無有效中文，記錄警告並回傳原始結果。

        Args:
            file_path: PDF 檔案路徑

        Returns:
            str: 萃取後的純文字（可能為空字串，由上層處理）
        """
        text_parts = []

        with fitz.open(str(file_path)) as doc:
            for page in doc:
                # 第一階段：blocks 模式，對 CIDFont PDF 相容性最好
                # blocks 回傳格式：[(x0, y0, x1, y1, text, block_no, block_type), ...]
                # block_type=0 為文字區塊
                blocks = page.get_text("blocks", sort=True)
                page_text = "\n".join(
                    b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()
                )

                # 第二階段：若 blocks 模式無法取得中文（只有數字與空白），
                # 嘗試帶旗標的 get_text("text")
                if page_text and not _has_chinese(page_text):
                    page_text = page.get_text(
                        "text",
                        flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE,
                    )

                if page_text.strip():
                    text_parts.append(page_text)

        full_text = "\n".join(text_parts)

        # 診斷 log：顯示擷取結果前 80 字元，讓開發者快速確認中文是否正確擷取
        preview = full_text[:80].replace("\n", " ")
        has_cn = _has_chinese(full_text)
        print(f"[RAG]   PDF 擷取預覽（含中文：{has_cn}）：{preview}")

        return full_text

    # ── 私有方法：輔助工具 ───────────────────────────────────────────────────

    def _make_id(self, *parts: str) -> str:
        """
        產生安全的 ChromaDB chunk ID。
        將各部分以底線連接，並將所有半形/全形空格替換為底線，
        避免含空格的檔名（如「附表一  晶圓...」）產生無效 ID。

        Args:
            *parts: ID 組成片段（如 law_name、article_normalized）

        Returns:
            str: 符合 ChromaDB 規範的 ID 字串
        """
        raw = "_".join(parts)
        return re.sub(r'[\s\u3000]+', '_', raw)  # 半形空格、全形空格 → _

    def _extract_law_name(self, file_path: Path) -> str:
        """
        從檔名（不含副檔名）取得法規名稱。
        例如：「水污染防治法.pdf」→「水污染防治法」

        Args:
            file_path: 檔案路徑

        Returns:
            str: 法規名稱
        """
        return file_path.stem.strip()

    # ── 私有方法：文字切割 ───────────────────────────────────────────────────

    def _split_by_article(self,
                          text: str,
                          law_name: str,
                          source_file: str,
                          ) -> List[Dict]:
        """
        依「第 X 條」為分隔基準切割法規文字。

        內建兩項保護：
        1. 重複 ID 保護：同一條號出現多次時（如目錄區段），後續加後綴 _1, _2 ...
        2. 長度保護：條文超過 MAX_CHUNK_CHARS 字元時，呼叫 _split_by_length 二次切割

        Args:
            text:        法規全文
            law_name:    法規名稱
            source_file: 來源檔名（含副檔名）

        Returns:
            List[Dict]: 每個元素為 {"id": ..., "text": ..., "metadata": {...}}
                        若無條文結構則回傳空串列
        """
        matches = list(ARTICLE_PATTERN.finditer(text))

        if not matches:
            return []

        chunks = []
        used_ids: set = set()  # 追蹤已用 ID，防止目錄重複條號造成 ChromaDB 衝突

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            chunk_text = text[start:end].strip()
            if not chunk_text:
                continue

            # 取得條號字串，去除多餘空白（如「第 1 條」→「第1條」，用於 ID）
            article_raw = match.group().strip()
            article_normalized = re.sub(r'\s+', '', article_raw)
            base_id = self._make_id(law_name, article_normalized)

            # 重複 ID 保護：目錄重複條號加後綴 _1, _2, ...
            chunk_id = base_id
            if chunk_id in used_ids:
                n = 1
                while f"{base_id}_{n}" in used_ids:
                    n += 1
                chunk_id = f"{base_id}_{n}"
            used_ids.add(chunk_id)

            # 長度保護：超過 MAX_CHUNK_CHARS 時二次切割，避免超出 nomic-embed-text context window
            if len(chunk_text) > MAX_CHUNK_CHARS:
                sub_chunks = self._split_by_length(chunk_text, law_name, source_file)
                for j, sub in enumerate(sub_chunks):
                    sub_id = f"{chunk_id}_part{j}"
                    used_ids.add(sub_id)
                    chunks.append({
                        "id":   sub_id,
                        "text": sub["text"],
                        "metadata": {
                            "law_name":    law_name,
                            "article":     article_raw,
                            "source_file": source_file,
                        },
                    })
                continue

            chunks.append({
                "id":   chunk_id,
                "text": chunk_text,
                "metadata": {
                    "law_name":    law_name,
                    "article":     article_raw,      # 保留原始格式供顯示
                    "source_file": source_file,
                },
            })

        return chunks

    def _split_by_length(self,
                         text: str,
                         law_name: str,
                         source_file: str,
                         ) -> List[Dict]:
        """
        固定字元數切割，作為無條文結構（如附表）時的 fallback。

        Args:
            text:        文字全文
            law_name:    法規或文件名稱
            source_file: 來源檔名

        Returns:
            List[Dict]: chunk 串列
        """
        chunks = []
        index = 0
        start = 0

        while start < len(text):
            chunk_text = text[start: start + FALLBACK_CHUNK_SIZE].strip()
            if chunk_text:
                chunks.append({
                    "id":   self._make_id(law_name, f"chunk{index}"),
                    "text": chunk_text,
                    "metadata": {
                        "law_name":    law_name,
                        "article":     "",           # 附表無條號
                        "source_file": source_file,
                    },
                })
                index += 1
            start += FALLBACK_CHUNK_SIZE

        return chunks

    # ── 私有方法：資料庫寫入 ─────────────────────────────────────────────────

    def _upsert_chunks(self, chunks: List[Dict]) -> None:
        """
        分批 upsert chunks 至 ChromaDB。

        為何分批：
            單一法規檔案可能有 100+ 條文，全部一次送 Ollama embedding 請求，
            可能造成 HTTP 超時（nomic-embed-text 每次請求需要幾秒）或記憶體不足。
            分批寫入可降低單次請求大小，並讓進度可見。

        為何加獨立 try-except：
            原有設計把 upsert 包在外層 build_from_folder 的 try-except 中，
            embedding 失敗時無法區分是「讀檔錯誤」還是「寫入錯誤」。
            此處獨立捕捉讓錯誤訊息更具體。

        Args:
            chunks: _split_by_article 或 _split_by_length 回傳的串列
        """
        if not chunks:
            return

        total = len(chunks)
        written = 0

        # 依 UPSERT_BATCH_SIZE 分批處理，避免單次請求過大
        for batch_start in range(0, total, UPSERT_BATCH_SIZE):
            batch = chunks[batch_start: batch_start + UPSERT_BATCH_SIZE]
            try:
                self._collection.upsert(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[c["metadata"] for c in batch],
                )
                written += len(batch)
            except Exception as e:
                # 精確顯示哪一批失敗，幫助定位問題（embedding 失敗或 ChromaDB 寫入失敗）
                raise RuntimeError(
                    f"第 {batch_start // UPSERT_BATCH_SIZE + 1} 批 upsert 失敗"
                    f"（chunk {batch_start}~{batch_start + len(batch) - 1}）：{str(e)}"
                ) from e

if __name__ == "__main__":
    # 指定你的法規檔案放在哪個資料夾（假設是專案目錄下的 data 資料夾）
    TARGET_FOLDER = "./廢水法規"
    
    print("啟動法規知識庫建置程序...")
    
    # 1. 實例化你寫好的 LawRAG 類別
    rag = LawRAG()
    
    # 2. 呼叫 build_from_folder 開始讀取並寫入資料庫
    try:
        total_upserted = rag.build_from_folder(TARGET_FOLDER)
        print(f"\n建置成功 總共寫入了 {total_upserted} 筆 chunk。")
    except Exception as e:
        print(f"\n執行過程中發生錯誤: {e}")