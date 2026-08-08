"""混合检索：BM25（jieba 分词）+ 向量语义 + RRF 融合。

降级链（向量层不可用时自动退化为纯 BM25，并在 trace 中记录）：
- DeepSeek 无 embedding 模型，向量层用本地 bge-small-zh（sentence-transformers）
- 若本地模型未安装/加载失败 → BM25 单路召回（对 24 商品/5 文档的规模仍可用）

chunk 策略（吸收 BuyPilot 低 Context Precision 的教训——切分是质量天花板）：
- 知识库：按 markdown 二级标题切分，一篇文档 ≈ 5-8 个语义完整 chunk
- 商品：一商品一 chunk（名称+品类+材质+规格+标签+描述），不做二次切分
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jieba
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RRF_K = 60  # RRF 平滑常数（业界标准值）


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


class Retriever:
    def __init__(self) -> None:
        self.docs: list[dict] = []          # {doc_id, kind, text, payload}
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None
        self._embedder = None
        self.vector_ready = False
        self._load_docs()
        self._build_bm25()
        self._try_build_vector()

    # ------------------------------------------------------------ 索引构建
    def _load_docs(self) -> None:
        products = json.loads((DATA_DIR / "products.json").read_text("utf-8"))
        for p in products:
            text = (
                f"{p['name']} {p['category']} 材质{p['material']} "
                f"规格{p['spec']} {' '.join(p['tags'])} {p['desc']}"
            )
            self.docs.append({
                "doc_id": p["product_id"], "kind": "product",
                "text": text, "payload": p,
            })
        for md in sorted((DATA_DIR / "knowledge").glob("*.md")):
            self._chunk_knowledge(md)

    def _chunk_knowledge(self, path: Path) -> None:
        """按二级标题切分，保留标题上下文。"""
        title = path.stem
        current_head, buf = "", []
        idx = 0

        def flush() -> None:
            nonlocal idx
            if not buf:
                return
            body = "\n".join(buf).strip()
            if body:
                self.docs.append({
                    "doc_id": f"KB-{title}-{idx}", "kind": "knowledge",
                    "text": f"{title} {current_head}\n{body}",
                    "payload": {"doc": title, "section": current_head},
                })
                idx += 1

        for line in path.read_text("utf-8").splitlines():
            if line.startswith("## "):
                flush()
                current_head, buf = line[3:].strip(), []
            elif not line.startswith("# "):
                buf.append(line)
        flush()

    def _build_bm25(self) -> None:
        self.bm25 = BM25Okapi([_tokenize(d["text"]) for d in self.docs])

    def _try_build_vector(self) -> None:
        def _load(offline: bool) -> None:
            if offline:
                # 优先本地缓存加载：HF Hub 网络抖动不应导致向量层静默失效
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")

        try:
            try:
                _load(offline=True)
            except Exception:
                _load(offline=False)
            embs = self._embedder.encode(
                [d["text"] for d in self.docs], normalize_embeddings=True
            )
            self.embeddings = np.asarray(embs)
            self.vector_ready = True
            logger.info("vector index ready (bge-small-zh)")
        except Exception as e:  # noqa: BLE001 — 降级链：任何失败都退回纯 BM25
            logger.warning(f"vector index unavailable, BM25 only: {e}")

    # ------------------------------------------------------------ 检索
    def search(
        self, query: str, *, kind: str | None = None, top_k: int = 5
    ) -> list[dict]:
        """RRF 融合排序。kind 限定 'product' / 'knowledge'。"""
        n = len(self.docs)
        bm25_ranks = self._bm25_rank(query)
        vector_ranks = self._vector_rank(query) if self.vector_ready else None

        scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranks):
            scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)
        if vector_ranks:
            for rank, idx in enumerate(vector_ranks):
                scores[idx] = scores.get(idx, 0) + 1 / (RRF_K + rank + 1)

        ranked = sorted(scores, key=scores.get, reverse=True)
        results = []
        for idx in ranked:
            doc = self.docs[idx]
            if kind and doc["kind"] != kind:
                continue
            results.append({**doc, "score": round(scores[idx], 4)})
            if len(results) >= top_k:
                break
        return results

    def _bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)

    def _vector_rank(self, query: str) -> list[int]:
        q = self._embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        return sorted(range(len(self.docs)), key=lambda i: sims[i], reverse=True)


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
