"""
真实数据一键处理
===============
MediaCrawler 数据 → 清洗 → Chunking → 入 Milvus → 重跑评估

"""

import json
import math
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.crawler import load_raw_data, get_keywords_map
from src.data.cleaner import clean_note, infer_category
from src.data.chunker import chunk_notes
from src.retrieval.milvus_store import MilvusStore, EMBEDDING_DIM
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.eval_set_generator import EvalSetGenerator


def random_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """临时用随机向量，等 bge-m3 接入后替换。"""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def merge_comments(notes: list[dict], comments_file: Path) -> list[dict]:
    """把评论数据合并到笔记里。"""
    if not comments_file.exists():
        return notes

    comments_data = json.loads(comments_file.read_text(encoding="utf-8"))

    # 按 note_id 分组评论
    comment_map: dict[str, list[str]] = {}
    for c in comments_data:
        nid = c.get("note_id", "")
        content = c.get("content", "")
        if nid and content:
            comment_map.setdefault(nid, []).append(content)

    # 合并到笔记
    for note in notes:
        nid = note.get("note_id", "")
        if nid in comment_map:
            note["comments"] = comment_map[nid][:20]

    merged_count = sum(1 for n in notes if n.get("comments"))
    print(f"  评论合并: {merged_count}/{len(notes)} 条笔记有评论")
    return notes


def main():
    print("=" * 60)
    print("  真实数据一键处理")
    print("=" * 60)

    keywords_map = get_keywords_map()

    # ── Step 1: 读取原始数据 ──────────────────────────────────
    print("\n【Step 1】读取原始数据")
    raw_notes = load_raw_data("data/raw")

    if not raw_notes:
        print("  ❌ data/raw/ 下没有数据文件，请先复制 MediaCrawler 数据")
        return

    # 合并评论
    comments_file = Path("data/raw/search_comments_2026-05-17.json")
    raw_notes = merge_comments(raw_notes, comments_file)

    # ── Step 2: 清洗 ─────────────────────────────────────────
    print("\n【Step 2】清洗数据")
    cleaned = []
    category_count = {}
    for raw in raw_notes:
        category = infer_category(raw, keywords_map)
        # MediaCrawler 的字段适配
        raw_adapted = {
            "note_id": raw.get("note_id", ""),
            "title": raw.get("title", ""),
            "desc": raw.get("desc", ""),
            "tag_list": raw.get("tag_list", ""),
            "interact_info": {
                "liked_count": raw.get("liked_count", 0),
                "comment_count": raw.get("comment_count", 0),
            },
            "user": {"nickname": raw.get("nickname", "")},
            "comments": raw.get("comments", []),
        }
        # 解析 tag_list（MediaCrawler 存的是字符串）
        tag_str = raw.get("tag_list", "")
        if isinstance(tag_str, str) and tag_str:
            raw_adapted["tag_list"] = [
                {"name": t.strip()} for t in tag_str.replace("[", "").replace("]", "").split(",") if t.strip()
            ]

        note = clean_note(raw_adapted, category=category)
        if note:
            cleaned.append(note)
            category_count[category] = category_count.get(category, 0) + 1

    print(f"  清洗结果: {len(raw_notes)} 条原始 → {len(cleaned)} 条有效")
    for cat, cnt in sorted(category_count.items()):
        print(f"    {cat}: {cnt} 条")

    # 保存清洗结果
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    notes_file = Path("data/processed/notes.json")
    notes_file.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存: {notes_file}")

    # ── Step 3: Chunking ─────────────────────────────────────
    print("\n【Step 3】Chunking")
    chunks = chunk_notes(cleaned, max_chars=300)
    print(f"  {len(cleaned)} 条笔记 → {len(chunks)} 个 chunks")

    # 保存 chunks
    chunks_data = [
        {"chunk_id": c.chunk_id, "note_id": c.note_id, "text": c.text,
         "chunk_type": c.chunk_type, "metadata": c.metadata}
        for c in chunks
    ]
    chunks_file = Path("data/processed/chunks.json")
    chunks_file.write_text(json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存: {chunks_file}")

    # ── Step 4: 入 Milvus ────────────────────────────────────
    print("\n【Step 4】入 Milvus 向量库")
    store = MilvusStore(db_path="data/milvus_lite.db")
    store.init_collection(drop_if_exists=True)

    # 生成 embedding（暂用随机向量，等 bge-m3 替换）
    milvus_chunks = []
    for c in chunks_data:
        milvus_chunks.append({
            "chunk_id": c["chunk_id"],
            "note_id": c["note_id"],
            "text": c["text"][:4096],
            "embedding": random_embedding(),
            "metadata": c["metadata"],
        })

    inserted = store.insert(milvus_chunks)
    print(f"  插入 {inserted} 条到 Milvus | 总量: {store.count()}")

    # ── Step 5: 建 BM25 索引 ─────────────────────────────────
    print("\n【Step 5】建 BM25 索引")
    bm25 = BM25Retriever()
    bm25.build_index(chunks_data)
    bm25.save_index("data/bm25_index.pkl")

    # ── Step 6: 生成评估集 ────────────────────────────────────
    print("\n【Step 6】生成评估集")
    generator = EvalSetGenerator(llm_client=None)
    eval_set = generator.generate_rule_based(cleaned)
    generator.save(eval_set, "data/eval/eval_set.json")

    # ── Step 7: 跑评估 ───────────────────────────────────────
    print("\n【Step 7】消融实验")

    class MockDense:
        def __init__(self, chunks):
            self.chunks = chunks
        def search(self, query, top_k=10):
            shuffled = self.chunks.copy()
            random.shuffle(shuffled)
            return [{**c, "score": round(random.uniform(0.5, 0.95), 4),
                     "rank": i+1, "source": "dense"}
                    for i, c in enumerate(shuffled[:top_k])]

    class HybridWrapper:
        def __init__(self, bm25, dense):
            self.bm25 = bm25
            self.dense = dense
        def search(self, query, top_k=10):
            b = self.bm25.search(query, top_k=top_k)
            d = self.dense.search(query, top_k=top_k)
            return reciprocal_rank_fusion([b, d])[:top_k]

    mock_dense = MockDense(chunks_data)
    hybrid = HybridWrapper(bm25, mock_dense)

    evaluator = RetrievalEvaluator()
    evaluator.load_eval_set_from_list(eval_set)

    bm25_metrics = evaluator.evaluate(bm25, k=10)
    hybrid_metrics = evaluator.evaluate(hybrid, k=10)

    print(f"\n  {'配置':<25} {'Recall@10':>10} {'MRR':>8} {'NDCG@10':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*8}")
    print(f"  {'纯 BM25':<25} {bm25_metrics['recall@10']:>10.4f} {bm25_metrics['mrr']:>8.4f} {bm25_metrics['ndcg@10']:>8.4f}")
    print(f"  {'BM25+Dense+RRF':<25} {hybrid_metrics['recall@10']:>10.4f} {hybrid_metrics['mrr']:>8.4f} {hybrid_metrics['ndcg@10']:>8.4f}")

    delta_recall = hybrid_metrics['recall@10'] - bm25_metrics['recall@10']
    print(f"\n  混合检索 Recall@10 提升: {delta_recall:+.4f}")

    # 保存报告
    report = {
        "data_stats": {
            "raw_notes": len(raw_notes),
            "cleaned_notes": len(cleaned),
            "chunks": len(chunks),
            "categories": category_count,
        },
        "eval_set_size": len(eval_set),
        "k": 10,
        "results": {
            "BM25": bm25_metrics,
            "Hybrid(BM25+Dense+RRF)": hybrid_metrics,
        },
        "delta": {
            "recall@10": round(delta_recall, 4),
        },
        "note": "Dense 检索暂用随机向量，接入 bge-m3 后数字会更准确",
    }
    evaluator.save_report(report, "evaluation/reports/real_data_report.json")

    print("\n" + "=" * 60)
    print("  ✅ 真实数据处理完成！")
    print(f"  清洗笔记: {len(cleaned)} 条")
    print(f"  Chunks: {len(chunks)} 个")
    print(f"  Milvus: {store.count()} 条向量")
    print(f"  评估集: {len(eval_set)} 条")
    print(f"  报告: evaluation/reports/real_data_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()