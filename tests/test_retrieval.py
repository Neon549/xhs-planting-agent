"""检索单元测试"""
from src.retrieval.bm25_retriever import BM25Retriever

def test_bm25_build_and_search():
    """BM25 建索引后能正常检索"""
    chunks = [
        {"chunk_id": "c1", "note_id": "n1", "text": "iPhone手机推荐"},
        {"chunk_id": "c2", "note_id": "n2", "text": "哑铃健身器材选购"},
    ]
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    results = bm25.search("iPhone", top_k=5)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"