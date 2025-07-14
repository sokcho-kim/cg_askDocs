import json
import time
from memory_profiler import memory_usage
from rag.retriever import ChromaRetriever, FaissRetriever, get_embedding_function

# 데이터 로드 함수 (예시)
def load_chunks():
    chunks = []
    for fname in [
        'data/processed/DR_공정회의자료_추출본(데모용)_chunks.json',
        'data/processed/DR_스마트야드개론(데모용)_chunks.json',
    ]:
        try:
            with open(fname, encoding='utf-8') as f:
                chunks.extend(json.load(f))
        except Exception as e:
            print(f"[경고] {fname} 로드 실패: {e}")
    return chunks

# 테스트 쿼리
queries = ["스마트 야드", "자동화", "AI 기술"]

# 벤치마크 함수
def run_index_and_search(retriever_class, chunks, queries):
    retriever = retriever_class(get_embedding_function())
    retriever.index_chunks(chunks)
    results = []
    for q in queries:
        start = time.time()
        res = retriever.search(q)
        elapsed = time.time() - start
        results.append((q, elapsed, [r['content'][:30] for r in res]))
    return results

def benchmark(retriever_class, chunks, queries, name):
    print(f"\n==== {name} ====")
    mem_usage, results = memory_usage(
        (run_index_and_search, (retriever_class, chunks, queries)),
        retval=True, max_usage=True, interval=0.1
    )
    print(f"[{name}] 최대 메모리 사용량: {mem_usage:.2f}MB")
    for q, elapsed, res in results:
        print(f"Q: {q} | 검색시간: {elapsed:.2f}s | 결과: {res}")

if __name__ == "__main__":
    chunks = load_chunks()
    benchmark(ChromaRetriever, chunks, queries, "ChromaDB")
    benchmark(FaissRetriever, chunks, queries, "FAISS") 