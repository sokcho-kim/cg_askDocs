# 문서 데이터를 ChromaDB 벡터 DB로 인덱싱하는 스크립트
"""
PDF/Excel 등에서 추출된 청크 리스트를 받아 ChromaDB에 저장합니다.
- 입력: get_chunks() 결과 리스트 또는 JSON 파일
- 출력: ChromaDB에 (chunk_id, content, embedding, metadata) 저장
- 담당: 속초/제로 모두 사용
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag.retriever import EnhancedRetriever


def load_chunks_from_json(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 청크 데이터를 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"[✓] {len(chunks)}개 청크를 {file_path}에서 로드했습니다.")
        return chunks
    except Exception as e:
        print(f"[❌] 파일 로드 실패: {e}")
        return []


def index_chunks_to_chroma(chunks: List[Dict[str, Any]], clear_existing: bool = False):
    """청크들을 ChromaDB에 인덱싱합니다."""
    retriever = EnhancedRetriever()
    
    # 기존 데이터 삭제 (선택사항)
    if clear_existing:
        print("[🗑️] 기존 컬렉션을 초기화합니다...")
        retriever.clear_collection()
    
    # 청크들을 벡터 DB에 추가
    print(f"[📥] {len(chunks)}개 청크를 ChromaDB에 추가 중...")
    retriever.add_chunks(chunks)
    
    # 통계 정보 출력
    stats = retriever.get_collection_stats()
    print(f"[📊] 인덱싱 완료! 통계: {stats}")
    
    return retriever


def test_search_functionality(retriever: EnhancedRetriever):
    """검색 기능을 테스트합니다."""
    print("\n" + "=" * 50)
    print("🔍 검색 기능 테스트")
    print("=" * 50)
    
    test_queries = [
        "스마트 야드",
        "자동화",
        "AI 기술",
        "생산성 향상",
        "조선소"
    ]
    
    for query in test_queries:
        print(f"\n🔍 쿼리: '{query}'")
        
        # 하이브리드 검색
        results = retriever.hybrid_search(query, 3)
        if results:
            print(f"  하이브리드 검색 결과 ({len(results)}개):")
            for i, result in enumerate(results, 1):
                score = result.get('final_score', 0)
                content_preview = result['content'][:50] + "..." if len(result['content']) > 50 else result['content']
                print(f"    {i}. {result['chunk_id']} (점수: {score:.3f})")
                print(f"       내용: {content_preview}")
        else:
            print("  검색 결과가 없습니다.")
    
    # 의미적 검색만 테스트
    print(f"\n🔍 의미적 검색 테스트: '스마트 야드 자동화'")
    semantic_results = retriever.semantic_search("스마트 야드 자동화", 2)
    if semantic_results:
        print(f"  의미적 검색 결과 ({len(semantic_results)}개):")
        for i, result in enumerate(semantic_results, 1):
            score = result.get('score', 0)
            print(f"    {i}. {result['chunk_id']} (유사도: {score:.3f})")
    else:
        print("  의미적 검색 결과가 없습니다.")


def main():
    """메인 함수"""
    print("🚀 ChromaDB 인덱싱 시작")
    print("=" * 50)
    
    # 처리할 파일들
    chunk_files = [
        "data/processed/DR_공정회의자료_추출본(데모용)_chunks.json",
        "data/processed/DR_스마트야드개론(데모용)_chunks.json"
    ]
    
    all_chunks = []
    
    # 각 파일에서 청크 로드
    for file_path in chunk_files:
        if Path(file_path).exists():
            print(f"\n📁 파일 처리 중: {file_path}")
            chunks = load_chunks_from_json(file_path)
            all_chunks.extend(chunks)
            print(f"  ✓ {len(chunks)}개 청크 추가됨")
        else:
            print(f"  ⚠️ 파일이 없습니다: {file_path}")
    
    if not all_chunks:
        print("\n❌ 처리할 청크가 없습니다!")
        return
    
    print(f"\n📊 총 {len(all_chunks)}개 청크를 처리합니다.")
    
    # ChromaDB에 인덱싱
    retriever = index_chunks_to_chroma(all_chunks, clear_existing=True)
    
    # 검색 기능 테스트
    test_search_functionality(retriever)
    
    print("\n" + "=" * 50)
    print("🎉 ChromaDB 인덱싱 완료!")
    print("=" * 50)
    print("\n💡 다음 단계:")
    print("  1. RAG 챗봇에서 retriever.hybrid_search() 사용")
    print("  2. 품질 필터링: retriever.quality_filtered_search()")
    print("  3. 우선순위 검색: retriever.search_by_priority()")


if __name__ == "__main__":
    main()
