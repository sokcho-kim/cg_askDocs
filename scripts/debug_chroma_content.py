#!/usr/bin/env python3
"""
ChromaDB 인덱싱 상태 디버그 스크립트
현재 벡터 DB에 어떤 데이터가 저장되어 있는지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag.retriever import EnhancedRetriever


def debug_chroma_content():
    """ChromaDB 내용을 디버그합니다."""
    print("🔍 ChromaDB 내용 디버그")
    print("=" * 50)
    
    retriever = EnhancedRetriever()
    
    # 컬렉션 통계
    stats = retriever.get_collection_stats()
    print(f"📊 컬렉션 통계: {stats}")
    
    # 모든 문서 가져오기
    all_docs = retriever.collection.get()
    
    if not all_docs or not all_docs.get('ids'):
        print("❌ ChromaDB에 데이터가 없습니다!")
        return
    
    ids = all_docs.get('ids', [])
    documents = all_docs.get('documents', [])
    metadatas = all_docs.get('metadatas', [])
    
    print(f"\n📄 총 {len(ids)}개 문서가 인덱싱되어 있습니다.")
    
    # 문서 타입별 분류
    excel_count = 0
    pdf_count = 0
    other_count = 0
    
    excel_samples = []
    pdf_samples = []
    
    for i, doc_id in enumerate(ids):
        content = documents[i] if documents and i < len(documents) else ""
        metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
        
        # 문서 타입 판별
        if 'excel' in doc_id.lower() or metadata.get('data_type') == 'excel_row':
            excel_count += 1
            if len(excel_samples) < 3:
                excel_samples.append({
                    'id': doc_id,
                    'content': content[:200] + "..." if len(content) > 200 else content,
                    'metadata': metadata
                })
        elif 'pdf' in doc_id.lower() or metadata.get('data_type') in ['text', 'section']:
            pdf_count += 1
            if len(pdf_samples) < 3:
                pdf_samples.append({
                    'id': doc_id,
                    'content': content[:200] + "..." if len(content) > 200 else content,
                    'metadata': metadata
                })
        else:
            other_count += 1
    
    print(f"\n📊 문서 타입별 분류:")
    print(f"  • Excel 문서: {excel_count}개")
    print(f"  • PDF 문서: {pdf_count}개")
    print(f"  • 기타: {other_count}개")
    
    # 샘플 출력
    if excel_samples:
        print(f"\n📋 Excel 문서 샘플:")
        for i, sample in enumerate(excel_samples, 1):
            print(f"  {i}. ID: {sample['id']}")
            print(f"     내용: {sample['content']}")
            print(f"     메타데이터: {sample['metadata']}")
            print()
    
    if pdf_samples:
        print(f"\n📄 PDF 문서 샘플:")
        for i, sample in enumerate(pdf_samples, 1):
            print(f"  {i}. ID: {sample['id']}")
            print(f"     내용: {sample['content']}")
            print(f"     메타데이터: {sample['metadata']}")
            print()
    
    # 검색 테스트
    print("\n🔍 검색 테스트:")
    test_queries = [
        "공정 지연",
        "2452주차",
        "밸브재",
        "스마트 야드"
    ]
    
    for query in test_queries:
        print(f"\n  쿼리: '{query}'")
        
        # 하이브리드 검색
        results = retriever.hybrid_search(query, 3)
        if results:
            print(f"    하이브리드 검색 결과 ({len(results)}개):")
            for j, result in enumerate(results, 1):
                score = result.get('final_score', result.get('score', 0))
                doc_type = "Excel" if 'excel' in result['chunk_id'].lower() else "PDF"
                print(f"      {j}. {result['chunk_id']} ({doc_type}, 점수: {score:.3f})")
                print(f"         내용: {result['content'][:100]}...")
        else:
            print("    검색 결과 없음")


if __name__ == "__main__":
    debug_chroma_content() 