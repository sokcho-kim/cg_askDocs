"""
RAG 최적화 기능 테스트 스크립트
통일된 청크 형식, 품질 점수, 키워드 추출, 향상된 검색 기능을 테스트합니다.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.chunk_processor import ChunkProcessor, ExcelChunkProcessor
from rag.retriever import EnhancedRetriever


def test_chunk_processor():
    """청크 프로세서 기능 테스트"""
    print("=" * 50)
    print("🧪 청크 프로세서 테스트")
    print("=" * 50)
    
    # 테스트용 프로세서 생성
    processor = ChunkProcessor(document_id="test_doc")
    
    # 테스트 텍스트
    test_text = "스마트 야드는 4차 산업혁명 기술을 융합하여 지능화된 생산 환경을 구축합니다. AI, IoT, 빅데이터 기술이 핵심입니다."
    
    print(f"📝 테스트 텍스트: {test_text}")
    
    # 키워드 추출 테스트
    keywords = processor.extract_keywords(test_text)
    print(f"🔑 추출된 키워드: {keywords}")
    
    # 품질 점수 계산 테스트
    quality_score = processor.calculate_chunk_quality_score(test_text, "text")
    print(f"📊 품질 점수: {quality_score:.3f}")
    
    # 청크 생성 테스트
    chunk = processor.create_text_chunk(test_text, "page:1")
    print(f"📦 생성된 청크:")
    print(f"  - ID: {chunk['chunk_id']}")
    print(f"  - 타입: {chunk['chunk_type']}")
    print(f"  - 품질 점수: {chunk['metadata']['quality_score']:.3f}")
    print(f"  - 키워드: {chunk['metadata']['keywords']}")
    print(f"  - 우선순위: {chunk['metadata']['search_priority']}")
    
    return chunk


def test_excel_processor():
    """Excel 프로세서 기능 테스트"""
    print("\n" + "=" * 50)
    print("📊 Excel 프로세서 테스트")
    print("=" * 50)
    
    # 테스트용 Excel 데이터
    test_data = [
        {
            "주차": "2452",
            "대분류": "사외",
            "팀": "사외공정관리팀",
            "이슈": "사외블록 2개 입고 지연",
            "리스크": "공정 지연"
        },
        {
            "주차": "2451",
            "대분류": "자재",
            "팀": "배관재구매팀",
            "이슈": "밸브재 보급충족률 저조",
            "리스크": "공정 지연"
        }
    ]
    
    # Excel 프로세서 생성
    processor = ExcelChunkProcessor(document_id="test_excel")
    
    # Excel 데이터 처리
    chunks = processor.process_excel_data(test_data, "Sheet1")
    
    print(f"📋 처리된 Excel 청크 수: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📦 Excel 청크 {i}:")
        print(f"  - ID: {chunk['chunk_id']}")
        print(f"  - 위치: {chunk['location']}")
        print(f"  - 내용: {chunk['content'][:100]}...")
        print(f"  - 품질 점수: {chunk['metadata']['quality_score']:.3f}")
        print(f"  - 키워드: {chunk['metadata']['keywords'][:5]}")
    
    return chunks


def test_enhanced_retriever():
    """향상된 검색 기능 테스트"""
    print("\n" + "=" * 50)
    print("🔍 향상된 검색 기능 테스트")
    print("=" * 50)
    
    # Retriever 초기화
    retriever = EnhancedRetriever()
    
    # 테스트용 청크들 생성
    test_chunks = [
        {
            "chunk_id": "test_1",
            "content": "스마트 야드는 4차 산업혁명 기술을 융합한 지능화된 조선소입니다.",
            "metadata": {
                "quality_score": 0.8,
                "keywords": ["스마트", "야드", "4차", "산업혁명", "기술", "지능화", "조선소"],
                "search_priority": "high"
            }
        },
        {
            "chunk_id": "test_2", 
            "content": "AI와 IoT 기술을 활용하여 생산성을 향상시킵니다.",
            "metadata": {
                "quality_score": 0.7,
                "keywords": ["AI", "IoT", "기술", "생산성", "향상"],
                "search_priority": "medium"
            }
        },
        {
            "chunk_id": "test_3",
            "content": "자동화 시스템으로 안전성을 강화합니다.",
            "metadata": {
                "quality_score": 0.6,
                "keywords": ["자동화", "시스템", "안전성", "강화"],
                "search_priority": "medium"
            }
        }
    ]
    
    # 청크들을 벡터 DB에 추가
    print("📥 테스트 청크들을 벡터 DB에 추가 중...")
    retriever.add_chunks(test_chunks)
    
    # 검색 테스트
    test_queries = [
        "스마트 야드 자동화",
        "AI 기술 활용",
        "생산성 향상"
    ]
    
    for query in test_queries:
        print(f"\n🔍 검색 쿼리: '{query}'")
        
        # 하이브리드 검색
        results = retriever.hybrid_search(query, 2)
        print(f"  하이브리드 검색 결과:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['chunk_id']} (점수: {result['final_score']:.3f})")
        
        # 품질 필터링 검색
        results = retriever.quality_filtered_search(query, 2, min_quality_score=0.6)
        print(f"  품질 필터링 검색 결과:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['chunk_id']} (품질: {result['metadata']['quality_score']:.2f})")
    
    # 통계 정보
    stats = retriever.get_collection_stats()
    print(f"\n📊 컬렉션 통계: {stats}")
    
    return retriever


def test_format_unification():
    """형식 통일 테스트"""
    print("\n" + "=" * 50)
    print("🔄 형식 통일 테스트")
    print("=" * 50)
    
    # 기존 형식의 청크들 (PDF와 Excel)
    pdf_chunk = {
        "chunk_id": "smart_yard_intro_text_0",
        "document_id": "smart_yard_intro",
        "chunk_index": 0,
        "chunk_type": "text",
        "location": "page:1",
        "content": "Part １조선 산업의 현황 및 도전 과제...",
        "embedding": None,
        "metadata": {
            "length": 361,
            "chunk_in_content": 0,
            "page": 1
        }
    }
    
    excel_chunk = {
        "chunk_id": "bfc7d58f-4bba-43da-9fc9-7e1379ce0a75",
        "row_index": 0,
        "column": "주차",
        "content": "주차: 2452",
        "metadata": {
            "source_file": "{df}",
            "length": 8
        }
    }
    
    print("📄 기존 PDF 청크 형식:")
    print(f"  - 필드 수: {len(pdf_chunk)}")
    print(f"  - 필드: {list(pdf_chunk.keys())}")
    
    print("\n📊 기존 Excel 청크 형식:")
    print(f"  - 필드 수: {len(excel_chunk)}")
    print(f"  - 필드: {list(excel_chunk.keys())}")
    
    # 통일된 형식으로 변환
    processor = ChunkProcessor(document_id="unified_test")
    
    # PDF 청크를 통일된 형식으로 변환
    unified_pdf_chunk = processor.create_text_chunk(
        pdf_chunk["content"], 
        pdf_chunk["location"],
        pdf_chunk["metadata"]
    )
    
    # Excel 청크를 통일된 형식으로 변환
    excel_data = {excel_chunk["column"]: excel_chunk["content"].split(": ")[1]}
    unified_excel_chunk = processor.create_excel_row_chunk(
        excel_data, 
        excel_chunk["row_index"]
    )
    
    print("\n🔄 통일된 형식:")
    print(f"  - PDF 청크 필드: {list(unified_pdf_chunk.keys())}")
    print(f"  - Excel 청크 필드: {list(unified_excel_chunk.keys())}")
    
    # 공통 필드 확인
    pdf_fields = set(unified_pdf_chunk.keys())
    excel_fields = set(unified_excel_chunk.keys())
    common_fields = pdf_fields & excel_fields
    
    print(f"\n✅ 공통 필드 ({len(common_fields)}개): {sorted(common_fields)}")
    
    return unified_pdf_chunk, unified_excel_chunk


def main():
    """메인 테스트 함수"""
    print("🚀 RAG 최적화 기능 통합 테스트 시작")
    print("=" * 60)
    
    try:
        # 1. 청크 프로세서 테스트
        test_chunk = test_chunk_processor()
        
        # 2. Excel 프로세서 테스트
        excel_chunks = test_excel_processor()
        
        # 3. 향상된 검색 기능 테스트
        retriever = test_enhanced_retriever()
        
        # 4. 형식 통일 테스트
        unified_chunks = test_format_unification()
        
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 완료!")
        print("=" * 60)
        
        # 결과 요약
        print("\n📋 테스트 결과 요약:")
        print(f"  ✅ 청크 프로세서: 품질 점수 {test_chunk['metadata']['quality_score']:.3f}")
        print(f"  ✅ Excel 프로세서: {len(excel_chunks)}개 청크 생성")
        print(f"  ✅ 향상된 검색: 다양한 검색 방법 지원")
        print(f"  ✅ 형식 통일: PDF/Excel 청크 형식 통합")
        
        print("\n💡 RAG 최적화 아이디어:")
        print("  1. 통일된 청크 형식으로 검색 일관성 향상")
        print("  2. 품질 점수 기반 검색 우선순위 설정")
        print("  3. 키워드 추출로 정확한 매칭 강화")
        print("  4. 하이브리드 검색으로 의미적+키워드 검색 결합")
        print("  5. 메타데이터 활용한 고급 필터링")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 