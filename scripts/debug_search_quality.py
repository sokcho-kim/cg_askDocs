#!/usr/bin/env python3
"""
검색 품질 디버깅 스크립트
특정 질문에 대한 검색 결과를 자세히 분석
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag.retriever import EnhancedRetriever
from rag.chatbot import RAGChatbot


def debug_search_quality(query: str):
    """특정 질문의 검색 품질을 디버깅합니다."""
    print(f"🔍 검색 품질 디버깅: '{query}'")
    print("=" * 60)
    
    retriever = EnhancedRetriever()
    chatbot = RAGChatbot(retriever)
    
    # 1. 키워드 검색 테스트
    print("\n1️⃣ 키워드 검색 결과:")
    keyword_results = retriever.keyword_search(query, 5)
    for i, result in enumerate(keyword_results, 1):
        score = result.get('score', 0)
        chunk_id = result.get('chunk_id', '')
        content = result.get('content', '')[:100] + "..."
        doc_type = "Excel" if 'excel' in chunk_id.lower() else "PDF"
        print(f"   {i}. {chunk_id} ({doc_type}, 점수: {score:.3f})")
        print(f"      내용: {content}")
    
    # 2. 하이브리드 검색 테스트
    print("\n2️⃣ 하이브리드 검색 결과:")
    hybrid_results = retriever.hybrid_search(query, 5)
    for i, result in enumerate(hybrid_results, 1):
        score = result.get('final_score', result.get('score', 0))
        chunk_id = result.get('chunk_id', '')
        content = result.get('content', '')[:100] + "..."
        doc_type = "Excel" if 'excel' in chunk_id.lower() else "PDF"
        print(f"   {i}. {chunk_id} ({doc_type}, 점수: {score:.3f})")
        print(f"      내용: {content}")
    
    # 3. 향상된 검색 테스트
    print("\n3️⃣ 향상된 검색 결과:")
    enhanced_results = chatbot._enhanced_search(query, 5)
    for i, result in enumerate(enhanced_results, 1):
        score = result.get('score', 0)
        chunk_id = result.get('chunk_id', '')
        content = result.get('content', '')[:100] + "..."
        source = result.get('source', 'unknown')
        doc_type = "Excel" if 'excel' in chunk_id.lower() else "PDF"
        print(f"   {i}. {chunk_id} ({doc_type}, 점수: {score:.3f}, 소스: {source})")
        print(f"      내용: {content}")
    
    # 4. 챗봇 응답 테스트
    print("\n4️⃣ 챗봇 응답:")
    chat_result = chatbot.chat(query, search_method="enhanced", n_results=3)
    print(f"   응답: {chat_result['response']}")
    print(f"   검색된 컨텍스트: {chat_result['contexts_found']}개")


def main():
    """메인 함수"""
    test_queries = [
        "사외공정관리팀이 담당하는 주요 업무는?",
        "2452주차에 발생한 이슈는?",
        "밸브재 보급충족률 저조 문제의 원인은?"
    ]
    
    for query in test_queries:
        debug_search_quality(query)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main() 