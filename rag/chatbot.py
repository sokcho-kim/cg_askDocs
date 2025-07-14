"""
RAG 챗봇 구현
향상된 검색 기능(하이브리드, 품질 필터링, 우선순위)을 활용한 챗봇
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag.retriever import EnhancedRetriever


class RAGChatbot:
    """향상된 검색 기능을 활용하는 RAG 챗봇"""
    
    def __init__(self, retriever: Optional[EnhancedRetriever] = None):
        """
        Args:
            retriever: EnhancedRetriever 인스턴스 (None이면 자동 생성)
        """
        self.retriever = retriever or EnhancedRetriever()
        self.conversation_history = []
        
    def search_context(self, query: str, search_method: str = "hybrid", **kwargs) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 컨텍스트를 검색합니다.
        
        Args:
            query: 사용자 질문
            search_method: 검색 방법
                - "hybrid": 하이브리드 검색 (기본)
                - "semantic": 의미적 검색만
                - "keyword": 키워드 검색만
                - "quality": 품질 필터링 검색
                - "priority": 우선순위 검색
            **kwargs: 검색 파라미터 (n_results, min_quality_score 등)
        
        Returns:
            검색된 컨텍스트 리스트
        """
        n_results = kwargs.get('n_results', 5)
        
        if search_method == "hybrid":
            return self.retriever.hybrid_search(query, n_results)
        elif search_method == "semantic":
            return self.retriever.semantic_search(query, n_results)
        elif search_method == "keyword":
            return self.retriever.keyword_search(query, n_results)
        elif search_method == "quality":
            min_quality = kwargs.get('min_quality_score', 0.5)
            return self.retriever.quality_filtered_search(query, n_results, min_quality)
        elif search_method == "priority":
            return self.retriever.search_by_priority(query, n_results)
        else:
            raise ValueError(f"지원하지 않는 검색 방법: {search_method}")
    
    def format_context_for_llm(self, contexts: List[Dict[str, Any]]) -> str:
        """검색된 컨텍스트를 LLM 입력용으로 포맷팅합니다."""
        if not contexts:
            return "관련 정보를 찾을 수 없습니다."
        
        formatted_contexts = []
        for i, context in enumerate(contexts, 1):
            chunk_id = context.get('chunk_id', 'unknown')
            content = context.get('content', '')
            score = context.get('final_score', context.get('score', 0))
            
            # 내용이 너무 길면 자르기
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted_contexts.append(f"[{i}] 청크 ID: {chunk_id} (점수: {score:.3f})\n{content}")
        
        return "\n\n".join(formatted_contexts)
    
    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """컨텍스트를 바탕으로 응답을 생성합니다 (LLM 시뮬레이션)"""
        if not contexts:
            return "죄송합니다. 질문에 대한 관련 정보를 찾을 수 없습니다."
        
        # 간단한 응답 생성 (실제로는 LLM 사용)
        context_summary = self.format_context_for_llm(contexts)
        
        response = f"""질문: {query}

찾은 관련 정보:
{context_summary}

응답: 위 정보를 바탕으로 답변드리겠습니다.

"""
        
        # 컨텍스트 내용에 따른 간단한 응답 생성
        if "스마트 야드" in query.lower():
            response += "스마트 야드는 4차 산업혁명 기술을 융합한 지능화된 조선소입니다. "
            if "자동화" in query.lower():
                response += "자동화를 통해 생산성을 향상시키고 안전성을 강화합니다."
            elif "기술" in query.lower():
                response += "AI, IoT, 빅데이터, 로봇, 디지털 트윈 등의 핵심 기술을 활용합니다."
        elif "조선소" in query.lower():
            response += "조선소는 선박을 건조하는 시설로, 스마트 야드 기술로 혁신하고 있습니다."
        else:
            response += "관련 정보를 확인했습니다. 더 구체적인 질문이 있으시면 말씀해 주세요."
        
        return response
    
    def chat(self, query: str, search_method: str = "hybrid", **kwargs) -> Dict[str, Any]:
        """
        챗봇과 대화합니다.
        
        Args:
            query: 사용자 질문
            search_method: 검색 방법
            **kwargs: 검색 파라미터
        
        Returns:
            응답 정보 딕셔너리
        """
        # 1. 컨텍스트 검색
        contexts = self.search_context(query, search_method, **kwargs)
        
        # 2. 응답 생성
        response = self.generate_response(query, contexts)
        
        # 3. 대화 기록 저장
        conversation_entry = {
            "query": query,
            "search_method": search_method,
            "contexts_found": len(contexts),
            "response": response,
            "contexts": contexts
        }
        self.conversation_history.append(conversation_entry)
        
        return {
            "query": query,
            "response": response,
            "contexts_found": len(contexts),
            "search_method": search_method,
            "contexts": contexts
        }
    
    def compare_search_methods(self, query: str) -> Dict[str, Any]:
        """다양한 검색 방법을 비교합니다."""
        methods = ["hybrid", "semantic", "keyword", "quality", "priority"]
        results = {}
        
        for method in methods:
            try:
                contexts = self.search_context(query, method, n_results=3)
                results[method] = {
                    "contexts_found": len(contexts),
                    "avg_score": sum(ctx.get('final_score', ctx.get('score', 0)) for ctx in contexts) / len(contexts) if contexts else 0,
                    "top_context": contexts[0] if contexts else None
                }
            except Exception as e:
                results[method] = {"error": str(e)}
        
        return results
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """대화 기록을 반환합니다."""
        return self.conversation_history
    
    def clear_history(self):
        """대화 기록을 초기화합니다."""
        self.conversation_history = []


def test_rag_chatbot():
    """RAG 챗봇 테스트"""
    print("🤖 RAG 챗봇 테스트 시작")
    print("=" * 50)
    
    # 챗봇 초기화
    chatbot = RAGChatbot()
    
    # 테스트 질문들
    test_queries = [
        "스마트 야드란 무엇인가요?",
        "자동화 기술에 대해 알려주세요",
        "AI 기술이 어떻게 활용되나요?",
        "조선소의 생산성 향상 방법은?",
        "스마트 야드의 주요 기술 요소는?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 질문 {i}: {query}")
        print("-" * 40)
        
        # 하이브리드 검색으로 응답
        result = chatbot.chat(query, search_method="hybrid", n_results=3)
        
        print(f"📊 검색 결과: {result['contexts_found']}개 컨텍스트")
        print(f"🤖 응답: {result['response'][:200]}...")
        
        # 상위 컨텍스트 정보
        if result['contexts']:
            top_context = result['contexts'][0]
            print(f"🏆 최고 점수 컨텍스트: {top_context.get('chunk_id', 'unknown')} (점수: {top_context.get('final_score', 0):.3f})")
    
    # 검색 방법 비교
    print(f"\n🔍 검색 방법 비교: '스마트 야드 자동화'")
    print("=" * 50)
    
    comparison = chatbot.compare_search_methods("스마트 야드 자동화")
    for method, result in comparison.items():
        if "error" in result:
            print(f"  {method.upper()}: ❌ {result['error']}")
        else:
            print(f"  {method.upper()}: {result['contexts_found']}개 컨텍스트, 평균 점수 {result['avg_score']:.3f}")
    
    # 대화 기록 요약
    history = chatbot.get_conversation_history()
    print(f"\n📋 대화 기록: {len(history)}개 질문")
    
    return chatbot


if __name__ == "__main__":
    chatbot = test_rag_chatbot()
