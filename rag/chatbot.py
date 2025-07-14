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
                - "enhanced": 향상된 검색 (엑셀 우선 + 하이브리드)
            **kwargs: 검색 파라미터 (n_results, min_quality_score 등)
        
        Returns:
            검색된 컨텍스트 리스트
        """
        n_results = kwargs.get('n_results', 5)
        
        if search_method == "enhanced":
            return self._enhanced_search(query, n_results)
        elif search_method == "hybrid":
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
    
    def _enhanced_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """향상된 검색: 엑셀 데이터 우선 + 하이브리드 검색"""
        
        # 1. 키워드 검색으로 엑셀 데이터 우선 찾기
        keyword_results = self.retriever.keyword_search(query, n_results * 2)
        
        # 2. 하이브리드 검색으로 전체 검색
        hybrid_results = self.retriever.hybrid_search(query, n_results * 2)
        
        # 3. 결과 통합 및 정렬
        combined_results = {}
        
        # 키워드 검색 결과 (엑셀 우선) - 가중치 높게
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id not in combined_results:
                combined_results[chunk_id] = {
                    'chunk_id': chunk_id,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'score': result.get('score', 0) * 2.0,  # 엑셀 데이터 가중치 대폭 증가
                    'source': 'keyword'
                }
        
        # 하이브리드 검색 결과 추가 (키워드 결과가 없을 때만)
        for result in hybrid_results:
            chunk_id = result['chunk_id']
            if chunk_id not in combined_results:  # 키워드 결과가 없을 때만 추가
                combined_results[chunk_id] = {
                    'chunk_id': chunk_id,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'score': result.get('final_score', result.get('score', 0)),
                    'source': 'hybrid'
                }
        
        # 점수 순으로 정렬
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results[:n_results]
    
    def format_context_for_llm(self, contexts: List[Dict[str, Any]]) -> str:
        """검색된 컨텍스트를 LLM 입력용으로 포맷팅합니다."""
        if not contexts:
            return "관련 정보를 찾을 수 없습니다."
        
        formatted_contexts = []
        for i, context in enumerate(contexts, 1):
            chunk_id = context.get('chunk_id', 'unknown')
            content = context.get('content', '')
            score = context.get('final_score', context.get('score', 0))
            metadata = context.get('metadata', {})
            
            # 문서 정보 추출
            document_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            location = metadata.get('location', 'unknown')
            title = metadata.get('title', '')
            
            # 내용이 너무 길면 자르기
            if len(content) > 400:
                content = content[:400] + "..."
            
            # 구조화된 컨텍스트 포맷
            context_info = f"[{i}] 📄 문서: {document_id}"
            if location != 'unknown':
                context_info += f" | 📍 위치: {location}"
            if title:
                context_info += f" | 📝 제목: {title}"
            context_info += f" | ⭐ 점수: {score:.3f}\n"
            context_info += f"💬 내용: {content}"
            
            formatted_contexts.append(context_info)
        
        return "\n\n".join(formatted_contexts)
    
    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """컨텍스트를 바탕으로 응답을 생성합니다 (LLM 시뮬레이션)"""
        if not contexts:
            return "죄송합니다. 질문에 대한 관련 정보를 찾을 수 없습니다."
        
        # 검색된 컨텍스트 분석
        excel_contents = []
        pdf_contents = []
        all_contents = []
        
        for ctx in contexts[:10]:  # 상위 10개 컨텍스트 사용
            content = ctx.get('content', '')
            chunk_id = ctx.get('chunk_id', '')
            metadata = ctx.get('metadata', {})
            
            if content and len(content.strip()) > 10:  # 의미있는 내용만
                all_contents.append({
                    'content': content.strip(),
                    'chunk_id': chunk_id,
                    'metadata': metadata,
                    'score': ctx.get('score', 0)
                })
                
                # 문서 타입별 분류 (chunk_id 기반)
                if 'excel' in chunk_id.lower():
                    excel_contents.append({
                        'content': content.strip(),
                        'chunk_id': chunk_id,
                        'metadata': metadata,
                        'score': ctx.get('score', 0)
                    })
                elif 'pdf' in chunk_id.lower() or 'smart_yard' in chunk_id.lower():
                    pdf_contents.append({
                        'content': content.strip(),
                        'chunk_id': chunk_id,
                        'metadata': metadata,
                        'score': ctx.get('score', 0)
                    })
        
        if not all_contents:
            return "관련 정보를 찾았지만 구체적인 내용이 부족합니다."
        
        # 질문 유형 분석
        query_lower = query.lower()
        
        # 공정 관련 질문 처리
        if any(keyword in query_lower for keyword in ["공정", "지연", "이슈", "주차", "위험", "밸브재", "협력사", "입고", "사외블록"]):
            response = "공정 관련 정보를 찾았습니다:\n\n"
            
            if excel_contents:
                # 엑셀 데이터 기반 답변
                response += "📊 공정 이슈 현황:\n"
                
                # 주차별 이슈 분석
                if "주차" in query_lower:
                    response += "주차별 이슈 현황:\n"
                    for item in excel_contents[:3]:
                        content = item['content']
                        if "주차:" in content:
                            response += f"• {content[:300]}...\n"
                    response += "\n"
                
                # 지연 관련 분석
                if "지연" in query_lower:
                    response += "공정 지연 관련 이슈:\n"
                    for item in excel_contents:
                        content = item['content']
                        if "지연" in content:
                            response += f"• {content[:300]}...\n"
                    response += "\n"
                
                # 이슈 분석
                if "이슈" in query_lower:
                    response += "주요 이슈 현황:\n"
                    for item in excel_contents[:3]:
                        content = item['content']
                        if "이슈:" in content:
                            response += f"• {content[:300]}...\n"
                    response += "\n"
                
                # 밸브재 관련
                if "밸브재" in query_lower:
                    response += "밸브재 관련 이슈:\n"
                    for item in excel_contents:
                        content = item['content']
                        if "밸브재" in content:
                            response += f"• {content[:300]}...\n"
                    response += "\n"
                
                # 협력사 관련
                if "협력사" in query_lower:
                    response += "협력사 관련 이슈:\n"
                    for item in excel_contents:
                        content = item['content']
                        if "협력사" in content:
                            response += f"• {content[:300]}...\n"
                    response += "\n"
                
                # 대응방안
                response += "대응방안:\n"
                for item in excel_contents:
                    content = item['content']
                    if "대응방안:" in content:
                        response += f"• {content[:300]}...\n"
                        break
                
            else:
                # PDF 데이터 기반 답변
                response += "관련 정보:\n"
                for item in all_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            
            # 출처 정보 추가
            sources = set()
            for item in all_contents[:3]:
                chunk_id = item['chunk_id']
                doc_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
                sources.add(doc_name)
            
            response += f"\n📍 출처: {', '.join(sources)}\n\n"
        
        # 스마트 야드 관련 질문
        elif "스마트 야드" in query_lower or "스마트야드" in query_lower:
            response = "스마트 야드에 대한 정보를 찾았습니다:\n\n"
            
            if pdf_contents:
                response += "📋 스마트 야드 정보:\n"
                for item in pdf_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            else:
                response += "관련 정보:\n"
                for item in all_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            
            # 출처 정보 추가
            sources = set()
            for item in all_contents[:3]:
                chunk_id = item['chunk_id']
                doc_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
                sources.add(doc_name)
            
            response += f"\n📍 출처: {', '.join(sources)}\n\n"
        
        # 기술 관련 질문
        elif any(keyword in query_lower for keyword in ["기술", "ai", "인공지능", "자동화", "iot"]):
            response = "기술 관련 정보를 찾았습니다:\n\n"
            
            if pdf_contents:
                response += "🔧 기술 정보:\n"
                for item in pdf_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            else:
                response += "관련 정보:\n"
                for item in all_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            
            # 출처 정보 추가
            sources = set()
            for item in all_contents[:3]:
                chunk_id = item['chunk_id']
                doc_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
                sources.add(doc_name)
            
            response += f"\n📍 출처: {', '.join(sources)}\n\n"
        
        # 일반적인 질문
        else:
            response = "질문에 대한 관련 정보를 찾았습니다:\n\n"
            
            # 가장 관련성 높은 컨텐츠 선택
            if excel_contents:
                response += "📊 데이터 기반 정보:\n"
                for item in excel_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            elif pdf_contents:
                response += "📋 문서 기반 정보:\n"
                for item in pdf_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            else:
                response += "관련 정보:\n"
                for item in all_contents[:3]:
                    response += f"• {item['content'][:300]}...\n"
            
            # 출처 정보 추가
            sources = set()
            for item in all_contents[:3]:
                chunk_id = item['chunk_id']
                doc_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
                sources.add(doc_name)
            
            response += f"\n📍 출처: {', '.join(sources)}\n\n"
        
        response += "더 구체적인 질문이 있으시면 말씀해 주세요."
        return response
    
    def chat(self, query: str, search_method: str = "enhanced", **kwargs) -> Dict[str, Any]:
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
