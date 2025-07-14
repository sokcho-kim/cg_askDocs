"""
RAG 챗봇 구현
향상된 검색 기능(하이브리드, 품질 필터링, 우선순위)을 활용한 챗봇
"""

import json
import os
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag.retriever import EnhancedRetriever

# --- smart_filter 함수 추가 ---
def is_duplicate(content, seen=None):
    if seen is None:
        seen = set()
    key = content.strip()[:50]
    if key in seen:
        return True
    seen.add(key)
    return False

def smart_filter(results, score_threshold=0.4, min_length=50):
    seen = set()
    filtered = []
    for r in results:
        if (
            r.get("score", 0) >= score_threshold
            and len(r.get("content", "")) >= min_length
            and not is_duplicate(r["content"], seen)
        ):
            filtered.append(r)
    return filtered

class RAGChatbot:
    """향상된 검색 기능을 활용하는 RAG 챗봇"""
    
    def __init__(self, retriever: Optional[EnhancedRetriever] = None, 
                 gemma_api_url: Optional[str] = None, 
                 gemma_model: str = "google/gemma-3-12b-it"):
        """
        Args:
            retriever: EnhancedRetriever 인스턴스 (None이면 자동 생성)
            gemma_api_url: Gemma API URL (None이면 환경변수에서 자동 로드)
            gemma_model: 사용할 Gemma 모델명
        """
        self.retriever = retriever or EnhancedRetriever()
        self.conversation_history = []
        
        # Gemma API 설정
        self.gemma_model = gemma_model
        if gemma_api_url:
            self.gemma_api_url = gemma_api_url
        else:
            # 환경변수에서 API URL 로드
            self.gemma_api_url = os.getenv('GEMMA_API_URL')
            if not self.gemma_api_url:
                print("Warning: GEMMA_API_URL 환경변수가 설정되지 않았습니다.")
                self.gemma_api_url = None
        
        # API 사용 가능 여부 확인
        self.llm_available = self.gemma_api_url is not None
        
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
        n_results = kwargs.get('n_results', 10)
        
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
    
    def format_search_results(self, contexts: List[Dict[str, Any]]) -> str:
        """검색 결과를 사용자 친화적으로 포맷팅합니다."""
        if not contexts:
            return "관련 정보를 찾을 수 없습니다."
        
        formatted_results = []
        for i, context in enumerate(contexts, 1):
            chunk_id = context.get('chunk_id', 'unknown')
            content = context.get('content', '')
            score = context.get('final_score', context.get('score', 0))
            metadata = context.get('metadata', {})
            
            # 문서 정보 추출
            document_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            location = metadata.get('location', 'unknown')
            title = metadata.get('title', '')
            
            # 결과 포맷팅
            result_info = f"[{i}] 📄 문서: {document_id}"
            if location != 'unknown':
                result_info += f" | 📍 위치: {location}"
            if title:
                result_info += f" | 📝 제목: {title}"
            result_info += f" | ⭐ 점수: {score:.3f}\n"
            result_info += f"💬 내용: {content}"
            
            formatted_results.append(result_info)
        
        return "\n\n".join(formatted_results)
    
    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """컨텍스트를 바탕으로 응답을 생성합니다."""
        filtered = smart_filter(contexts)
        if not filtered:
            return "관련 정보를 찾을 수 없습니다."
        answer = "질문에 가장 관련 있는 정보:\n"
        for ctx in filtered[:3]:
            chunk_id = ctx.get('chunk_id', 'unknown')
            metadata = ctx.get('metadata', {})
            doc_name = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            location = metadata.get('location', '')
            answer += f"[문서: {doc_name} | 위치: {location}]\n"
            answer += f"내용: {ctx['content'][:300]}...\n"
        answer += "\n더 구체적인 질문이 있으시면 말씀해 주세요."
        return answer
    
    def _build_llm_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """LLM용 프롬프트를 생성합니다."""
        if not contexts:
            return "관련 정보를 찾을 수 없습니다."
        
        # 컨텍스트 포맷팅
        context_strs = []
        for i, ctx in enumerate(contexts, 1):
            chunk_id = ctx.get('chunk_id', 'unknown')
            content = ctx.get('content', '')
            metadata = ctx.get('metadata', {})
            
            # 문서 정보 추출
            document_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            location = metadata.get('location', 'unknown')
            title = metadata.get('title', '')
            
            # 컨텍스트 정보 구성
            context_info = f"[{i}] 문서: {document_id}"
            if location != 'unknown':
                context_info += f" | 위치: {location}"
            if title:
                context_info += f" | 제목: {title}"
            
            context_strs.append(f"{context_info}\n내용: {content}")
        
        context_block = "\n\n".join(context_strs)
        
        # 프롬프트 구성
        prompt = f"""다음은 사용자의 질문과 관련된 문서 정보입니다.

---
{context_block}
---

위 정보를 참고하여 아래 질문에 답변하세요.

질문: {query}

답변 형식:
1. 질문에 대한 직접적인 답변을 제공하세요
2. 답변 후 "참고 정보:" 섹션에서 사용한 문서의 출처와 위치를 명시하세요
3. 답변은 한국어로 작성하세요

답변:"""
        
        return prompt
    
    def _call_gemma_api(self, prompt: str) -> str:
        """Gemma API를 호출하여 답변을 생성합니다."""
        if not self.gemma_api_url:
            raise Exception("Gemma API URL이 설정되지 않았습니다.")
        
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.gemma_model,
            "stream": False
        }
        
        try:
            response = requests.post(self.gemma_api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"API 응답 형식 오류: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Gemma API 호출 중 오류 발생: {e}")
            return None
        except Exception as e:
            print(f"예상치 못한 오류: {e}")
            return None
    
    def _generate_simple_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """단순한 검색 결과 기반 답변 생성"""
        if not contexts:
            return "죄송합니다. 질문에 대한 관련 정보를 찾을 수 없습니다."
        
        # 상위 3개 컨텍스트 사용
        top_contexts = contexts[:3]
        
        response = f"질문: {query}\n\n"
        response += "관련 정보를 찾았습니다:\n\n"
        
        for i, ctx in enumerate(top_contexts, 1):
            chunk_id = ctx.get('chunk_id', 'unknown')
            content = ctx.get('content', '')
            metadata = ctx.get('metadata', {})
            
            # 문서 정보 추출
            document_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            location = metadata.get('location', 'unknown')
            
            response += f"[{i}] 문서: {document_id}"
            if location != 'unknown':
                response += f" | 위치: {location}"
            response += f"\n내용: {content[:300]}...\n\n"
        
        # 출처 정보 추가
        sources = set()
        for ctx in top_contexts:
            chunk_id = ctx.get('chunk_id', 'unknown')
            document_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
            sources.add(document_id)
        
        response += f"참고 정보:\n출처: {', '.join(sources)}\n\n"
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
        print(f"🤖 응답: {result['response']}")
        
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
