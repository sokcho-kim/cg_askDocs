# RAG 챗봇 테스트 스크립트
"""
개선된 RAG 챗봇을 테스트하는 스크립트
실제 운영 환경에서 챗봇이 어떻게 동작하는지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rag.retriever import EnhancedRetriever
from rag.chatbot import RAGChatbot


def test_rag_chatbot():
    """RAG 챗봇 테스트"""
    print("🤖 RAG 챗봇 테스트 시작")
    print("=" * 60)
    
    # Retriever 초기화 (기존 인덱스 사용)
    retriever = EnhancedRetriever()
    
    # 컬렉션 상태 확인
    stats = retriever.get_collection_stats()
    print(f"📊 현재 인덱스 상태: {stats}")
    
    # 챗봇 초기화
    chatbot = RAGChatbot(retriever)
    
    # 테스트 질문들
    test_queries = [
        "현재 공정 지연 위험이 있는 항목은?",
        "2452주차에 발생한 이슈는?",
        "밸브재 관련 문제점은?",
        "협력사 생산 지연에 대한 대응방안은?",
        "스마트 야드란 무엇인가요?"
    ]
    
    print("\n📝 챗봇 테스트:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 질문: {query}")
        print("-" * 40)
        
        try:
            # 챗봇 응답
            result = chatbot.chat(query, search_method="enhanced", n_results=3)
            
            print(f"🔍 검색된 컨텍스트: {result['contexts_found']}개")
            print(f"🤖 응답:")
            print(result['response'])
            
            # 상위 컨텍스트 정보
            if result['contexts']:
                top_context = result['contexts'][0]
                print(f"\n🏆 최고 점수 컨텍스트:")
                print(f"   ID: {top_context.get('chunk_id', 'unknown')}")
                print(f"   점수: {top_context.get('final_score', 0):.3f}")
                print(f"   내용 미리보기: {top_context.get('content', '')[:100]}...")
            
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    print(f"\n🎉 챗봇 테스트 완료!")
    print("=" * 60)


def interactive_chat():
    """대화형 챗봇"""
    print("💬 대화형 RAG 챗봇 시작")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("=" * 60)
    
    retriever = EnhancedRetriever()
    chatbot = RAGChatbot(retriever)
    
    while True:
        try:
            query = input("\n질문: ").strip()
            
            if query.lower() in ['quit', 'exit', '종료']:
                print("👋 챗봇을 종료합니다.")
                break
            
            if not query:
                continue
            
            # 챗봇 응답
            result = chatbot.chat(query, search_method="enhanced", n_results=3)
            
            print(f"\n🤖 답변:")
            print(result['response'])
            
            print(f"\n📊 검색 정보: {result['contexts_found']}개 컨텍스트 발견")
            
        except KeyboardInterrupt:
            print("\n👋 챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 챗봇 테스트")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="대화형 모드로 실행")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_chat()
    else:
        test_rag_chatbot() 