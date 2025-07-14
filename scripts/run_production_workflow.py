# 실제 운영용 RAG 시스템 구축 워크플로우
"""
실험 결과를 바탕으로 최적의 청킹 방식을 선택하여
실제 운영용 RAG 시스템을 구축하는 스크립트

워크플로우:
1. 실험 결과 분석 (최적 청킹 방식 선택)
2. 선택된 방식으로 청크 생성
3. ChromaDB 인덱싱
4. RAG 챗봇 테스트
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.parse_pdf import parse_pdf_to_chunks
from scripts.parse_excel import parse_excel_file
from rag.retriever import EnhancedRetriever
from rag.chatbot import RAGChatbot


class ProductionWorkflow:
    """실제 운영용 RAG 시스템 구축 워크플로우"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.best_chunking_method = None
        self.experiment_results = None
        
    def load_experiment_results(self):
        """실험 결과 로드 및 최적 청킹 방식 선택"""
        experiment_file = self.output_dir / "chunking_experiment_results.json"
        
        if not experiment_file.exists():
            print("⚠️ 실험 결과 파일이 없습니다. 기본값(adaptive)을 사용합니다.")
            self.best_chunking_method = "adaptive"
            return
        
        print("📊 실험 결과 분석 중...")
        with open(experiment_file, 'r', encoding='utf-8') as f:
            self.experiment_results = json.load(f)
        
        # 최적 청킹 방식 선택 (평균 점수 기준)
        valid_methods = []
        for method, result in self.experiment_results.items():
            if "error" not in result and "rag_performance" in result:
                valid_methods.append({
                    "method": method,
                    "avg_score": result["rag_performance"]["avg_score"],
                    "avg_search_time": result["rag_performance"]["avg_search_time"],
                    "chunk_count": result["chunk_count"]
                })
        
        if valid_methods:
            # 점수와 속도를 고려한 종합 점수 계산
            for method_data in valid_methods:
                # 점수는 높을수록 좋고, 검색 시간은 낮을수록 좋음
                method_data["composite_score"] = (
                    method_data["avg_score"] * 0.7 +  # 점수 가중치 70%
                    (1.0 - min(method_data["avg_search_time"], 1.0)) * 0.3  # 속도 가중치 30%
                )
            
            # 종합 점수가 가장 높은 방식 선택
            best_method = max(valid_methods, key=lambda x: x["composite_score"])
            self.best_chunking_method = best_method["method"]
            
            print(f"🏆 최적 청킹 방식 선택: {self.best_chunking_method.upper()}")
            print(f"  • 평균 점수: {best_method['avg_score']:.3f}")
            print(f"  • 평균 검색 시간: {best_method['avg_search_time']:.3f}초")
            print(f"  • 청크 수: {best_method['chunk_count']}개")
            print(f"  • 종합 점수: {best_method['composite_score']:.3f}")
        else:
            print("⚠️ 유효한 실험 결과가 없습니다. 기본값(adaptive)을 사용합니다.")
            self.best_chunking_method = "adaptive"
    
    def generate_production_chunks(self):
        """선택된 방식으로 실제 운영용 청크 생성"""
        print(f"\n🚀 실제 운영용 청크 생성 시작 ({self.best_chunking_method.upper()} 방식)")
        print("=" * 60)
        
        # PDF 청크 생성
        pdf_file = "./data/raw/DR_스마트야드개론(데모용).pdf"
        pdf_output = self.output_dir / "DR_스마트야드개론(데모용)_chunks.json"
        
        print(f"📄 PDF 처리 중: {pdf_file}")
        pdf_chunks = parse_pdf_to_chunks(
            pdf_path=pdf_file,
            output_path=str(pdf_output),
            document_id="smart_yard_intro_production",
            chunking_method=self.best_chunking_method
        )
        print(f"  ✓ PDF 청크 생성 완료: {len(pdf_chunks)}개")
        
        # Excel 청크 생성
        excel_file = "./data/raw/DR_공정회의자료_추출본(데모용).xlsx"
        excel_output = self.output_dir / "DR_공정회의자료_추출본(데모용)_chunks.json"
        
        print(f"📊 Excel 처리 중: {excel_file}")
        excel_chunks = parse_excel_file(
            filepath=excel_file,
            output_path=str(excel_output)
        )
        print(f"  ✓ Excel 청크 생성 완료: {len(excel_chunks)}개")
        
        return {
            "pdf_chunks": pdf_chunks,
            "excel_chunks": excel_chunks,
            "total_chunks": len(pdf_chunks) + len(excel_chunks)
        }
    
    def build_rag_system(self, chunks_info: Dict[str, Any]):
        """RAG 시스템 구축 (ChromaDB 인덱싱)"""
        print(f"\n🔧 RAG 시스템 구축 시작")
        print("=" * 60)
        
        # 모든 청크 합치기
        all_chunks = chunks_info["pdf_chunks"] + chunks_info["excel_chunks"]
        
        # ChromaDB 인덱싱
        retriever = EnhancedRetriever()
        retriever.clear_collection()
        
        print(f"📥 ChromaDB 인덱싱 중... ({len(all_chunks)}개 청크)")
        retriever.add_chunks(all_chunks)
        
        # 통계 정보 출력
        stats = retriever.get_collection_stats()
        print(f"  ✓ 인덱싱 완료! 통계: {stats}")
        
        return retriever
    
    def test_rag_chatbot(self, retriever: EnhancedRetriever):
        """RAG 챗봇 테스트"""
        print(f"\n🤖 RAG 챗봇 테스트 시작")
        print("=" * 60)
        
        chatbot = RAGChatbot(retriever)
        
        test_queries = [
            "스마트 야드란 무엇인가요?",
            "조선 산업의 주요 도전 과제는?",
            "AI 기술이 조선소에서 어떻게 활용되나요?",
            "생산성 향상을 위한 방법은?",
            "자동화 시스템의 효과는?"
        ]
        
        print("📝 테스트 쿼리 실행:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. 쿼리: '{query}'")
            try:
                result = chatbot.chat(query)
                response = result['response']  # 딕셔너리에서 response 키 추출
                print(f"   응답: {response[:200]}...")
            except Exception as e:
                print(f"   ❌ 오류: {e}")
        
        print(f"\n✅ RAG 챗봇 테스트 완료")
    
    def save_production_config(self, chunks_info: Dict[str, Any]):
        """운영 설정 저장"""
        config = {
            "chunking_method": self.best_chunking_method,
            "pdf_chunks_count": len(chunks_info["pdf_chunks"]),
            "excel_chunks_count": len(chunks_info["excel_chunks"]),
            "total_chunks": chunks_info["total_chunks"],
            "files": {
                "pdf_chunks": "DR_스마트야드개론(데모용)_chunks.json",
                "excel_chunks": "DR_공정회의자료_추출본(데모용)_chunks.json"
            }
        }
        
        config_path = self.output_dir / "production_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 운영 설정 저장: {config_path}")
        return config
    
    def run_full_workflow(self):
        """전체 운영 워크플로우 실행"""
        print("🚀 실제 운영용 RAG 시스템 구축 시작")
        print("=" * 60)
        
        # 1. 실험 결과 분석 및 최적 방식 선택
        self.load_experiment_results()
        
        # 2. 실제 운영용 청크 생성
        chunks_info = self.generate_production_chunks()
        
        # 3. RAG 시스템 구축
        retriever = self.build_rag_system(chunks_info)
        
        # 4. RAG 챗봇 테스트
        self.test_rag_chatbot(retriever)
        
        # 5. 운영 설정 저장
        config = self.save_production_config(chunks_info)
        
        print(f"\n🎉 운영용 RAG 시스템 구축 완료!")
        print("=" * 60)
        print(f"📋 구축 요약:")
        print(f"  • 선택된 청킹 방식: {self.best_chunking_method.upper()}")
        print(f"  • 총 청크 수: {chunks_info['total_chunks']}개")
        print(f"  • PDF 청크: {len(chunks_info['pdf_chunks'])}개")
        print(f"  • Excel 청크: {len(chunks_info['excel_chunks'])}개")
        print(f"  • 설정 파일: data/processed/production_config.json")
        
        return {
            "retriever": retriever,
            "config": config,
            "chunks_info": chunks_info
        }


def main():
    """메인 실행 함수"""
    workflow = ProductionWorkflow()
    result = workflow.run_full_workflow()
    
    print(f"\n💡 다음 단계:")
    print(f"  1. RAG 챗봇 사용: result['retriever']로 검색")
    print(f"  2. 설정 확인: data/processed/production_config.json")
    print(f"  3. 청크 파일: data/processed/DR_*_chunks.json")


if __name__ == "__main__":
    main() 