# 청킹 방식별 실험 및 성능 비교 스크립트
"""
4가지 청킹 방식(page, block, section, adaptive)을 각각 테스트하고
RAG 검색 성능을 비교하는 실험 스크립트

실험 과정:
1. 각 청킹 방식으로 PDF 청크 생성
2. ChromaDB에 인덱싱
3. 동일한 쿼리로 검색 성능 테스트
4. 결과 비교 및 분석
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Literal

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.parse_pdf import parse_pdf_to_chunks, compare_chunking_methods
from rag.retriever import EnhancedRetriever


class ChunkingExperiment:
    """청킹 방식별 실험을 수행하는 클래스"""
    
    def __init__(self, pdf_path: str, output_dir: str = "data/processed"):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.methods: List[Literal["page", "block", "section", "adaptive"]] = ["page", "block", "section", "adaptive"]
        self.results = {}
        
    def generate_chunks_for_all_methods(self):
        """모든 청킹 방식으로 청크 파일 생성"""
        print("🔬 청킹 방식별 청크 생성 시작")
        print("=" * 60)
        
        for method in self.methods:
            print(f"\n📋 {method.upper()} 방식 처리 중...")
            output_path = self.output_dir / f"pdf_chunks_{method}.json"
            
            try:
                chunks = parse_pdf_to_chunks(
                    pdf_path=self.pdf_path,
                    output_path=str(output_path),
                    document_id=f"smart_yard_intro_{method}",
                    chunking_method=method
                )
                
                self.results[method] = {
                    "chunk_count": len(chunks),
                    "avg_length": sum(len(chunk.get("content", "")) for chunk in chunks) // len(chunks) if chunks else 0,
                    "file_path": str(output_path),
                    "chunks": chunks
                }
                print(f"  ✓ {len(chunks)}개 청크 생성 완료")
                
            except Exception as e:
                print(f"  ❌ {method} 방식 실패: {e}")
                self.results[method] = {"error": str(e)}
    
    def test_rag_performance(self, test_queries: List[str]):
        """각 청킹 방식별 RAG 성능 테스트"""
        print(f"\n🔍 RAG 성능 테스트 시작 ({len(test_queries)}개 쿼리)")
        print("=" * 60)
        
        for method in self.methods:
            if "error" in self.results[method]:
                print(f"\n❌ {method.upper()}: 실패 - {self.results[method]['error']}")
                continue
                
            print(f"\n📊 {method.upper()} 방식 RAG 테스트 중...")
            
            try:
                # ChromaDB 초기화 및 인덱싱
                retriever = EnhancedRetriever()
                retriever.clear_collection()
                
                chunks = self.results[method]["chunks"]
                retriever.add_chunks(chunks)
                
                # 검색 성능 테스트
                search_results = {}
                total_time = 0
                
                for query in test_queries:
                    start_time = time.time()
                    results = retriever.hybrid_search(query, n_results=3)
                    end_time = time.time()
                    
                    search_time = end_time - start_time
                    total_time += search_time
                    
                    # 검색 결과 품질 평가
                    avg_score = sum(result.get('final_score', 0) for result in results) / len(results) if results else 0
                    content_diversity = len(set(result.get('chunk_type', '') for result in results))
                    
                    search_results[query] = {
                        "results_count": len(results),
                        "avg_score": avg_score,
                        "search_time": search_time,
                        "content_diversity": content_diversity,
                        "top_result": results[0] if results else None
                    }
                
                # 성능 지표 저장
                self.results[method]["rag_performance"] = {
                    "avg_search_time": total_time / len(test_queries),
                    "total_search_time": total_time,
                    "search_results": search_results,
                    "avg_score": sum(sr["avg_score"] for sr in search_results.values()) / len(search_results),
                    "avg_diversity": sum(sr["content_diversity"] for sr in search_results.values()) / len(search_results)
                }
                
                print(f"  ✓ 평균 검색 시간: {self.results[method]['rag_performance']['avg_search_time']:.3f}초")
                print(f"  ✓ 평균 점수: {self.results[method]['rag_performance']['avg_score']:.3f}")
                print(f"  ✓ 평균 다양성: {self.results[method]['rag_performance']['avg_diversity']:.1f}")
                
            except Exception as e:
                print(f"  ❌ RAG 테스트 실패: {e}")
                self.results[method]["rag_performance"] = {"error": str(e)}
    
    def generate_comparison_report(self):
        """실험 결과 비교 리포트 생성"""
        print(f"\n📈 실험 결과 비교 리포트")
        print("=" * 60)
        
        # 데이터 준비
        report_data = []
        for method in self.methods:
            if "error" in self.results[method]:
                continue
                
            data = {
                "방식": method.upper(),
                "청크 수": self.results[method]["chunk_count"],
                "평균 청크 길이": self.results[method]["avg_length"],
                "평균 검색 시간(초)": self.results[method]["rag_performance"]["avg_search_time"],
                "평균 점수": self.results[method]["rag_performance"]["avg_score"],
                "평균 다양성": self.results[method]["rag_performance"]["avg_diversity"]
            }
            report_data.append(data)
        
        # 표 형태로 출력
        print("\n📊 청킹 방식별 성능 비교:")
        print("-" * 80)
        print(f"{'방식':<12} {'청크 수':<8} {'평균 길이':<10} {'검색 시간':<12} {'평균 점수':<10} {'다양성':<8}")
        print("-" * 80)
        
        for data in report_data:
            print(f"{data['방식']:<12} {data['청크 수']:<8} {data['평균 청크 길이']:<10} "
                  f"{data['평균 검색 시간(초)']:<12.3f} {data['평균 점수']:<10.3f} {data['평균 다양성']:<8.1f}")
        
        # 최고 성능 방식 찾기
        if report_data:
            best_score = max(report_data, key=lambda x: x['평균 점수'])
            best_speed = min(report_data, key=lambda x: x['평균 검색 시간(초)'])
            best_diversity = max(report_data, key=lambda x: x['평균 다양성'])
            
            print(f"\n🏆 성능 분석 결과:")
            print(f"  • 최고 점수: {best_score['방식']} ({best_score['평균 점수']:.3f})")
            print(f"  • 최고 속도: {best_speed['방식']} ({best_speed['평균 검색 시간(초)']:.3f}초)")
            print(f"  • 최고 다양성: {best_diversity['방식']} ({best_diversity['평균 다양성']:.1f})")
        
        # 상세 결과를 JSON으로 저장
        report_path = self.output_dir / "chunking_experiment_results.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 상세 결과 저장: {report_path}")
        
        return report_data
    
    def run_full_experiment(self):
        """전체 실험 워크플로우 실행"""
        print("🚀 청킹 방식 실험 시작")
        print("=" * 60)
        
        # 1. 청크 생성
        self.generate_chunks_for_all_methods()
        
        # 2. RAG 성능 테스트
        test_queries = [
            "스마트 야드란 무엇인가요?",
            "조선 산업의 도전 과제는?",
            "AI 기술의 활용 방안은?",
            "생산성 향상 방법은?",
            "자동화 시스템의 효과는?"
        ]
        self.test_rag_performance(test_queries)
        
        # 3. 결과 리포트 생성
        report_data = self.generate_comparison_report()
        
        print(f"\n🎉 실험 완료!")
        print("=" * 60)
        
        return report_data


def main():
    """메인 실행 함수"""
    pdf_file = "./data/raw/DR_스마트야드개론(데모용).pdf"
    
    # 실험 실행
    experiment = ChunkingExperiment(pdf_file)
    results_data = experiment.run_full_experiment()
    
    # 결과 요약
    print(f"\n📋 실험 요약:")
    print(f"  • 테스트된 청킹 방식: {len(experiment.methods)}개")
    print(f"  • 생성된 청크 파일: {len([m for m in experiment.methods if 'error' not in experiment.results[m]])}개")
    print(f"  • 테스트 쿼리 수: 5개")
    print(f"  • 결과 파일: data/processed/chunking_experiment_results.json")


if __name__ == "__main__":
    main() 