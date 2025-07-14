# PDF 파일의 텍스트 및 이미지 설명을 처리하는 스크립트
"""
PDF 파일을 다양한 방식으로 청킹하여 텍스트, 표, 이미지 청크를 추출합니다.
각 청크는 get_chunks() 포맷에 맞춰 dict로 저장합니다.

청킹 방식:
1. page: 페이지 단위 (기본)
2. block: 블록 단위 (더 세밀한 분할)
3. section: 섹션 단위 (제목 기반 분할)
4. adaptive: 적응형 (내용에 따라 자동 선택)

예시:
{
    "chunk_id": "...",
    "document_id": "...",
    "chunk_index": 0,
    "chunk_type": "text",
    "location": "page:1",
    "content": "...",
    "embedding": null,
    "metadata": {"length": 123}
}
"""

import sys
from pathlib import Path
from typing import Literal
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from docs.pdf_document import PDFDocument
from utils.chunk_processor import PDFChunkProcessor
import json


def parse_pdf_to_chunks(
    pdf_path: str, 
    output_path: str, 
    document_id: str | None = None,
    chunking_method: Literal["page", "block", "section", "adaptive"] = "adaptive"
):
    """
    PDF 파일을 청크 단위로 분할하여 get_chunks() 포맷에 맞는 결과를 생성합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        output_path: 출력 JSON 파일 경로
        document_id: 문서 ID (None이면 자동 생성)
        chunking_method: 청킹 방식
            - "page": 페이지 단위
            - "block": 블록 단위 (더 세밀한 분할)
            - "section": 섹션 단위 (제목 기반 분할)
            - "adaptive": 적응형 (내용에 따라 자동 선택)
    """
    # 1. 문서 ID 자동 생성
    if document_id is None:
        document_id = os.path.splitext(os.path.basename(pdf_path))[0]
    # 1. 문서 객체 생성
    doc = PDFDocument(pdf_path)
    
    # 2. 청크 프로세서로 분할
    processor = PDFChunkProcessor(document_id=document_id)
    
    # 청킹 방식에 따라 처리
    if chunking_method == "page":
        chunks = processor.process_pdf_by_pages(doc)
        print(f"[📄] 페이지 단위 청킹: {len(chunks)}개 청크")
    elif chunking_method == "block":
        chunks = processor.process_pdf_by_blocks(doc)
        print(f"[🧱] 블록 단위 청킹: {len(chunks)}개 청크")
    elif chunking_method == "section":
        chunks = processor.process_pdf_by_sections(doc)
        print(f"[📑] 섹션 단위 청킹: {len(chunks)}개 청크")
    elif chunking_method == "adaptive":
        chunks = processor.process_pdf_adaptive(doc)
        print(f"[🎯] 적응형 청킹: {len(chunks)}개 청크")
    else:
        raise ValueError(f"지원하지 않는 청킹 방식: {chunking_method}")
    
    # 3. 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] 청크 분석 완료: {len(chunks)}개 청크 생성")
    print(f"[✓] 결과 저장: {output_path}")
    
    # 청킹 통계 출력
    print_chunking_stats(chunks, chunking_method)
    
    return chunks


def print_chunking_stats(chunks: list, method: str):
    """청킹 통계를 출력합니다."""
    if not chunks:
        return
    
    # 청크 타입별 통계
    type_counts = {}
    method_counts = {}
    total_length = 0
    
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "unknown")
        chunking_method = chunk.get("metadata", {}).get("chunking_method", "unknown")
        content_length = len(chunk.get("content", ""))
        
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        method_counts[chunking_method] = method_counts.get(chunking_method, 0) + 1
        total_length += content_length
    
    print(f"\n📊 청킹 통계 ({method} 방식):")
    print(f"  - 총 청크 수: {len(chunks)}")
    print(f"  - 평균 청크 길이: {total_length // len(chunks)} 문자")
    print(f"  - 청크 타입별:")
    for chunk_type, count in type_counts.items():
        print(f"    • {chunk_type}: {count}개")
    print(f"  - 청킹 방식별:")
    for chunking_method, count in method_counts.items():
        print(f"    • {chunking_method}: {count}개")


def compare_chunking_methods(pdf_path: str, output_dir: str = "data/processed"):
    """다양한 청킹 방식을 비교합니다."""
    print("🔍 청킹 방식 비교 분석")
    print("=" * 50)
    
    methods = ["page", "block", "section", "adaptive"]
    results = {}
    
    for method in methods:
        output_path = f"{output_dir}/pdf_chunks_{method}.json"
        print(f"\n📋 {method.upper()} 방식 테스트 중...")
        
        try:
            chunks = parse_pdf_to_chunks(
                pdf_path=pdf_path,
                output_path=output_path,
                document_id=f"pdf_{method}",
                chunking_method=method
            )
            results[method] = {
                "chunk_count": len(chunks),
                "avg_length": sum(len(chunk.get("content", "")) for chunk in chunks) // len(chunks) if chunks else 0,
                "file_path": output_path
            }
        except Exception as e:
            print(f"  ❌ {method} 방식 실패: {e}")
            results[method] = {"error": str(e)}
    
    # 비교 결과 출력
    print(f"\n📊 청킹 방식 비교 결과:")
    print("-" * 50)
    for method, result in results.items():
        if "error" in result:
            print(f"  {method.upper()}: ❌ 실패 - {result['error']}")
        else:
            print(f"  {method.upper()}: {result['chunk_count']}개 청크, 평균 {result['avg_length']}자")
    
    return results


# ✅ 실행 예시
if __name__ == "__main__":
    pdf_file = "./data/raw/DR_스마트야드개론(데모용).pdf"
    
    # 1. 실제용 실행 (BLOCK 청킹 - 최적 성능)
    print("🚀 PDF 파싱 시작 (실제용 - BLOCK 청킹)")
    parse_pdf_to_chunks(
        pdf_path=pdf_file,
        output_path="./data/processed/DR_스마트야드개론(데모용)_chunks.json",
        document_id="smart_yard_intro_production",
        chunking_method="block"  # 실험 결과 최적 성능 방식
    )
    
    # 2. 실험용 실행 (모든 방식 비교) - 주석 처리됨
    # print("\n" + "=" * 60)
    # print("🔬 실험용: 청킹 방식 비교 분석")
    # compare_chunking_methods(pdf_file)