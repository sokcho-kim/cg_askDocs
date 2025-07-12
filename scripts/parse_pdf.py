# PDF 파일의 텍스트 및 이미지 설명을 처리하는 스크립트
"""
PDF 파일을 페이지/블록 단위로 분할하여 텍스트, 표, 이미지(및 OCR) 청크를 추출합니다.
각 청크는 get_chunks() 포맷에 맞춰 dict로 저장합니다.
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

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from docs.pdf_document import PDFDocument
from utils.chunk_processor import PDFChunkProcessor
import json


def parse_pdf_to_chunks(pdf_path: str, output_path: str, document_id: str | None = None):
    """
    PDF 파일을 청크 단위로 분할하여 get_chunks() 포맷에 맞는 결과를 생성합니다.
    """
    # 1. 문서 객체 생성
    doc = PDFDocument(pdf_path)
    # 2. 청크 프로세서로 분할
    processor = PDFChunkProcessor(document_id=document_id)
    chunks = processor.process(doc)
    # 3. 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[✓] 청크 분석 완료: {len(chunks)}개 청크 생성")
    print(f"[✓] 결과 저장: {output_path}")
    return chunks


# ✅ 실행 예시
if __name__ == "__main__":
    parse_pdf_to_chunks(
        pdf_path="./data/raw/DR_스마트야드개론(데모용).pdf",
        output_path="./data/processed/pdf_chunks.json",
        document_id="smart_yard_intro"
    )