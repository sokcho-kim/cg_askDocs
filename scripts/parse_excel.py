# 공정회의록 Excel 파일을 처리하는 스크립트
# 담당: 제로
"""
Excel 파일을 판다스로 읽어 행/시트 단위로 텍스트 청크를 추출합니다.
각 청크는 get_chunks() 포맷에 맞춰 dict로 저장합니다.
예시:
{
    "chunk_id": "...",
    "document_id": "...",
    "chunk_index": 0,
    "chunk_type": "table",
    "location": "sheet:Sheet1,row:2",
    "content": "...",
    "embedding": null,
    "metadata": {"length": 45}
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

# 실제 구현은 추후 추가
def excel_to_chunks(excel_path, output_path, document_id: str | None = None):
    doc = ExcelDocument(excel_path)
    processor = ExcelChunkProcessor(document_id=document_id)
    chunks = processor.process(doc)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[✓] 청크 분석 완료: {len(chunks)}개 청크 생성")
    print(f"[✓] 결과 저장: {output_path}")
    return chunks


if __name__ == "__main__":
    excel_to_chunks(
        excel_path="./data/raw/공정회의록.xlsx",
        output_path="./data/processed/excel_chunks.json",
        document_id="process_meeting_record"
    )