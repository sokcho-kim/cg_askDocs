# PDF 파일의 텍스트 및 이미지 설명을 처리하는 스크립트
# 담당: 속초
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

# 실제 구현은 추후 추가
