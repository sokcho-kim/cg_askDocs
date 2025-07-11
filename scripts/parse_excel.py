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

# 실제 구현은 추후 추가
