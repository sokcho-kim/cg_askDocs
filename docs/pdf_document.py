"""
PDF 문서 전처리 및 메타/청크 추출 구현 클래스
담당: 속초
"""
from .base_document import AbstractDocument

class PDFDocument(AbstractDocument):
    def __init__(self, filepath: str):
        """PDF 파일 경로를 받아 초기화"""
        ...

    def get_metadata(self) -> dict:
        """PDF 문서의 메타데이터 반환"""
        ...

    def get_chunks(self) -> list[dict]:
        """PDF 문서의 청크(페이지/블록 등) 반환"""
        ...
