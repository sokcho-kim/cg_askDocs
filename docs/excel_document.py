"""
Excel 문서 전처리 및 메타/청크 추출 구현 클래스
담당: 제로
"""
from .base_document import AbstractDocument

class ExcelDocument(AbstractDocument):
    def __init__(self, filepath: str):
        """Excel 파일 경로를 받아 초기화"""
        ...

    def get_metadata(self) -> dict:
        """Excel 문서의 메타데이터 반환"""
        ...

    def get_chunks(self) -> list[dict]:
        """Excel 문서의 청크(행/시트 등) 반환"""
        ...
