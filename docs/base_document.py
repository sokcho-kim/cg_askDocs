"""
문서 추상화의 기반이 되는 AbstractDocument 클래스.
모든 문서 타입(PDF, Excel, PPT 등)은 이 클래스를 상속받아 구현.
"""
# 속초: PDF 전처리 담당
# 제로: Excel 전처리 담당

from abc import ABC, abstractmethod

class AbstractDocument(ABC):
    @abstractmethod
    def get_metadata(self) -> dict:
        """문서 단위 메타데이터 반환"""
        pass

    @abstractmethod
    def get_chunks(self) -> list[dict]:
        """문서 내 청크(조각) 단위 정보 반환"""
        pass
