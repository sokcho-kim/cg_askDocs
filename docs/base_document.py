"""
문서 추상화의 기반이 되는 AbstractDocument 클래스.
모든 문서 타입(PDF, Excel, PPT 등)은 이 클래스를 상속받아 구현.
- get_metadata(): documents 테이블에 저장할 dict 반환
- get_chunks(): chunks 테이블/벡터DB에 저장할 dict 리스트 반환

필드 예시 및 포맷은 README 참고
"""
# 속초: PDF 전처리 담당
# 제로: Excel 전처리 담당

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class AbstractDocument(ABC):
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        문서 단위 메타데이터 반환
        예시 반환값:
        {
            "document_id": "...",
            "source_name": "파일명.pdf",
            "doc_type": "pdf",
            "created_at": "2024-07-11T12:00:00",
            "extra_meta": {"project": "스마트야드"}
        }
        """
        pass

    @abstractmethod
    def get_chunks(self) -> List[Dict[str, Any]]:
        """
        문서 내 청크(조각) 단위 정보 반환
        예시 반환값:
        [
            {
                "chunk_id": "...",
                "document_id": "...",
                "chunk_index": 0,
                "chunk_type": "text",
                "location": "page:1",
                "content": "...",
                "embedding": None,
                "metadata": {"length": 123}
            },
            ...
        ]
        """
        pass
