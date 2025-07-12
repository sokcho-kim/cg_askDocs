"""
PDF 문서 전처리 및 메타/청크 추출 구현 클래스
담당: 속초
"""
from .base_document import AbstractDocument
from pathlib import Path
import fitz  # PyMuPDF

class PDFDocument(AbstractDocument):
    def __init__(self, filepath: str):
        """PDF 파일 경로를 받아 초기화"""
        self.filepath = filepath
        self.filename = Path(filepath).name
        self.pdf = fitz.open(filepath)

    def get_metadata(self) -> dict:
        """PDF 문서의 메타데이터 반환"""
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "page_count": self.pdf.page_count,
        }

    def get_pages(self):
        """페이지별 텍스트/이미지 정보 반환 (generator)"""
        for idx in range(self.pdf.page_count):
            page = self.pdf.load_page(idx)
            text = page.get_text().strip()  # type: ignore
            images = page.get_images(full=True)
            yield {
                "page_num": idx + 1,
                "text": text,
                "images": images,
                "page_obj": page,
            }

    def get_chunks(self) -> list[dict]:
        """PDF 문서의 청크(페이지/블록 등) 반환 (기본: 페이지 단위)"""
        chunks = []
        for page_info in self.get_pages():
            # 텍스트 청크
            if page_info["text"]:
                chunks.append({
                    "type": "text",
                    "page": page_info["page_num"],
                    "content": page_info["text"]
                })
            # 이미지 청크 (xref만 제공, 실제 이미지는 필요시 추출)
            for img in page_info["images"]:
                chunks.append({
                    "type": "image",
                    "page": page_info["page_num"],
                    "xref": img[0]
                })
        return chunks
