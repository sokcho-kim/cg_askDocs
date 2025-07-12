"""
문서 청크 분할 및 처리를 위한 공통 클래스
PDF, Excel, 기타 문서 타입에 대해 재사용 가능한 청크 처리 로직
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path


class ChunkProcessor(ABC):
    """문서 청크 처리를 위한 추상 기본 클래스"""
    
    def __init__(self, document_id: Optional[str] = None, max_chunk_size: int = 1000, overlap: int = 100):
        """
        Args:
            document_id: 문서 고유 ID (None이면 자동 생성)
            max_chunk_size: 최대 청크 크기 (문자 수)
            overlap: 청크 간 오버랩 크기 (문자 수)
        """
        self.document_id = document_id or str(uuid.uuid4())
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.chunk_index = 0
        self.chunks: List[Dict[str, Any]] = []
    
    def split_text_to_chunks(self, text: str) -> List[str]:
        """텍스트를 청크 단위로 분할합니다."""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chunk_size
            
            # 문장 경계에서 자르기
            if end < len(text):
                # 마침표, 느낌표, 물음표, 줄바꿈 뒤에서 자르기
                for punct in ['.', '!', '?', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start:
                        end = last_punct + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def create_text_chunk(self, content: str, location: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """텍스트 청크를 생성합니다."""
        chunk = {
            "chunk_id": f"{self.document_id}_text_{self.chunk_index}",
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_type": "text",
            "location": location,
            "content": content,
            "embedding": None,
            "metadata": {
                "length": len(content),
                **(metadata or {})
            }
        }
        self.chunk_index += 1
        return chunk
    
    def create_image_chunk(self, caption: str, location: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """이미지 청크를 생성합니다."""
        chunk = {
            "chunk_id": f"{self.document_id}_image_{self.chunk_index}",
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_type": "image",
            "location": location,
            "content": caption,
            "embedding": None,
            "metadata": {
                "length": len(caption),
                **(metadata or {})
            }
        }
        self.chunk_index += 1
        return chunk
    
    def create_table_chunk(self, content: str, location: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """표 청크를 생성합니다."""
        chunk = {
            "chunk_id": f"{self.document_id}_table_{self.chunk_index}",
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_type": "table",
            "location": location,
            "content": content,
            "embedding": None,
            "metadata": {
                "length": len(content),
                **(metadata or {})
            }
        }
        self.chunk_index += 1
        return chunk
    
    def add_chunk(self, chunk: Dict[str, Any]):
        """청크를 리스트에 추가합니다."""
        self.chunks.append(chunk)
    
    def process_text_content(self, text: str, location: str, metadata: Optional[Dict[str, Any]] = None):
        """텍스트 내용을 청크로 분할하여 처리합니다."""
        if not text.strip():
            return
        
        text_chunks = self.split_text_to_chunks(text)
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = {
                "chunk_in_content": i,
                **(metadata or {})
            }
            chunk = self.create_text_chunk(chunk_text, location, chunk_metadata)
            self.add_chunk(chunk)
    
    def save_chunks(self, output_path: str):
        """청크들을 JSON 파일로 저장합니다."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        print(f"[✓] 청크 저장 완료: {len(self.chunks)}개 청크 -> {output_path}")
    
    def get_chunks(self) -> List[Dict[str, Any]]:
        """처리된 청크들을 반환합니다."""
        return self.chunks
    
    def process(self, document) -> List[Dict[str, Any]]:
        """문서 객체(AbstractDocument)를 받아 청크 분할"""
        self.chunks = []
        self.chunk_index = 0
        # 기본적으로 get_pages()를 사용 (PDF, Excel 등)
        if hasattr(document, 'get_pages'):
            for page_info in document.get_pages():
                location = f"page:{page_info.get('page_num', '?')}"
                # 텍스트 청크
                if page_info.get("text"):
                    self.process_text_content(page_info["text"], location, {"page": page_info.get("page_num")})
                # 이미지 청크 (xref만 제공, 실제 이미지는 필요시 추출)
                for img in page_info.get("images", []):
                    metadata = {
                        "page": page_info.get("page_num"),
                        "image_index": img[0] if isinstance(img, (list, tuple)) else None
                    }
                    chunk = self.create_image_chunk(f"[Image xref {img[0]}]", location, metadata)
                    self.add_chunk(chunk)
        elif hasattr(document, 'get_chunks'):
            # 이미 청크화된 문서 객체 지원
            for chunk in document.get_chunks():
                self.add_chunk(chunk)
        else:
            raise ValueError("지원하지 않는 문서 객체입니다.")
        return self.chunks


class PDFChunkProcessor(ChunkProcessor):
    """PDF 문서 전용 청크 프로세서 (이제 문서 객체만 받음)"""
    pass


class ExcelChunkProcessor(ChunkProcessor):
    """Excel 문서 전용 청크 프로세서 (이제 문서 객체만 받음)"""
    pass 