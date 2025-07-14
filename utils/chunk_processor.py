"""
문서 청크 분할 및 처리를 위한 공통 클래스
PDF, Excel, 기타 문서 타입에 대해 재사용 가능한 청크 처리 로직
"""

import json
import uuid
import re
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
    
    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드를 추출합니다 (RAG 검색 최적화용)"""
        # 한글, 영문, 숫자로 구성된 단어 추출
        keywords = re.findall(r'[가-힣a-zA-Z0-9]+', text)
        
        # 길이가 2 이상인 단어만 필터링
        keywords = [kw for kw in keywords if len(kw) >= 2]
        
        # 빈도수 기반으로 상위 키워드 선택 (최대 10개)
        keyword_freq = {}
        for kw in keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        # 빈도수 순으로 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, freq in sorted_keywords[:10]]
    
    def calculate_chunk_quality_score(self, content: str, chunk_type: str) -> float:
        """청크의 품질 점수를 계산합니다 (RAG 검색 우선순위용)"""
        score = 0.0
        
        # 기본 점수: 길이 기반
        if len(content) > 50:
            score += 0.3
        elif len(content) > 20:
            score += 0.2
        else:
            score += 0.1
        
        # 키워드 다양성 점수
        keywords = self.extract_keywords(content)
        if len(keywords) >= 5:
            score += 0.3
        elif len(keywords) >= 3:
            score += 0.2
        else:
            score += 0.1
        
        # 청크 타입별 점수
        if chunk_type == "table":
            score += 0.2  # 구조화된 데이터는 높은 점수
        elif chunk_type == "text":
            score += 0.1
        elif chunk_type == "image":
            score += 0.15
        
        # 특수 패턴 점수
        if re.search(r'[0-9]+', content):  # 숫자 포함
            score += 0.1
        if re.search(r'[가-힣]+', content):  # 한글 포함
            score += 0.1
        
        return min(score, 1.0)  # 최대 1.0
    
    def create_standard_chunk(self, 
                            content: str, 
                            chunk_type: str = "text", 
                            location: str = "", 
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """통일된 형식의 청크를 생성합니다 (RAG 최적화용)"""
        # 키워드 추출
        keywords = self.extract_keywords(content)
        
        # 품질 점수 계산
        quality_score = self.calculate_chunk_quality_score(content, chunk_type)
        
        chunk = {
            "chunk_id": f"{self.document_id}_{chunk_type}_{self.chunk_index}",
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunk_type": chunk_type,
            "location": location,
            "content": content,
            "embedding": None,
            "metadata": {
                "length": len(content),
                "chunk_in_content": 0,
                "keywords": keywords,
                "quality_score": quality_score,
                "search_priority": "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "low",
                **(metadata or {})
            }
        }
        self.chunk_index += 1
        return chunk
    
    def create_text_chunk(self, content: str, location: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """텍스트 청크를 생성합니다."""
        return self.create_standard_chunk(content, "text", location, metadata)
    
    def create_image_chunk(self, caption: str, location: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """이미지 청크를 생성합니다."""
        return self.create_standard_chunk(caption, "image", location, metadata)
    
    def create_table_chunk(self, content: str, location: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """표 청크를 생성합니다."""
        return self.create_standard_chunk(content, "table", location, metadata)
    
    def create_excel_row_chunk(self, row_data: Dict[str, str], row_index: int, sheet_name: str = "Sheet1") -> Dict[str, Any]:
        """Excel 행 데이터를 통일된 형식으로 변환합니다."""
        # 행 데이터를 구조화된 텍스트로 변환
        content_parts = []
        for column, value in row_data.items():
            if value and str(value).strip():
                content_parts.append(f"{column}: {value}")
        
        content = " | ".join(content_parts)
        location = f"sheet:{sheet_name},row:{row_index}"
        
        metadata = {
            "row_index": row_index,
            "sheet_name": sheet_name,
            "columns": list(row_data.keys()),
            "data_type": "excel_row"
        }
        
        return self.create_standard_chunk(content, "table", location, metadata)
    
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
    
    def filter_high_quality_chunks(self, min_quality_score: float = 0.5) -> List[Dict[str, Any]]:
        """품질 점수가 높은 청크만 필터링합니다."""
        return [chunk for chunk in self.chunks 
                if chunk.get('metadata', {}).get('quality_score', 0) >= min_quality_score]
    
    def sort_chunks_by_quality(self) -> List[Dict[str, Any]]:
        """청크를 품질 점수 순으로 정렬합니다."""
        return sorted(self.chunks, 
                     key=lambda x: x.get('metadata', {}).get('quality_score', 0), 
                     reverse=True)
    
    def save_chunks(self, output_path: str):
        """청크들을 JSON 파일로 저장합니다."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        print(f"[✓] 청크 저장 완료: {len(self.chunks)}개 청크 -> {output_path}")
        
        # 품질 통계 출력
        quality_scores = [chunk.get('metadata', {}).get('quality_score', 0) for chunk in self.chunks]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            high_quality_count = len([s for s in quality_scores if s > 0.7])
            print(f"[📊] 품질 통계: 평균 {avg_quality:.2f}, 고품질 청크 {high_quality_count}개")
    
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
    
    def process_pdf_by_blocks(self, document) -> List[Dict[str, Any]]:
        """블록 단위로 PDF 청킹 (더 세밀한 분할)"""
        self.chunks = []
        self.chunk_index = 0
        
        if hasattr(document, 'get_blocks'):
            for block_info in document.get_blocks():
                location = f"page:{block_info.get('page_num', '?')},block:{block_info.get('block_num', '?')}"
                
                if block_info.get("type") == "text":
                    self.process_text_content(
                        block_info["text"], 
                        location, 
                        {
                            "page": block_info.get("page_num"),
                            "block": block_info.get("block_num"),
                            "bbox": block_info.get("bbox"),
                            "chunking_method": "block"
                        }
                    )
                elif block_info.get("type") == "image":
                    metadata = {
                        "page": block_info.get("page_num"),
                        "block": block_info.get("block_num"),
                        "bbox": block_info.get("bbox"),
                        "chunking_method": "block"
                    }
                    chunk = self.create_image_chunk(f"[Image block {block_info.get('block_num')}]", location, metadata)
                    self.add_chunk(chunk)
        
        return self.chunks
    
    def process_pdf_by_sections(self, document) -> List[Dict[str, Any]]:
        """섹션 단위로 PDF 청킹 (제목 기반 분할)"""
        self.chunks = []
        self.chunk_index = 0
        
        if hasattr(document, 'get_sections'):
            for section_info in document.get_sections():
                location = f"page:{section_info.get('page_num', '?')},section:{section_info.get('title', '?')}"
                
                # 섹션 제목을 포함한 전체 내용
                full_content = f"{section_info.get('title', '')}\n{section_info.get('content', '')}"
                
                self.process_text_content(
                    full_content,
                    location,
                    {
                        "page": section_info.get("page_num"),
                        "title": section_info.get("title"),
                        "chunking_method": "section"
                    }
                )
        
        return self.chunks
    
    def process_pdf_by_pages(self, document) -> List[Dict[str, Any]]:
        """페이지 단위로 PDF 청킹 (기본 방식)"""
        return self.process(document)
    
    def process_pdf_adaptive(self, document, max_chunk_size: int = 800) -> List[Dict[str, Any]]:
        """적응형 PDF 청킹 (내용에 따라 자동 선택)"""
        self.chunks = []
        self.chunk_index = 0
        
        if hasattr(document, 'get_pages'):
            for page_info in document.get_pages():
                page_text = page_info.get("text", "")
                page_num = page_info.get("page_num", 0)
                
                # 페이지 크기에 따라 청킹 방식 결정
                if len(page_text) <= max_chunk_size:
                    # 작은 페이지: 페이지 단위
                    location = f"page:{page_num}"
                    self.process_text_content(page_text, location, {"page": page_num, "chunking_method": "page"})
                else:
                    # 큰 페이지: 블록 단위로 재시도
                    if hasattr(document, 'get_blocks'):
                        for block_info in document.get_blocks():
                            if block_info.get("page_num") == page_num:
                                location = f"page:{page_num},block:{block_info.get('block_num', '?')}"
                                
                                if block_info.get("type") == "text":
                                    self.process_text_content(
                                        block_info["text"],
                                        location,
                                        {
                                            "page": page_num,
                                            "block": block_info.get("block_num"),
                                            "chunking_method": "adaptive_block"
                                        }
                                    )
                
                # 이미지 처리
                for img in page_info.get("images", []):
                    metadata = {
                        "page": page_num,
                        "image_index": img[0] if isinstance(img, (list, tuple)) else None,
                        "chunking_method": "adaptive"
                    }
                    chunk = self.create_image_chunk(f"[Image xref {img[0]}]", f"page:{page_num}", metadata)
                    self.add_chunk(chunk)
        
        return self.chunks


class ExcelChunkProcessor(ChunkProcessor):
    """Excel 문서 전용 청크 프로세서 (이제 문서 객체만 받음)"""
    
    def process_excel_data(self, df_data: List[Dict[str, Any]], sheet_name: str = "Sheet1") -> List[Dict[str, Any]]:
        """Excel 데이터프레임을 통일된 청크 형식으로 변환합니다."""
        self.chunks = []
        self.chunk_index = 0
        
        for row_index, row_data in enumerate(df_data):
            # 각 행을 하나의 청크로 변환
            chunk = self.create_excel_row_chunk(row_data, row_index, sheet_name)
            self.add_chunk(chunk)
        
        return self.chunks 