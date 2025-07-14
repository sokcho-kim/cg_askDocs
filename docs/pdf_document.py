"""
PDF 문서 전처리 및 메타/청크 추출 구현 클래스
담당: 속초
"""
from .base_document import AbstractDocument
from pathlib import Path
import fitz  # PyMuPDF
import re
import base64
import os
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import os
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

def image_to_base64(img_bytes):
    import base64
    return base64.b64encode(img_bytes).decode("utf-8")

def get_image_caption_with_vlm(image_bytes, context_text=""):
    if client is None:
        return "[이미지 설명 오류: OpenAI 키 없음 또는 초기화 실패]"
    try:
        base64_image = image_to_base64(image_bytes)
        image_url = f"data:image/png;base64,{base64_image}"
        prompt = f"""
You are analyzing an image extracted from a smart yard presentation slide.\nFocus on industrial/technical keywords like AI, robotics, digital twin, automation, smart factory, etc.\nAvoid vague or artistic expressions.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[이미지 설명 오류: {str(e)}]"

def clean_symbols(text):
    symbols_to_remove = "•▪⚫"
    for sym in symbols_to_remove:
        text = text.replace(sym, "")
    return text

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
            text = (page.get_text() or '').strip()  # type: ignore
            text = clean_symbols(text)
            images = page.get_images(full=True)
            yield {
                "page_num": idx + 1,
                "text": text,
                "images": images,
                "page_obj": page,
            }

    def get_blocks(self):
        """블록 단위로 텍스트/이미지 정보 반환 (generator) - 개선된 버전"""
        for idx in range(self.pdf.page_count):
            page = self.pdf.load_page(idx)
            
            # 블록 단위로 텍스트 추출
            blocks = page.get_text("dict")["blocks"]
            
            for block_idx, block in enumerate(blocks):
                if "lines" in block:  # 텍스트 블록
                    text_content = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_content += span["text"]
                    
                    if text_content.strip():
                        yield {
                            "page_num": idx + 1,
                            "block_num": block_idx + 1,
                            "text": text_content.strip(),
                            "bbox": block["bbox"],  # 블록 위치 정보
                            "type": "text"
                        }
                
                elif "image" in block:  # 이미지 블록
                    yield {
                        "page_num": idx + 1,
                        "block_num": block_idx + 1,
                        "image": block["image"],
                        "bbox": block["bbox"],
                        "type": "image"
                    }

    def get_sections(self):
        """섹션 단위로 텍스트 정보 반환 (generator) - 제목 기반 분할"""
        current_section = []
        current_title = "Introduction"
        
        for page_info in self.get_pages():
            page_text = page_info["text"]
            page_num = page_info["page_num"]
            
            # 제목 패턴 찾기 (예: "Part 1", "Chapter", "1.", "1.1" 등)
            lines = page_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 제목 패턴 확인
                if self._is_title(line):
                    # 이전 섹션 저장
                    if current_section:
                        yield {
                            "page_num": page_num,
                            "title": current_title,
                            "content": "\n".join(current_section),
                            "type": "section"
                        }
                    
                    # 새 섹션 시작
                    current_title = line
                    current_section = [line]
                else:
                    current_section.append(line)
            
            # 페이지 끝에서 섹션 저장
            if current_section:
                yield {
                    "page_num": page_num,
                    "title": current_title,
                    "content": "\n".join(current_section),
                    "type": "section"
                }
                current_section = []

    def _is_title(self, line: str) -> bool:
        """라인이 제목인지 판단"""
        # 제목 패턴들
        title_patterns = [
            r'^Part\s+\d+',  # Part 1, Part 2, ...
            r'^Chapter\s+\d+',  # Chapter 1, Chapter 2, ...
            r'^\d+\.\s+',  # 1. 제목, 2. 제목, ...
            r'^\d+\.\d+\s+',  # 1.1 제목, 1.2 제목, ...
            r'^[A-Z][A-Z\s]+$',  # 대문자로만 된 라인
            r'^[가-힣\s]+$',  # 한글로만 된 라인 (짧은 경우)
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, line.strip()):
                return True
        
        # 길이가 짧고 특수문자가 적은 경우
        if len(line.strip()) < 50 and len(re.findall(r'[^\w\s]', line)) < 3:
            return True
        
        return False

    def get_chunks(self) -> list[dict]:
        """PDF 문서의 청크(페이지/블록 등) 반환 (기본: 페이지 단위)"""
        chunks = []
        for page_info in self.get_pages():
            text = page_info["text"] or ""
            # 이미지가 있으면 VLM 설명을 텍스트에 이어붙임
            if page_info["images"]:
                for img in page_info["images"]:
                    xref = img[0]
                    image_data = self.pdf.extract_image(xref)
                    img_bytes = image_data["image"]
                    try:
                        caption = get_image_caption_with_vlm(img_bytes)
                    except Exception as e:
                        caption = f"[이미지 설명 오류: {str(e)}]"
                    text += f"\n[이미지 설명] {caption}"
            if text.strip():
                chunks.append({
                    "type": "text",
                    "page": page_info["page_num"],
                    "content": text.strip()
                })
        return chunks
