"""
청크 프로세서 클래스들의 사용 예시 및 테스트 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.chunk_processor import ChunkProcessor, PDFChunkProcessor, ExcelChunkProcessor
from docs.pdf_document import PDFDocument
# from scripts.parse_excel import MeetingRecordExcelProcessor  # 삭제


def test_basic_chunk_processor():
    """기본 ChunkProcessor 사용 예시"""
    print("=== 기본 ChunkProcessor 테스트 ===")
    
    # 기본 프로세서 생성 (추상 클래스이므로 구체적인 구현체 사용)
    processor = PDFChunkProcessor(document_id="test_doc", max_chunk_size=500, overlap=50)
    
    # 텍스트 청크 생성
    text = "이것은 테스트 텍스트입니다. " * 20  # 긴 텍스트
    processor.process_text_content(text, "test_location", {"source": "test"})
    
    # 이미지 청크 생성
    processor.create_image_chunk("테스트 이미지 설명", "test_location", {"image_type": "test"})
    
    # 결과 확인
    chunks = processor.get_chunks()
    print(f"생성된 청크 수: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):  # 처음 2개만 출력
        print(f"청크 {i+1}: {chunk['chunk_type']} - {chunk['content'][:50]}...")
    
    print()


def test_pdf_processor():
    """PDF 프로세서 사용 예시"""
    print("=== PDF 프로세서 테스트 ===")
    
    pdf_path = "./data/raw/DR_스마트야드개론(데모용).pdf"
    
    if Path(pdf_path).exists():
        # 기본 PDF 프로세서
        basic_processor = PDFChunkProcessor(document_id="basic_pdf")
        try:
            chunks = basic_processor.process(PDFDocument(pdf_path))
            print(f"기본 PDF 프로세서: {len(chunks)}개 청크 생성")
        except ImportError as e:
            print(f"PyMuPDF 미설치: {e}")
        # SmartYardPDFProcessor 관련 코드는 삭제
    else:
        print(f"PDF 파일이 없습니다: {pdf_path}")
    
    print()


def test_excel_processor():
    """Excel 프로세서 사용 예시"""
    print("=== Excel 프로세서 테스트 ===")
    
    excel_path = "./data/raw/DR_공정회의자료_추출본(데모용).xlsx"
    
    if Path(excel_path).exists():
        # 기본 Excel 프로세서
        basic_processor = ExcelChunkProcessor(document_id="basic_excel")
        try:
            # 아래 줄은 실제 ExcelDocument 구현 후에만 동작함 (여기선 예시)
            # chunks = basic_processor.process(ExcelDocument(excel_path))
            print(f"(예시) Excel 프로세서: 동작하려면 ExcelDocument 구현 필요")
        except ImportError as e:
            print(f"pandas 미설치: {e}")
        # MeetingRecordExcelProcessor 관련 코드는 삭제
    else:
        print(f"Excel 파일이 없습니다: {excel_path}")
    
    print()


def test_custom_processor():
    """커스텀 프로세서 생성 예시"""
    print("=== 커스텀 프로세서 테스트 ===")
    
    class CustomTextProcessor(ChunkProcessor):
        """커스텀 텍스트 프로세서 예시"""
        
        def process_document(self, file_path: str):
            """간단한 텍스트 파일을 처리합니다."""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 텍스트를 문단 단위로 분할
                paragraphs = content.split('\n\n')
                
                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():
                        location = f"paragraph:{i+1}"
                        metadata = {"paragraph_index": i+1}
                        self.process_text_content(paragraph.strip(), location, metadata)
                
                return self.chunks
            except FileNotFoundError:
                print(f"파일을 찾을 수 없습니다: {file_path}")
                return []
    
    # 커스텀 프로세서 사용
    custom_processor = CustomTextProcessor(document_id="custom_text")
    
    # 테스트용 텍스트 파일 생성
    test_content = """첫 번째 문단입니다.
이것은 첫 번째 문단의 내용입니다.

두 번째 문단입니다.
이것은 두 번째 문단의 내용입니다.

세 번째 문단입니다.
이것은 세 번째 문단의 내용입니다."""
    
    test_file = Path("./data/raw/test_document.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(test_content, encoding='utf-8')
    
    # 처리
    chunks = custom_processor.process_document(str(test_file))
    print(f"커스텀 프로세서: {len(chunks)}개 청크 생성")
    
    # 결과 출력
    for chunk in chunks:
        print(f"- {chunk['chunk_type']}: {chunk['content'][:30]}...")
    
    # 테스트 파일 삭제
    test_file.unlink()
    print()


def main():
    """모든 테스트 실행"""
    print("🚀 청크 프로세서 테스트 시작\n")
    
    test_basic_chunk_processor()
    test_pdf_processor()
    test_excel_processor()
    test_custom_processor()
    
    print("✅ 모든 테스트 완료!")


if __name__ == "__main__":
    main() 