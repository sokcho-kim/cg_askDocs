# cg_askDocs

## 프로젝트 개요

이 프로젝트는 PDF, Excel 등 다양한 문서(향후 PPT 등 확장 가능)를 통합적으로 전처리·검색·응답하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

---

## 디렉토리 구조

```
project_root/
├── data/
│   ├── raw/                        # 원본 PDF, Excel 등
│   ├── processed/
│   │   ├── pdf_pages/             # PDF 페이지별 JSON
│   │   ├── excel_metadata.json    # 문서 메타데이터 예시
│   │   └── linked_documents.json  # 문서 간 연결정보 예시
│   └── db/                        # 벡터 DB (chroma 등)
│
├── docs/
│   ├── base_document.py           # 🔹 AbstractDocument (문서 추상화)
│   ├── pdf_document.py            # 🔹 PDFDocument (담당: 속초)
│   ├── excel_document.py          # 🔹 ExcelDocument (담당: 제로)
│
├── scripts/
│   ├── parse_pdf.py               # PDF 전처리 (담당: 속초)
│   ├── parse_excel.py             # Excel 전처리 (담당: 제로)
│   ├── index_to_chroma.py         # 벡터DB 인덱싱 (공용)
│   └── test_chunk_processors.py   # 청크 프로세서 테스트
│
├── rag/
│   ├── retriever.py               # Retriever 정의
│   ├── chatbot.py                 # Chatbot chain 정의
│   └── api.py                     # FastAPI endpoint
│
├── prompts/
│   ├── image_description.prompt
│   └── rag_chat.prompt
│
├── utils/
│   ├── file_utils.py              # 파일 관련 유틸
│   ├── ocr_or_image_utils.py      # OCR/이미지 유틸
│   └── chunk_processor.py         # 🔹 청크 처리 공통 클래스
│
├── .gitignore
├── .env.example
├── README.md
└── main.py
```

---

## 데이터베이스/벡터DB 스키마

### 1️⃣ documents (관계형 DB)
| 필드         | 타입   | 설명                       |
|--------------|--------|----------------------------|
| document_id  | UUID   | 문서 식별자                |
| source_name  | TEXT   | 원본 파일명                |
| doc_type     | TEXT   | 파일 형식 (pdf, excel 등)  |
| created_at   | DATETIME | 수집 시각                |
| extra_meta   | JSON   | (선택) 추가 정보           |

### 2️⃣ chunks (관계형 DB + 벡터DB)
| 필드         | 타입   | 설명                       |
|--------------|--------|----------------------------|
| chunk_id     | UUID   | 청크 식별자                |
| document_id  | UUID   | 소속 문서                  |
| chunk_index  | INTEGER| 문서 내 순서               |
| chunk_type   | TEXT   | text/table/image_ocr 등     |
| location     | TEXT   | page:5, sheet:Sheet1 등     |
| content      | TEXT   | 요약/핵심 문장              |
| embedding    | VECTOR | (벡터DB에만) 임베딩         |
| metadata     | JSON   | (선택) 추가 정보            |

---

## 파이프라인 요약
1. **문서 수집**: data/raw/에 PDF, Excel 등 저장
2. **전처리**: 담당자별 스크립트 실행
   - 속초: `scripts/parse_pdf.py` (PDF → 청크, VLM 이미지 캡션 포함)
   - 제로: `scripts/parse_excel.py` (Excel → 청크, 시트/행 단위 분할)
3. **청크 요약/임베딩**: LLM 요약, 임베딩 생성
4. **저장**: 관계형DB(documents, chunks), 벡터DB(chunks)
5. **검색/응답**: 벡터DB에서 유사도 검색 → LLM context로 활용

### 🔹 청크 처리 아키텍처
- **공통 클래스**: `utils/chunk_processor.py`의 `ChunkProcessor` 추상 클래스
- **PDF 전용**: `PDFChunkProcessor` (기본), `SmartYardPDFProcessor` (VLM 포함)
- **Excel 전용**: `ExcelChunkProcessor` (기본), `MeetingRecordExcelProcessor` (공정회의록 전용)
- **확장성**: 새로운 문서 타입은 `ChunkProcessor`를 상속하여 구현

---

## 협업 가이드 (담당자 분업)
- **속초**: PDF 전처리 담당 (`docs/pdf_document.py`, `scripts/parse_pdf.py`)
- **제로**: Excel 전처리 담당 (`docs/excel_document.py`, `scripts/parse_excel.py`)
- **공통**: 벡터DB 인덱싱, 검색/응답 파이프라인(`scripts/index_to_chroma.py`, `rag/` 등)
- **코드/포맷 통일**: `AbstractDocument`의 `get_metadata()`, `get_chunks()` 포맷에 맞춰 구현
- **예시/주석**: 각 파일 상단에 예시 및 담당자 주석 추가

---

## 예시 데이터/포맷
- `data/processed/excel_metadata.json`, `linked_documents.json`에 예시 JSON 포함
- 각 청크 예시:
```json
{
  "chunk_id": "...",
  "document_id": "...",
  "chunk_index": 0,
  "chunk_type": "text",
  "location": "page:1",
  "content": "...",
  "embedding": null,
  "metadata": {"length": 123}
}
```

---

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 등 필요한 API 키 설정
```

### 3. 실행
```bash
python main.py
```

### 4. 테스트 및 개별 실행

#### 청크 프로세서 테스트
```bash
python scripts/test_chunk_processors.py
```

#### 개별 문서 처리
```bash
# PDF 처리
python scripts/parse_pdf.py

# Excel 처리  
python scripts/parse_excel.py
```

### 5. 개발 환경 설정 (선택사항)
```bash
# 코드 포맷팅
black .

# 린팅
flake8 .

# 테스트 실행
pytest
```

(실제 실행 코드는 추후 구현 예정)

---

## 문의/협업
- 담당자: 속초(PDF), 제로(Excel)
- 코드/구조/DB/포맷 관련 논의는 README 또는 각 파일 주석 참고