# cg_askDocs

## 프로젝트 개요

이 프로젝트는 PDF, Excel 등 다양한 문서를 통합적으로 전처리·검색·응답하는 RAG(Retrieval-Augmented Generation) 시스템입니다. 
향상된 검색 기능(하이브리드, 품질 필터링, 우선순위)과 유연한 답변 생성으로 정확하고 관련성 높은 정보를 제공합니다.

---

## 디렉토리 구조

```
cg_askDocs/
├── data/
│   ├── raw/                        # 원본 PDF, Excel 파일
│   ├── processed/
│   │   ├── DR_스마트야드개론(데모용)_chunks.json    # 프로덕션 PDF 청크
│   │   ├── DR_공정회의자료_추출본(데모용)_chunks.json # 프로덕션 Excel 청크
│   │   ├── production_config.json                   # 프로덕션 설정
│   │   ├── excel_metadata.json                      # Excel 메타데이터
│   │   └── linked_documents.json                    # 문서 간 연결정보
│   └── db/                        # 벡터 DB (ChromaDB)
│
├── docs/
│   ├── base_document.py           # AbstractDocument (문서 추상화)
│   ├── pdf_document.py            # PDFDocument (다양한 청킹 방식 지원)
│   └── excel_document.py          # ExcelDocument (테이블 데이터 처리)
│
├── scripts/
│   ├── parse_pdf.py               # PDF 전처리 (블록/페이지/섹션/적응형 청킹)
│   ├── parse_excel.py             # Excel 전처리 (테이블 데이터 청킹)
│   ├── index_to_chroma.py         # ChromaDB 인덱싱
│   ├── run_production_workflow.py # 프로덕션 워크플로우
│   └── test_rag_chatbot.py        # RAG 챗봇 테스트
│
├── rag/
│   ├── retriever.py               # 향상된 검색기 (하이브리드, 키워드, 품질 필터링)
│   ├── chatbot.py                 # 향상된 RAG 챗봇 (유연한 답변 생성)
│   ├── api.py                     # FastAPI endpoint
│   └── webapp.py                  # 웹 애플리케이션
│
├── static/
│   └── style.css                  # 웹 UI 스타일
│
├── templates/
│   └── index.html                 # 웹 UI 템플릿
│
├── prompts/
│   ├── image_description.prompt
│   └── rag_chat.prompt
│
├── utils/
│   ├── file_utils.py              # 파일 관련 유틸
│   ├── ocr_or_image_utils.py      # OCR/이미지 유틸
│   └── chunk_processor.py         # 통일된 청크 처리 클래스
│
├── .gitignore
├── README.md
└── main.py
```

---

## 주요 기능

### 🔍 향상된 검색 기능
- **하이브리드 검색**: 의미적 검색 + 키워드 검색 조합
- **키워드 검색**: 정확한 키워드 매칭으로 관련성 높은 결과
- **품질 필터링**: 품질 점수 기반 결과 필터링
- **우선순위 검색**: 문서 타입별 우선순위 적용
- **향상된 검색**: Excel 데이터 우선 + 하이브리드 검색

### 🤖 유연한 답변 생성
- **질문 유형 분석**: 공정, 스마트야드, 기술 등 주제별 최적화
- **다중 문서 통합**: 여러 문서의 정보를 통합한 답변
- **출처 명시**: 답변에 사용된 문서 출처 표시
- **구조화된 답변**: 이슈, 대응방안, 현황 등 체계적 정리

### 📄 다양한 청킹 방식
- **페이지 청킹**: 페이지 단위 분할
- **블록 청킹**: 더 세밀한 블록 단위 분할 (최적 성능)
- **섹션 청킹**: 제목 기반 섹션 분할
- **적응형 청킹**: 내용에 따라 자동 선택

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
| metadata     | JSON   | 품질점수, 키워드, 데이터타입 등 |

---

## 파이프라인 요약
1. **문서 수집**: `data/raw/`에 PDF, Excel 파일 저장
2. **전처리**: 다양한 청킹 방식으로 문서 분할
   - PDF: `scripts/parse_pdf.py` (블록/페이지/섹션/적응형 청킹)
   - Excel: `scripts/parse_excel.py` (테이블 데이터 청킹)
3. **인덱싱**: `scripts/index_to_chroma.py`로 ChromaDB에 저장
4. **검색/응답**: 향상된 검색기 + 유연한 챗봇으로 정확한 답변 생성

### 🔹 청크 처리 아키텍처
- **통일된 형식**: `utils/chunk_processor.py`의 `ChunkProcessor` 기반
- **PDF 전용**: `PDFChunkProcessor` (다양한 청킹 방식 지원)
- **Excel 전용**: `ExcelChunkProcessor` (테이블 데이터 최적화)
- **메타데이터 자동 생성**: 품질점수, 키워드, 데이터타입 등

---

## 사용법

### 1. 프로덕션 워크플로우 실행
```bash
# 전체 파이프라인 실행 (PDF + Excel 처리 → 인덱싱)
python scripts/run_production_workflow.py
```

### 2. 개별 문서 처리
```bash
# PDF 처리 (블록 청킹 - 최적 성능)
python scripts/parse_pdf.py

# Excel 처리
python scripts/parse_excel.py

# ChromaDB 인덱싱
python scripts/index_to_chroma.py
```

### 3. RAG 챗봇 테스트
```bash
# 챗봇 테스트
python scripts/test_rag_chatbot.py

# 웹 애플리케이션 실행
python rag/webapp.py
```

### 4. Python에서 직접 사용
```python
from rag.chatbot import RAGChatbot

# 챗봇 초기화
chatbot = RAGChatbot()

# 질문하기
result = chatbot.chat("사외블록 입고 지연에 대해 알려줘")
print(result['response'])
```

---

## 예시 데이터/포맷

### 청크 형식
```json
{
  "chunk_id": "smart_yard_intro_production_text_0",
  "document_id": "smart_yard_intro_production",
  "chunk_index": 0,
  "chunk_type": "text",
  "location": "page:1,block:1",
  "content": "스마트 제조혁신 역량 강화 지원...",
  "embedding": null,
  "metadata": {
    "length": 17,
    "keywords": ["스마트", "제조혁신", "역량", "강화", "지원"],
    "quality_score": 0.6,
    "data_type": "pdf_text",
    "chunking_method": "block",
    "search_priority": "medium"
  }
}
```

### 챗봇 응답 예시
```
공정 관련 정보를 찾았습니다:

📊 공정 이슈 현황:
공정 지연 관련 이슈:
• 주차: 2452 | 대분류: 사외 | 팀: 사외공정관리팀 | 이슈: 사외블록 2개 입고 지연 | 리스크: 공정 지연 | 원인: 협력사 생산 지연 | 영향: 탑재 공정 지연 | 돌관작업으로 생산시수 증가 | 대응방안: 협력사 생산/입고 관리 요청 | 탑재 돌관작업...

📍 출처: excel, smart

더 구체적인 질문이 있으시면 말씀해 주세요.
```

---

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install chromadb sentence-transformers pandas openpyxl pypdf2
```

### 2. 데이터 준비
```bash
# data/raw/ 폴더에 PDF, Excel 파일 배치
mkdir -p data/raw
# PDF, Excel 파일을 data/raw/에 복사
```

### 3. 프로덕션 실행
```bash
# 전체 파이프라인 실행
python scripts/run_production_workflow.py

# 웹 애플리케이션 실행
python rag/webapp.py
```

### 4. 개발 환경 설정 (선택사항)
```bash
# 코드 포맷팅
pip install black
black .

# 린팅
pip install flake8
flake8 .
```

---

## 주요 개선사항

### ✅ 최근 업데이트
- **향상된 검색 기능**: 하이브리드, 키워드, 품질 필터링 검색 추가
- **유연한 답변 생성**: 질문 유형별 최적화된 답변 생성
- **메타데이터 자동 생성**: 품질점수, 키워드, 데이터타입 자동 설정
- **프로젝트 정리**: 실험용 파일 제거로 깔끔한 구조
- **웹 애플리케이션**: 사용자 친화적인 웹 UI 추가

### 🔧 기술적 개선
- **통일된 청크 형식**: PDF와 Excel 데이터 통합 처리
- **다양한 청킹 방식**: 페이지, 블록, 섹션, 적응형 청킹 지원
- **ChromaDB 최적화**: 효율적인 벡터 검색 및 저장
- **에러 처리**: 안정적인 예외 처리 및 복구

---

## 문의/협업
- 프로젝트 관련 문의: GitHub Issues 활용
- 코드/구조/DB/포맷 관련 논의는 README 또는 각 파일 주석 참고