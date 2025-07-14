# cg_askDocs

PDF, Excel 문서를 통합적으로 처리하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- **문서 처리**: PDF, Excel 파일 자동 파싱 및 청킹
- **벡터 검색**: ChromaDB 기반 하이브리드 검색
- **AI 답변**: Gemma API 기반 자연어 답변 생성
- **웹 인터페이스**: 파일 업로드 및 챗봇 대화

## 설치

```bash
pip install chromadb sentence-transformers pandas openpyxl pypdf2 fastapi uvicorn requests
```

## 실행

```bash
# 웹 애플리케이션 실행
uvicorn rag.webapp:app --reload

# 브라우저에서 접속
# http://localhost:8000
```

## 사용법

1. 웹페이지에서 PDF/Excel 파일 업로드
2. 파일이 자동으로 파싱되어 ChromaDB에 저장
3. 챗봇과 자연어로 질문/답변

## 프로젝트 구조

```
cg_askDocs/
├── data/
│   ├── raw/           # 원본 PDF, Excel 파일
│   ├── processed/     # 처리된 청크 데이터
│   └── db/           # ChromaDB 벡터 데이터베이스
├── rag/              # RAG 시스템
│   ├── webapp.py     # 웹 애플리케이션
│   ├── chatbot.py    # AI 챗봇
│   ├── retriever.py  # 벡터 검색기
│   └── api.py        # API 엔드포인트
├── scripts/          # 문서 처리 스크립트
│   ├── parse_pdf.py  # PDF 파싱
│   ├── parse_excel.py # Excel 파싱
│   └── index_to_chroma.py # ChromaDB 인덱싱
├── docs/             # 문서 처리 클래스
│   ├── base_document.py
│   ├── pdf_document.py
│   └── excel_document.py
├── utils/            # 유틸리티
│   ├── chunk_processor.py
│   └── file_utils.py
├── templates/        # 웹 UI 템플릿
│   └── index.html
├── static/          # 웹 UI 스타일
│   └── style.css
└── prompts/         # AI 프롬프트
    ├── rag_chat.prompt
    └── image_description.prompt
```

## ⚠️ ChromaDB 버전 및 DB 파일 관리 주의사항

- 본 프로젝트는 `chromadb==0.4.24` 버전에 맞춰 개발되었습니다. requirements.txt의 버전을 임의로 변경하지 마세요.
- **ChromaDB 버전을 업그레이드/다운그레이드하거나, 외부에서 생성된 DB 파일을 복사할 경우 반드시 기존 DB 파일(`data/db/chroma/` 내부 파일)을 삭제 후 재생성해야 합니다.**
- 버전 불일치 또는 DB 파일이 남아 있으면 `sqlite3.OperationalError: no such column: collections.topic` 등 스키마 오류가 발생할 수 있습니다.
- 오류 발생 시, 아래 명령어로 DB 파일을 삭제 후 서버를 재시작하세요:

```bash
rm -rf data/db/chroma/*
```

---
