"""
Excel 문서 전처리 및 메타/청크 추출 구현 클래스
담당: 제로
"""
from base_document import AbstractDocument
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import uuid

# row 별로 합쳐서 하나의 text로 chunkin -> metadata
class ExcelDocument(AbstractDocument):
    def __init__(self, filepath: str):
        """Excel 파일 경로를 받아 초기화"""
        self.filepath = filepath
        self.df = pd.read_excel(filepath)
        
    def get_metadata(self) -> Dict[str, Any]:
        """documents 테이블에 저장할 메타데이터를 반환"""
        return {
            "document_id": str(uuid.uuid4()),
            "source_name": self.filepath.split("/")[-1],
            "doc_type": "excel",
            "created_at": datetime.utcnow().isoformat(),
            "extra_meta": {"project": self.project}
        }

    def get_chunks(self, df) -> List[Dict[str, Any]]:
        """Excel 문서의 청크(행/시트 등) 반환"""
        chunks = []
        for idx, row in df.iterrows():
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    continue
                chunk_text = f"{col}: {val}"
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": "excel_row_converted",
                    "chunk_index": idx,
                    "chunk_type": "text",
                    "location": col,
                    "content": chunk_text,
                    "embedding": None,
                    "metadata": {
                        "source_file": "{df}",
                        "length": len(chunk_text),
                        "row_index": idx,
                        "location": col
                    }
                })
                    
        return chunks
            
            
            
