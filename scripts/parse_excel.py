# 공정회의록 Excel 파일을 처리하는 스크립트
# 담당: 제로
"""
Excel 파일을 판다스로 읽어 행/시트 단위로 텍스트 청크를 추출합니다.
각 청크는 통일된 get_chunks() 포맷에 맞춰 dict로 저장합니다.
예시:
{
    "chunk_id": "excel_doc_table_0",
    "document_id": "excel_doc",
    "chunk_index": 0,
    "chunk_type": "table",
    "location": "sheet:Sheet1,row:0",
    "content": "주차: 2452 | 대분류: 사외 | 팀: 사외공정관리팀 | 이슈: 사외블록 2개 입고 지연",
    "embedding": null,
    "metadata": {
        "length": 45,
        "chunk_in_content": 0,
        "row_index": 0,
        "sheet_name": "Sheet1",
        "columns": ["주차", "대분류", "팀", "이슈"],
        "data_type": "excel_row"
    }
}
"""

import pandas as pd
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils.chunk_processor import ExcelChunkProcessor


def parse_excel_file(filepath: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Excel 파일을 파싱하여 통일된 청크 형식으로 변환합니다.
    
    Args:
        filepath: Excel 파일 경로
        output_path: 출력 JSON 파일 경로 (None이면 자동 생성)
    
    Returns:
        청크 리스트
    """
    # 파일명에서 document_id 생성
    filename = Path(filepath).stem
    document_id = f"excel_{filename}"
    
    # Excel 파일 읽기
    excel_file = pd.ExcelFile(filepath)
    
    # 청크 프로세서 초기화
    processor = ExcelChunkProcessor(document_id=document_id)
    
    all_chunks = []
    
    # 각 시트별로 처리
    for sheet_name in excel_file.sheet_names:
        print(f"[📊] 시트 처리 중: {sheet_name}")
        
        # 시트 데이터 읽기
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        # 빈 행 제거
        df = df.dropna(how='all')
        
        # 데이터프레임을 딕셔너리 리스트로 변환
        df_data = df.to_dict('records')
        
        # 청크 생성
        chunks = processor.process_excel_data(df_data, sheet_name)
        all_chunks.extend(chunks)
        
        print(f"[✓] {sheet_name}: {len(chunks)}개 청크 생성")
    
    # 출력 파일 경로 설정
    if output_path is None:
        output_path = f"data/processed/{filename}_chunks.json"
    
    # 청크 저장
    processor.chunks = all_chunks
    processor.save_chunks(output_path)
    
    print(f"[🎉] 총 {len(all_chunks)}개 청크 생성 완료")
    return all_chunks


def convert_existing_excel_metadata(input_path: str, output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    기존 excel_metadata.json을 통일된 형식으로 변환합니다.
    
    Args:
        input_path: 기존 excel_metadata.json 경로
        output_path: 변환된 파일 저장 경로
    
    Returns:
        변환된 청크 리스트
    """
    # 기존 데이터 읽기
    with open(input_path, 'r', encoding='utf-8') as f:
        old_chunks = json.load(f)
    
    # 통일된 형식으로 변환
    converted_chunks = []
    chunk_index = 0
    
    for old_chunk in old_chunks:
        # 기존 형식에서 필요한 정보 추출
        row_index = old_chunk.get('row_index', 0)
        column = old_chunk.get('column', '')
        content = old_chunk.get('content', '')
        
        # 통일된 형식으로 변환
        new_chunk = {
            "chunk_id": f"excel_converted_{chunk_index}",
            "document_id": "excel_converted",
            "chunk_index": chunk_index,
            "chunk_type": "table",
            "location": f"row:{row_index}",
            "content": content,
            "embedding": None,
            "metadata": {
                "length": len(content),
                "chunk_in_content": 0,
                "row_index": row_index,
                "column": column,
                "data_type": "excel_row_converted"
            }
        }
        
        converted_chunks.append(new_chunk)
        chunk_index += 1
    
    # 변환된 데이터 저장
    if output_path is None:
        output_path = input_path.replace('.json', '_converted.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"[🔄] {len(converted_chunks)}개 청크 변환 완료 -> {output_path}")
    return converted_chunks


if __name__ == "__main__":
    # 예시 실행
    excel_file = "data/raw/DR_공정회의자료_추출본(데모용).xlsx"
    
    if Path(excel_file).exists():
        print(f"[🚀] Excel 파일 파싱 시작: {excel_file}")
        chunks = parse_excel_file(excel_file)
        print(f"[✅] 파싱 완료: {len(chunks)}개 청크")
    else:
        print(f"[❌] 파일을 찾을 수 없습니다: {excel_file}")
        
        # 기존 excel_metadata.json 변환
        existing_file = "data/processed/excel_metadata.json"
        if Path(existing_file).exists():
            print(f"[🔄] 기존 파일 변환 시작: {existing_file}")
            convert_existing_excel_metadata(existing_file)
        else:
            print(f"[❌] 변환할 파일도 없습니다: {existing_file}")
