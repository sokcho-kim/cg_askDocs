#!/usr/bin/env python3
"""
PDF 청크 메타데이터 수정 스크립트
PDF 청크에 data_type 필드를 추가하여 문서 타입을 명확히 구분
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def fix_pdf_chunks_metadata(file_path: str, output_path: str = ""):
    """PDF 청크 파일의 메타데이터에 data_type 필드를 추가합니다."""
    
    if not output_path:
        output_path = file_path
    
    print(f"🔧 PDF 청크 메타데이터 수정: {file_path}")
    
    # 파일 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"📄 {len(chunks)}개 청크 처리 중...")
    
    # 각 청크에 data_type 필드 추가
    fixed_count = 0
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        
        # data_type 필드가 없으면 추가
        if 'data_type' not in metadata:
            chunk_type = chunk.get('chunk_type', '')
            if chunk_type in ['text', 'section', 'block']:
                metadata['data_type'] = 'pdf_text'
            else:
                metadata['data_type'] = 'pdf_content'
            
            chunk['metadata'] = metadata
            fixed_count += 1
    
    # 수정된 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {fixed_count}개 청크의 메타데이터를 수정했습니다.")
    print(f"💾 수정된 파일 저장: {output_path}")
    
    return chunks


def main():
    """메인 함수"""
    print("🔧 PDF 청크 메타데이터 수정 시작")
    print("=" * 50)
    
    # 처리할 PDF 청크 파일들
    pdf_chunk_files = [
        "data/processed/DR_스마트야드개론(데모용)_chunks.json",
        "data/processed/pdf_chunks.json",
        "data/processed/pdf_chunks_block.json",
        "data/processed/pdf_chunks_section.json",
        "data/processed/pdf_chunks_page.json",
        "data/processed/pdf_chunks_adaptive.json"
    ]
    
    for file_path in pdf_chunk_files:
        if Path(file_path).exists():
            print(f"\n📁 파일 처리: {file_path}")
            fix_pdf_chunks_metadata(file_path)
        else:
            print(f"⚠️ 파일이 없습니다: {file_path}")
    
    print("\n" + "=" * 50)
    print("🎉 PDF 청크 메타데이터 수정 완료!")
    print("\n💡 다음 단계:")
    print("  1. ChromaDB 재인덱싱: python scripts/index_to_chroma.py")
    print("  2. 챗봇 테스트: python scripts/test_rag_chatbot.py")


if __name__ == "__main__":
    main() 