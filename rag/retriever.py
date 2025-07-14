"""
RAG 시스템의 검색 및 검색 결과 처리 모듈
품질 점수, 키워드 매칭, 하이브리드 검색 등을 지원
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    use_onnx=False  # ✅ 이거 꼭 추가!
)


class EnhancedRetriever:
    """향상된 검색 기능을 제공하는 Retriever 클래스"""
    
    def __init__(self, db_path: str = "data/db/chroma"):
        """
        Args:
            db_path: ChromaDB 저장 경로
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 컬렉션 이름
        self.collection_name = "documents"
        
        # 컬렉션 가져오기 또는 생성
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG system"}
            )
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """청크들을 벡터 DB에 추가합니다."""
        if not chunks:
            return
        
        # 청크 데이터 준비
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            
            if chunk_id and content:
                # ChromaDB 호환 형식으로 metadata 변환
                chroma_metadata = self._convert_metadata_for_chroma(metadata)
                
                ids.append(chunk_id)
                documents.append(content)
                metadatas.append(chroma_metadata)
        
        # 벡터 DB에 추가 (ChromaDB가 자동으로 임베딩 생성)
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"[✓] {len(ids)}개 청크를 벡터 DB에 추가했습니다.")
    
    def _convert_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB 호환 형식으로 metadata 변환"""
        import json
        
        chroma_metadata = {}
        
        for key, value in metadata.items():
            if isinstance(value, list):
                # 리스트는 JSON 문자열로 변환 (더 안전함)
                chroma_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # 딕셔너리도 JSON 문자열로 변환
                chroma_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # ChromaDB가 지원하는 타입은 그대로 유지
                chroma_metadata[key] = value
            else:
                # 기타 타입은 문자열로 변환
                chroma_metadata[key] = str(value)
        
        return chroma_metadata
    
    def keyword_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """키워드 기반 검색 (메타데이터의 keywords 필드 활용)"""
        # 쿼리에서 키워드 추출
        import re
        query_keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query)
        query_keywords = [kw for kw in query_keywords if len(kw) >= 2]
        
        if not query_keywords:
            return []
        
        # 모든 문서 가져오기
        all_docs = self.collection.get()
        
        # 키워드 매칭 점수 계산
        scored_results = []
        if all_docs['metadatas']:
            for i, metadata in enumerate(all_docs['metadatas']):
                if not metadata:
                    continue
                    
                # keywords 필드를 문자열에서 리스트로 파싱
                keywords_str = metadata.get('keywords', '')
                if isinstance(keywords_str, str):
                    try:
                        # JSON 문자열인 경우 파싱
                        if keywords_str.startswith('[') and keywords_str.endswith(']'):
                            import json
                            chunk_keywords = json.loads(keywords_str)
                        else:
                            # 쉼표로 구분된 문자열인 경우
                            chunk_keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                    except:
                        # 파싱 실패시 쉼표로 분리
                        chunk_keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                else:
                    chunk_keywords = []
                
                if not chunk_keywords:
                    continue
                
                # 키워드 매칭 점수 계산
                matches = set(query_keywords) & set(chunk_keywords)
                if matches:
                    score = len(matches) / len(query_keywords)
                    quality_score = metadata.get('quality_score', 0)
                    if not isinstance(quality_score, (int, float)):
                        quality_score = 0
                    
                    # 최종 점수 = 키워드 매칭 점수 * 품질 점수
                    final_score = score * quality_score
                    
                    scored_results.append({
                        'chunk_id': all_docs['ids'][i],
                        'content': all_docs['documents'][i],
                        'metadata': metadata,
                        'score': final_score,
                        'keyword_matches': list(matches)
                    })
        
        # 점수 순으로 정렬
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_results[:n_results]
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """의미적 검색 (벡터 유사도)"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # 결과를 표준 형식으로 변환
            formatted_results = []
            if results['ids'] and results['ids'][0]:  # 결과가 있는 경우
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'chunk_id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if 'distances' in results and results['distances'] and results['distances'][0] else None,
                        'score': 1 - (results['distances'][0][i] if 'distances' in results and results['distances'] and results['distances'][0] else 0)
                    })
            
            return formatted_results
        except Exception as e:
            print(f"[⚠️] 의미적 검색 실패: {e}")
            return []
    
    def hybrid_search(self, query: str, n_results: int = 5, 
                     semantic_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """하이브리드 검색 (의미적 + 키워드)"""
        # 각각의 검색 결과 가져오기
        semantic_results = self.semantic_search(query, n_results * 2)
        keyword_results = self.keyword_search(query, n_results * 2)
        
        # 결과 통합
        combined_results = {}
        
        # 의미적 검색 결과 추가
        for result in semantic_results:
            chunk_id = result['chunk_id']
            combined_results[chunk_id] = {
                'chunk_id': chunk_id,
                'content': result['content'],
                'metadata': result['metadata'],
                'semantic_score': result['score'],
                'keyword_score': 0,
                'final_score': result['score'] * semantic_weight
            }
        
        # 키워드 검색 결과 추가/업데이트
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined_results:
                combined_results[chunk_id]['keyword_score'] = result['score']
                combined_results[chunk_id]['final_score'] += result['score'] * keyword_weight
            else:
                combined_results[chunk_id] = {
                    'chunk_id': chunk_id,
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'semantic_score': 0,
                    'keyword_score': result['score'],
                    'final_score': result['score'] * keyword_weight
                }
        
        # 최종 점수로 정렬
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results[:n_results]
    
    def quality_filtered_search(self, query: str, n_results: int = 5, 
                              min_quality_score: float = 0.5) -> List[Dict[str, Any]]:
        """품질 점수 필터링이 적용된 검색"""
        results = self.hybrid_search(query, n_results * 3)
        
        # 품질 점수로 필터링
        filtered_results = []
        for result in results:
            quality_score = result['metadata'].get('quality_score', 0)
            if quality_score >= min_quality_score:
                filtered_results.append(result)
                if len(filtered_results) >= n_results:
                    break
        
        return filtered_results
    
    def search_by_priority(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """검색 우선순위를 고려한 검색"""
        # 모든 결과 가져오기
        all_results = self.hybrid_search(query, n_results * 5)
        
        # 우선순위별로 그룹화
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for result in all_results:
            priority = result['metadata'].get('search_priority', 'low')
            if priority == 'high':
                high_priority.append(result)
            elif priority == 'medium':
                medium_priority.append(result)
            else:
                low_priority.append(result)
        
        # 우선순위 순으로 결과 구성
        final_results = []
        high_count = min(len(high_priority), n_results // 2)
        medium_count = min(len(medium_priority), (n_results - high_count) // 2)
        low_count = n_results - high_count - medium_count
        
        final_results.extend(high_priority[:high_count])
        final_results.extend(medium_priority[:medium_count])
        final_results.extend(low_priority[:low_count])
        
        return final_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """청크 ID로 특정 청크 조회"""
        try:
            result = self.collection.get(ids=[chunk_id])
            if result['ids']:
                return {
                    'chunk_id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
        except:
            pass
        return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보 반환"""
        try:
            count = self.collection.count()
            
            # 품질 점수 통계
            all_metadata = self.collection.get()['metadatas']
            quality_scores = [meta.get('quality_score', 0) for meta in all_metadata if meta]
            
            stats = {
                'total_chunks': count,
                'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                'high_quality_chunks': len([s for s in quality_scores if s > 0.7]),
                'medium_quality_chunks': len([s for s in quality_scores if 0.4 <= s <= 0.7]),
                'low_quality_chunks': len([s for s in quality_scores if s < 0.4])
            }
            
            return stats
        except Exception as e:
            return {'error': str(e)}
    
    def clear_collection(self):
        """컬렉션의 모든 데이터 삭제"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG system"}
            )
            print(f"[✓] 컬렉션 '{self.collection_name}'을 초기화했습니다.")
        except Exception as e:
            print(f"[❌] 컬렉션 초기화 실패: {e}")


# 사용 예시
if __name__ == "__main__":
    retriever = EnhancedRetriever()
    
    # 검색 테스트
    query = "스마트 야드 자동화"
    print(f"[🔍] 검색 쿼리: {query}")
    
    # 다양한 검색 방법 테스트
    print("\n1. 하이브리드 검색:")
    results = retriever.hybrid_search(query, 3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['chunk_id']} (점수: {result['final_score']:.3f})")
    
    print("\n2. 품질 필터링 검색:")
    results = retriever.quality_filtered_search(query, 3, min_quality_score=0.6)
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['chunk_id']} (품질: {result['metadata'].get('quality_score', 0):.2f})")
    
    print("\n3. 우선순위 검색:")
    results = retriever.search_by_priority(query, 3)
    for i, result in enumerate(results, 1):
        priority = result['metadata'].get('search_priority', 'low')
        print(f"  {i}. {result['chunk_id']} (우선순위: {priority})")
    
    # 통계 정보
    stats = retriever.get_collection_stats()
    print(f"\n📊 컬렉션 통계: {stats}")
