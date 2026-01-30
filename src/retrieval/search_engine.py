"""Qdrant vector database integration for semantic search."""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, Range, SearchRequest
)
from qdrant_client.http.exceptions import UnexpectedResponse

from .embedder import EmbeddingService

logger = logging.getLogger(__name__)


class VectorDB:
    """Qdrant vector database wrapper with connection management and error handling."""

    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection_name: str = "products", vector_size: int = 384,
                 api_key: Optional[str] = None, timeout: int = 30, batch_size: int = 100):
        """Initialize Qdrant client with connection management.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of collection to use
            vector_size: Dimension of embeddings
            api_key: Optional API key for authentication
            timeout: Connection timeout in seconds
            
        Raises:
            ConnectionError: If cannot connect to Qdrant
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        
        try:
            self.client = QdrantClient(
                host=host, 
                port=port,
                api_key=api_key,
                timeout=timeout
            )
            # Test connection
            self.client.get_collections()
            logger.info(f"✓ Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant at {host}:{port}: {e}")
    
    def create_collection(self, recreate: bool = False) -> bool:
        """Create collection if it doesn't exist.
        
        Args:
            recreate: If True, delete and recreate collection
            
        Returns:
            True if successful
        """
        try:
            if recreate:
                try:
                    self.client.delete_collection(self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except:
                    pass
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists and not recreate:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✓ Created collection '{self.collection_name}' with {self.vector_size}D vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def index_products(self, products: List[Dict[str, Any]], batch_size: Optional[int] = None) -> int:
        """Index products to Qdrant with batching.
        
        Args:
            products: List of products with 'embedding' field
            batch_size: Number of products per batch
            
        Returns:
            Number of products successfully indexed
        """
        if not products:
            logger.warning("No products to index")
            return 0
        
        indexed_count = 0
        
        use_batch_size = batch_size if batch_size is not None else getattr(self, 'batch_size', 100)
        for i in range(0, len(products), use_batch_size):
            batch = products[i:i + use_batch_size]
            points = []
            
            for product in batch:
                if 'embedding' not in product:
                    logger.warning(f"Product {product.get('id', 'unknown')} missing embedding")
                    continue
                
                try:
                    point = PointStruct(
                        id=product.get('id', indexed_count),
                        vector=product['embedding'],
                        payload={
                            'product_id': product.get('id'),
                            'title': product.get('title', ''),
                            'description': product.get('description', ''),
                            'price': float(product.get('price', 0)),
                            'category': product.get('category', ''),
                            'brand': product.get('brand', ''),
                            'rating': float(product.get('rating', 0)),
                            'msrp': float(product.get('msrp', 0)),
                            'discount_pct': float(product.get('discount_pct', 0)),
                            'stock': int(product.get('stock', 0)),
                            'availability': product.get('availability', ''),
                            'payment_methods': product.get('payment_methods', ''),
                            'installment_available': product.get('installment_available', False),
                            'max_installments': int(product.get('max_installments', 0)),
                            'shipping_days': int(product.get('shipping_days', 0)),
                            'budget_band': product.get('budget_band', ''),
                            'tags': product.get('tags', ''),
                        }
                    )
                    points.append(point)
                except Exception as e:
                    logger.warning(f"Failed to create point for product {product.get('id')}: {e}")
                    continue
            
            if points:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    indexed_count += len(points)
                    logger.info(f"Indexed batch {i//batch_size + 1}: {len(points)} products")
                except Exception as e:
                    logger.error(f"Failed to index batch {i//batch_size + 1}: {e}")
        
        logger.info(f"✓ Total indexed: {indexed_count}/{len(products)} products")
        return indexed_count
    
    def search(self, query_vector: List[float], max_budget: Optional[float] = None,
               category: Optional[str] = None, top_k: int = 5,
               score_threshold: Optional[float] = None, budget: Optional[float] = None, query_text: Optional[str] = None, **kwargs) -> List[Any]:
        """Perform constraint-aware semantic search with fallback to keyword/fuzzy search if needed.
        Args:
            query_vector: Query embedding vector
            max_budget: Maximum price filter (optional)
            category: Category filter (optional)
            top_k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            query_text: Original query text (for fallback)
        Returns:
            List of search results with scores and payloads
        """
        if not query_vector or len(query_vector) != self.vector_size:
            logger.warning(f"Invalid query vector. Expected {self.vector_size}D, got {len(query_vector) if query_vector else 'None'}")
            return []

        # Build filter conditions
        filter_conditions = []
        effective_budget = max_budget if max_budget is not None else budget
        if effective_budget is not None:
            filter_conditions.append(
                FieldCondition(
                    key="price",
                    range=Range(lte=effective_budget)
                )
            )
        if category and str(category).strip().lower() not in ["", "all categories", "none", "null"]:
            filter_conditions.append(
                FieldCondition(
                    key="category",
                    match={"value": str(category).strip().title()}
                )
            )
        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )
            results = getattr(response, 'points', None)
            if results is None:
                results = getattr(response, 'result', None)
            if results is None:
                logger.error("Qdrant query_points returned no results or points field.")
                results = []
            logger.info(f"Semantic search for '{query_text}' returned {len(results)} results")

            # Always check for strict substring matches in semantic results
            if query_text:
                query_lc = query_text.strip().lower()
                strict_matches = []
                category_lc = str(category).strip().lower() if category else None
                for pt in results:
                    payload = pt.payload if hasattr(pt, 'payload') else pt
                    title = str(payload.get('title', '')).lower()
                    tags = ' '.join(payload.get('tags', []) if isinstance(payload.get('tags', []), list) else str(payload.get('tags', '')).split(';')).lower()
                    cat = str(payload.get('category', '')).lower()
                    # Enforce category filter if specified
                    if category_lc and cat != category_lc:
                        continue
                    if query_lc in title or query_lc in tags or query_lc in cat:
                        setattr(pt, '_fincommerce_match_type', 'strict')
                        # Set score to 1.0 for strict keyword match
                        if hasattr(pt, 'score'):
                            pt.score = 1.0
                        else:
                            try:
                                pt['score'] = 1.0
                            except Exception:
                                pass
                        strict_matches.append(pt)
                if strict_matches:
                    logger.info(f"Strict substring matches for '{query_text}' in semantic results: {len(strict_matches)}")
                    return strict_matches[:top_k]
                # If no strict matches, fallback to strict keyword search over all points
                logger.info(f"No strict substring matches for '{query_text}' in semantic results. Trying strict keyword fallback.")
                scroll_result = self.client.scroll(collection_name=self.collection_name, with_payload=True, limit=10000)
                all_points = scroll_result.points if hasattr(scroll_result, 'points') else (scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result)
                strict_matches = []
                for pt in all_points:
                    payload = pt.payload if hasattr(pt, 'payload') else pt
                    title = str(payload.get('title', '')).lower()
                    tags = ' '.join(payload.get('tags', []) if isinstance(payload.get('tags', []), list) else str(payload.get('tags', '')).split(';')).lower()
                    cat = str(payload.get('category', '')).lower()
                    # Enforce category filter if specified
                    if category_lc and cat != category_lc:
                        continue
                    if query_lc in title or query_lc in tags or query_lc in cat:
                        setattr(pt, '_fincommerce_match_type', 'strict')
                        # Set score to 1.0 for strict keyword match
                        if hasattr(pt, 'score'):
                            pt.score = 1.0
                        else:
                            try:
                                pt['score'] = 1.0
                            except Exception:
                                pass
                        strict_matches.append(pt)
                logger.info(f"Strict keyword fallback for '{query_text}' returned {len(strict_matches)} results.")
                return strict_matches[:top_k]
            return results
        except UnexpectedResponse as e:
            logger.error(f"Qdrant search failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            logger.info(f"[get_stats] Checking collection '{self.collection_name}' on {self.host}:{self.port}")
            info = self.client.get_collection(self.collection_name)
            points_count = getattr(info, 'points_count', None)
            # Fallback: if points_count is None or 0, count manually
            if not points_count:
                try:
                    scroll = self.client.scroll(collection_name=self.collection_name, with_payload=False, limit=100000)
                    if hasattr(scroll, 'points'):
                        points_count = len(scroll.points)
                    elif isinstance(scroll, tuple):
                        points_count = len(scroll[0])
                    else:
                        points_count = 0
                    logger.info(f"[get_stats] Fallback counted {points_count} points via scroll.")
                except Exception as e2:
                    logger.error(f"[get_stats] Fallback scroll failed: {e2}")
                    points_count = 0
            return {
                'collection_name': self.collection_name,
                'points_count': points_count,
                'vector_size': self.vector_size,
                'status': getattr(info, 'status', 'unknown')
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if Qdrant connection is healthy.
        
        Returns:
            True if healthy
        """
        try:
            self.client.get_collections()
            return True
        except:
            return False


class FinSearchEngine:
    """High-level search engine combining embedding and vector search.

    This is a convenience wrapper that maintains backward compatibility
    with the student's original API.
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize search engine with embedding service and vector DB.
        Args:
            host: Qdrant host
            port: Qdrant port
        """
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDB(host=host, port=port)
        self.collection_name = self.vector_db.collection_name
        self.embedder = self.embedding_service  # For test compatibility

    def search(self, query_text: str, top_k: int = 5, budget: Optional[float] = None, max_budget: Optional[float] = None, category: Optional[str] = None, **kwargs) -> List[Any]:
        """Perform end-to-end semantic search with constraints and fallback.
        Args:
            query_text: User query string
            top_k: Number of results
            budget: Maximum price constraint (alias for max_budget)
            max_budget: Maximum price constraint
            category: Optional category filter
        Returns:
            List of search results
        """
        # Robust query handling: normalize, lowercase, strip
        if not query_text or not query_text.strip():
            logger.warning("Empty or invalid query_text received for search.")
            return []
        query_text_norm = query_text.strip().lower()

        # Improved embedding: use title, description, tags, category, etc. if available
        # For user query, we can only use the query, but for product embedding (see indexer), use more fields
        query_vector = self.embedding_service.embed(query_text_norm)

        effective_budget = max_budget if max_budget is not None else budget

        # Search with constraints and fallback
        results = self.vector_db.search(
            query_vector=query_vector,
            max_budget=effective_budget,
            category=category,
            top_k=top_k,
            query_text=query_text_norm
        )
        logger.info(f"Search for '{query_text}' (budget={effective_budget}, category={category}) returned {len(results)} results.")
        return results


__all__ = ['VectorDB', 'FinSearchEngine']
