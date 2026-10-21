import hashlib
from typing import Any, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from src.config import settings


class VectorStore:
    """
    Manages semantic vector caching using Qdrant.

    Why:
    - Cosine Distance: Ideal for normalized dense embeddings.
    - MD5 IDs: Generates stable, deterministic UUIDs for vector points based on prompt text.
    """

    def __init__(self, location: Optional[str] = None):
        if location:
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(url=settings.qdrant_url, timeout=5.0)

        self.collection_name = settings.qdrant_collection
        self.vector_dim = settings.embedding_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the collection on Qdrant if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                ),
            )

    def search_similar(
        self,
        query_vector: list[float],
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Queries Qdrant for the closest vector above the similarity threshold."""
        target_threshold = (
            threshold if threshold is not None else settings.similarity_threshold
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1,
            score_threshold=target_threshold,
        ).points

        if results:
            hit = results[0]
            return {
                "score": hit.score,
                "payload": hit.payload,
            }
        return None

    def store(
        self,
        prompt: str,
        vector: list[float],
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stores a prompt-response vector pair in Qdrant."""
        point_id = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        payload = {
            "prompt": prompt,
            "response": response,
            **(metadata or {}),
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return point_id
