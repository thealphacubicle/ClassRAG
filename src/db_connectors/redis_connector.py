import redis
from src.utils.db_model import DBModel
from redis.commands.search.query import Query
import ollama
import numpy as np
from typing import List, Dict


class RedisConnector(DBModel):
    """
    Connector implementation for Redis vector database (inherited from DBModel abstract class).
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, index_name: str = "embedding_index",
                 distance_metric: str = "COSINE", doc_prefix: str = "doc:", vector_dim: int = 768):
        """
        Initialize the Redis connector with the given host and port.

        :param host: The host address of the Redis server.
        :param port: The port number of the Redis server, default is 6380.
        :param db: The database number to connect to, default is 0.
        :param index_name: The name of the index to be used in Redis.
        :param distance_metric: The distance metric to be used for vector similarity search.
        :param doc_prefix: The prefix to be used for document keys in Redis.
        :param vector_dim: The dimension of the vectors to be stored in Redis.
        """
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)
        self.index_name = index_name
        self.distance_metric = distance_metric
        self.doc_prefix = doc_prefix
        self.vector_dim = vector_dim

        # Create the index
        self._create_hnsw_index()

    def index_embeddings(self, documents: list, embeddings: list, metadata: list = None, ids: list = None):
        """
        Index the embeddings with associated metadata.
        :param documents: List of documents to index.
        :param embeddings: List of embeddings to index.
        :param metadata: List of metadata dicts associated with the documents (optional)
        :param ids: List of custom IDs for the documents (optional)
        :return: None
        """
        for i, embedding in enumerate(embeddings):
            # Use metadata for this document or an empty dict if not provided
            meta = metadata[i] if metadata is not None else {}

            # If custom IDs are provided, use the document prefix to create unique key
            if ids is not None:
                key = f"{self.doc_prefix}:{ids[i]}"

            else:
                # Dynamically build key from metadata: sort keys for consistency.
                if meta:
                    key_parts = [f"{k}_{meta[k]}" for k in sorted(meta.keys())]
                    key_suffix = "_".join(key_parts)
                    key = f"{self.doc_prefix}:{key_suffix}"
                else:
                    key = f"{self.doc_prefix}:{i}"

            # Create the mapping: start with metadata and update with text and embedding.
            redis_mapping = meta.copy()
            redis_mapping.update({
                "text": documents[i],
                "embedding": np.array(embedding, dtype=np.float32).tobytes(),  # Store as a byte array
            })

            self.redis_client.hset(key, mapping=redis_mapping)

    def query_db(self, query_embedding: list, top_k: int = 1) -> tuple[List[str], Dict]:
        """
        Query the database with an embedding and return the top_k results.
        :param query_embedding: Embedding form of the query.
        :param top_k: Number of results to return.
        :return:
            context: List of documents retrieved from the database.
            metadata: List of all metadata retrieved from the database.
        """
        # Convert embedding to bytes for Redis search
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        try:
            # Build the vector similarity search query to return all fields
            q = (
                Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .dialect(2)
            )

            # Run search query.
            results = self.redis_client.ft(self.index_name).search(
                q, query_params={"vec": query_vector}
            )

            context, metadata = [doc.text for doc in results.docs], results.docs[0].__dict__

            # Return text and metadata of the top_k results
            return context, metadata

        except Exception as e:
            print(f"Error querying database: {e}")
            return [], {}

    def _create_hnsw_index(self):
        try:
            self.redis_client.execute_command(f"FT.DROPINDEX {self.index_name} DD")
        except redis.exceptions.ResponseError:
            pass

        self.redis_client.execute_command(
            f"""
            FT.CREATE {self.index_name} ON HASH PREFIX 1 {"doc:"}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {768} TYPE FLOAT32 DISTANCE_METRIC {self.distance_metric}
            """
        )
        print("Index created successfully.")


if __name__ == "__main__":
    # SAMPLE IMPLEMENTATION
    redis_db = RedisConnector()

    # Check if Redis connection exists
    assert redis_db.redis_client.ping(), "Redis connection failed"

    # Create the index
    redis_db._create_hnsw_index()

    # Add sample documents to the database
    documents = ["Cookies are yummy", "Trucks are fast", "Yummy is defined as a taste that is delicious.",
                 "Fast is defined as moving quickly."]
    embeddings = [ollama.embeddings(model="nomic-embed-text", prompt=doc)["embedding"] for doc in documents]
    metadata = [{"file": "file1", "page": 1, "chunk": 1}, {"file": "file2", "page": 2, "chunk": 2},
                {"file": "file3", "page": 3, "chunk": 3}, {"file": "file4", "page": 4, "chunk": 4}]

    redis_db.index_embeddings(documents, embeddings, metadata)
    print("Embeddings indexed successfully.")

    # Query the database
    query = "What is the definition of yummy?"
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    top_k = 2
    results = redis_db.query_db(query_embedding, top_k)
    print(results)
