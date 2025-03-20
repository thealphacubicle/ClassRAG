import redis
from src.utils.db_model import DBModel
from redis.commands.search.query import Query
import numpy as np
from numpy import ndarray
from typing import List, Dict, Any, Optional, Tuple


class RedisConnector(DBModel):
    """
    Connector implementation for Redis vector database (inherited from DBModel abstract class).
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, index_name: str = "embedding_index",
                 distance_metric: str = "COSINE", doc_prefix: str = "doc:", vector_dim: int = 1024):
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

    def index_embeddings(
            self,
            documents: List[List[str]],
            embeddings: List[List[List[ndarray]]],
            metadata: Optional[List[List[Dict[str, Any]]]] = None,
            ids: Optional[List[str]] = None
    ) -> None:
        """
        Index the embeddings with associated metadata for each chunk of every document.

        :param documents: List of documents to index (each document is a list of chunks).
        :param embeddings: List of embeddings for each document's chunks.
        :param metadata: List of metadata for each document's chunks (optional).
        :param ids: List of custom IDs for the documents (optional).
        :return: None
        """
        for i, doc in enumerate(documents):
            for j, chunk in enumerate(doc):
                try:
                    # Get metadata for the current chunk if provided; otherwise, use an empty dict.
                    if metadata is not None and i < len(metadata) and j < len(metadata[i]):
                        meta = metadata[i][j]
                    else:
                        meta = {}

                    # Build a unique key:
                    # If custom document IDs are provided, combine the document ID and chunk index.
                    if ids is not None and i < len(ids):
                        key = f"{self.doc_prefix}:{ids[i]}_{j}"
                    else:
                        # If metadata is available, create a key based on sorted metadata items.
                        if meta:
                            key_parts = [f"{k}_{meta[k]}" for k in sorted(meta.keys())]
                            key_suffix = "_".join(key_parts)
                            key = f"{self.doc_prefix}:{key_suffix}"
                        else:
                            # Fallback: use document and chunk indices.
                            key = f"{self.doc_prefix}:{i}_{j}"

                    # Convert the embedding vector for the current chunk to bytes.
                    embedding_bytes = np.array(embeddings[i][j], dtype=np.float32).tobytes()

                    # Create a mapping: start with metadata and add the chunk text and embedding.
                    redis_mapping = meta.copy()
                    redis_mapping.update({
                        "text": chunk,
                        "embedding": embedding_bytes,
                    })

                    self.redis_client.hset(key, mapping=redis_mapping)

                except Exception as e:
                    print(f"Error indexing embeddings for document {i} chunk {j}: {e}")

    def query_db(self, query_embedding: list, top_k: int = 1) -> Tuple[List[str], List[Dict]]:
        """
        Query the database with an embedding and return the top_k results.

        :param query_embedding: Embedding form of the query.
        :param top_k: Number of results to return.
        :return:
            A tuple containing:
                - A list of text chunks (strings) for the top_k results.
                - A list of metadata dictionaries corresponding to each chunk.
        """
        # Convert the query embedding to bytes for Redis search.
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        try:
            # Build the vector similarity search query.
            q = (
                Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .dialect(2)
            )

            # Execute the search query.
            results = self.redis_client.ft(self.index_name).search(
                q, query_params={"vec": query_vector}
            )

            # Extract the text and metadata for each result.
            contexts = [doc.text for doc in results.docs]
            metadata_list = [doc.__dict__ for doc in results.docs]

            return contexts, metadata_list

        except Exception as e:
            print(f"Error querying database: {e}")
            return [], []

    def _create_hnsw_index(self):
        try:
            # Check if the index already exists
            self.redis_client.execute_command(f"FT.INFO {self.index_name}")
            print("Index already exists. Skipping creation.")
        except redis.exceptions.ResponseError:
            # Index does not exist, create it
            self.redis_client.execute_command(
                f"""
                FT.CREATE {self.index_name} ON HASH PREFIX 1 {"doc:"}
                SCHEMA text TEXT
                embedding VECTOR HNSW 6 DIM {self.vector_dim} TYPE FLOAT32 DISTANCE_METRIC {self.distance_metric}
                """
            )
            print("Index created successfully.")