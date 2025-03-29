from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.utils.db_model import DBModel
from typing import List, Dict, Any, Optional, Tuple
from numpy import ndarray

class QdrantConnector(DBModel):
    """
    Qdrant connector for embedding indexing and retrieval.
    """

    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "rag"):
        """
        Initialize the Qdrant connector with the given host, port, and collection name.
        """
        self.qdrant_client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]

        # Create the collection if it does not exist
        if collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
        else:
            print(f"Collection {collection_name} already exists.")

    def index_embeddings(
            self,
            documents: List[List[str]],
            embeddings: List[List[List[ndarray]]],
            metadata: Optional[List[List[Dict[str, Any]]]] = None,
            ids: Optional[List[str]] = None  # Ignored in this basic implementation
    ) -> None:
        """
        Index the embeddings with associated metadata.

        :param documents: A list of documents, where each document is a list of text chunks (strings).
        :param embeddings: A list of lists of embeddings, where each embedding is a list of numpy arrays
                            corresponding to the text chunks in the documents.
        :param metadata: Optional list of lists of metadata dictionaries corresponding to each text chunk.
        :param ids: Optional list of unique IDs for each document. If not provided, a default ID will be generated.

        :return: None
        """
        payload = []
        point_id = 0

        for doc_index, doc in enumerate(documents):
            for chunk_index, chunk in enumerate(doc):
                try:
                    # Use a basic number as the point ID.
                    current_id = point_id
                    point_id += 1

                    # Retrieve the embedding for the current chunk.
                    embedding_to_index = embeddings[doc_index][chunk_index]

                    # Retrieve metadata for the current chunk if available. Otherwise, use an empty dict.
                    if metadata is not None and doc_index < len(metadata) and chunk_index < len(metadata[doc_index]):
                        chunk_metadata = metadata[doc_index][chunk_index]
                    else:
                        chunk_metadata = {}

                    chunk_payload = chunk_metadata.copy()
                    chunk_payload["text"] = chunk

                    # Create the point structure for Qdrant
                    point = PointStruct(
                        id=current_id,
                        vector=embedding_to_index,
                        payload=chunk_payload
                    )
                    payload.append(point)
                except Exception as e:
                    print(f"Error preparing point for document {doc_index} chunk {chunk_index}: {e}")

        # Insert payload to Qdrant
        status = self.qdrant_client.upsert(collection_name=self.collection_name, points=payload)
        print(status.status)

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
        try:
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                with_payload=True,
                limit=top_k
            ).points

            # Extract the text and metadata from each point.
            context = [point.payload.get('text', '') for point in results]
            metadata = [point.payload for point in results]

            return context, metadata

        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            return [], []