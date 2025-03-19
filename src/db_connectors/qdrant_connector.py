from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.utils.db_model import DBModel
import ollama
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
        Index the embeddings with associated metadata for each chunk of every document into Qdrant,
        using basic integer IDs.

        :param documents: List of documents (each document is a list of text chunks).
        :param embeddings: List of embeddings for each document's chunks.
        :param metadata: List of metadata for each document's chunks (optional).
        :param ids: List of custom IDs for the documents (ignored here).
        :return: None
        """
        payload = []
        point_id = 0  # Use a simple global counter for point IDs.

        for doc_index, doc in enumerate(documents):
            for chunk_index, chunk in enumerate(doc):
                try:
                    # Use a basic unsigned integer as the point ID.
                    current_id = point_id
                    point_id += 1

                    # Retrieve the embedding for the current chunk.
                    embedding_to_index = embeddings[doc_index][chunk_index]

                    # Retrieve metadata for the current chunk if available; otherwise, use an empty dict.
                    if metadata is not None and doc_index < len(metadata) and chunk_index < len(metadata[doc_index]):
                        chunk_metadata = metadata[doc_index][chunk_index]
                    else:
                        chunk_metadata = {}

                    # Optionally include the text in the payload.
                    chunk_payload = chunk_metadata.copy()
                    chunk_payload["text"] = chunk

                    # Create the point structure for Qdrant.
                    point = PointStruct(
                        id=current_id,
                        vector=embedding_to_index,
                        payload=chunk_payload
                    )
                    payload.append(point)
                except Exception as e:
                    print(f"Error preparing point for document {doc_index} chunk {chunk_index}: {e}")

        # Upsert the prepared payload into Qdrant.
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

if __name__ == "__main__":
    qdb_connector = QdrantConnector()
    print("Qdrant connector initialized.")  # Test the connector initialization
    print(qdb_connector.qdrant_client.get_collections())

    #Example upsert
    documents = ["Cookies are yummy", "Trucks are fast", "Yummy is defined as a taste that is delicious.",
                 "Fast is defined as moving quickly."]
    embeddings = [ollama.embeddings(model="nomic-embed-text", prompt=doc)["embedding"] for doc in documents]
    metadata = [{"text": doc} for doc in documents]
    qdb_connector.index_embeddings(documents, embeddings, metadata)
    print("Embeddings indexed successfully.")  # Test the indexing function
    print(qdb_connector.qdrant_client.get_collection("rag"))  # Check the collections in Qdrant

    # Example search
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt="What is yummy?")["embedding"]
    context, meta = qdb_connector.query_db(query_embedding, top_k=2)
    print("Context:", context)
    print("Metadata:", meta)  # Test the query function