from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.utils.db_model import DBModel
import ollama

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

        # Create a collection if it doesn't exist
        if not self.qdrant_client.get_collection(collection_name):
            self.qdrant_client.create_collection(self.collection_name,
                                                 vectors_config=VectorParams(size=768, distance=Distance.COSINE))

    def index_embeddings(self, documents: list, embeddings: list, metadata: list = None, ids: list = None):
        """
        Index the embeddings with associated metadata.
        """
        # Prepare the payload
        payload = []
        for i, embedding in enumerate(embeddings):
            point = PointStruct(id=ids[i] if ids else i, vector=embedding, payload=metadata[i] if metadata else {})
            payload.append(point)

        # Upload the embeddings to Qdrant
        status = self.qdrant_client.upsert(collection_name=self.collection_name, points=payload)

        print(status.status)

    def query_db(self, query_embedding: list, top_k: int = 1) -> tuple[list[str], dict]:
        """
        Query the database with an embedding and return the top_k results.
        """
        results = self.qdrant_client.query_points(collection_name=self.collection_name, query=query_embedding, with_payload=True, limit=top_k).points

        # Unpack the results
        context, metadata = [point.payload['text'] for point in results], [point.payload for point in results]

        return context, metadata

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