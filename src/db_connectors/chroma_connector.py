import chromadb
from src.utils.db_model import DBModel
import ollama
from typing import List, Dict


class ChromaConnector(DBModel):
    """
    Connector implementation for ChromaDB vector database (inherited from DBModel abstract class).
    """

    def __init__(self, host: str = "localhost", port: int = 8000,
                 collection_name: str = "rag"):
        """
        Initialize the ChromaDB connector with the given host, port, database name, and embedding function. The
        embedding function must either be a pre-defined function from chromadb.EmbeddingFunction or a custom
        implementation of the EmbeddingFunction protocol. Currently only supports embeddings of dimension 768.

        :param host: The host address of the ChromaDB server.
        :param port: The port number of the ChromaDB server, default is 8000.
        :param collection_name: The name of the collection to use in the database.
        """
        self.chroma_client = chromadb.HttpClient(host, port)
        self.collection_name = collection_name
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)

    def index_embeddings(self, documents: list, embeddings: list, metadata: list = None, ids: list = None):
        """
        Index the embeddings with associated metadata.

        :param documents: List of documents to index.
        :param embeddings: List of embeddings to index.
        :param metadata: List of metadata associated with the documents (optional)
        :param ids: List of custom IDs for the documents (optional)
        """
        try:
            # Check if custom IDs are passed. If not, create generic ones
            if ids is None:
                ids = [f"id{i}" for i in range(len(documents))]

            # Add the embeddings
            self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadata)

        except Exception as e:
            print(f"Error indexing embeddings: {e}")



    def query_db(self, query_embedding: list, top_k: int = 1) -> tuple[List[str], Dict]:
        """
        Query the database with an embedding and return the top_k results.
        :param query_embedding: Embedding form of the query.
        :param top_k: Number of results to return.
        :return:
            context: List of documents retrieved from the database.
            results: List of all metadata retrieved from the database.
        """
        try:
            # Perform the search
            results = self.collection.query(query_embedding, n_results=top_k,
                                            include=["documents", "embeddings", "metadatas"])

            context = results['documents'][0]

            return context, results

        except Exception as e:
            print(f"Error querying database: {e}")
            return [], {}


if __name__ == "__main__":

    # SAMPLE IMPLEMENTATION
    chroma_db = ChromaConnector()

    # Check if ChromaDB connection exists
    assert chroma_db.chroma_client.heartbeat() is not None, "ChromaDB connection failed"

    # Flush out the collection

    # Add sample documents to the database
    documents = ["Cookies are yummy", "Trucks are fast", "Yummy is defined as a taste that is delicious.",
                 "Fast is defined as moving quickly."]
    embeddings = [ollama.embeddings(model="nomic-embed-text", prompt=doc)["embedding"] for doc in documents]

    chroma_db.index_embeddings(documents, embeddings)
    print("Embeddings indexed successfully.")

    # Print out all the documents in the collection
    print("Documents in the collection:")
    print(chroma_db.collection)




