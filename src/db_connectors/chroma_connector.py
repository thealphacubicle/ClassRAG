import chromadb
from src.utils.db_model import DBModel
from typing import List, Dict
from typing import Any, Optional, Tuple
from numpy import ndarray

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

    def index_embeddings(
            self,
            documents: List[List[str]],
            embeddings: List[List[List[ndarray]]],  # Updated type
            metadata: Optional[List[List[Dict[str, Any]]]] = None,
            ids: Optional[List[str]] = None
    ) -> None:
        """
        Index the embeddings with associated metadata.

        :param documents: List of documents to index (each document is a list of chunks).
        :param embeddings: List of embeddings for each document's chunks.
        :param metadata: List of metadata for each document's chunks (optional).
        :param ids: List of custom IDs for the documents (optional).
        """
        # Generate a base ID for each document if none provided.
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Index each chunk of each document individually.
        for i, document in enumerate(documents):
            for j, chunk in enumerate(document):
                try:
                    # Optionally, create a unique ID for each chunk.
                    unique_id = f"{ids[i]}_{j}"

                    # Extract the corresponding embedding and metadata.
                    embedding_to_index = embeddings[i][j]
                    metadata_to_index = metadata[i][j] if metadata is not None else {}

                    # Wrap values in lists if the collection expects batch inputs.
                    self.collection.add(
                        ids=[unique_id],
                        embeddings=[embedding_to_index],
                        documents=[chunk],
                        metadatas=[metadata_to_index]
                    )

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
        try:
            # Perform the search in Chroma.
            results = self.collection.query(
                query_embedding,
                n_results=top_k,
                include=["documents", "embeddings", "metadatas"]
            )

            # Extract the text chunks and metadata.
            # 'documents' should be a list of text chunks (strings) and
            # 'metadatas' should be a list of dictionaries for each chunk.
            documents = results.get("documents", [])[0]
            metadatas = results.get("metadatas", [])

            return documents, metadatas

        except Exception as e:
            print(f"Error querying database: {e}")
            return [], []




