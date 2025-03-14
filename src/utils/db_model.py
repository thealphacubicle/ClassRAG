from abc import ABC, abstractmethod
from typing import Optional, List, Dict

class DBModel(ABC):
    """
    This file contains the abstract class for the DBModel and defines the interface for the database connectors that
    are used in the RAG pipeline to store and query embeddings.

    Attributes:
        None

    Required Methods:
        index_embeddings(embeddings: list, metadata: list): Index the embeddings with associated metadata.
        query_db(query_embedding: list, top_k: int) -> list: Query the database with an embedding and return the top_k
        results.

    Author: Srihari Raman
    """
    @abstractmethod
    def index_embeddings(self, documents: list, embeddings: list, metadata: Optional[list] = None, ids: Optional[list] = None):
        """Index multiple embeddings and documents with associated metadata into the database

        Args:
            documents (list): List of documents to index.
            embeddings (list): List of embeddings to index.
            metadata (Optional[list]): List of metadata dicts associated with the documents (optional)
            ids (Optional[list]): List of custom IDs for the documents (optional)
        """
        pass

    @abstractmethod
    def query_db(self, query_embedding: list, top_k: int) -> tuple[List, Dict]:
        """Query the database with an embedding and return the top_k results.

        Args:
            query_embedding (list): Embedding form of the query.
            top_k (int): Number of results to return.


        Returns:
            context (List): List of documents retrieved from the database.
            metadata (Dict): List of all metadata retrieved from the database.
        """
        pass