from abc import ABC, abstractmethod


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
    def index_embeddings(self, embeddings: list, metadata: list):
        """Index the embeddings with associated metadata."""
        pass

    @abstractmethod
    def query_db(self, query_embedding: list, top_k: int) -> list:
        """Query the database with an embedding and return the top_k results."""
        pass