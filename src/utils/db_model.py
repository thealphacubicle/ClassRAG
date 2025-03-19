from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from numpy import ndarray

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
    def index_embeddings(
            self,
            documents: List[List[str]],
            embeddings: List[List[List[ndarray]]],  # Updated type
            metadata: Optional[List[List[Dict[str, Any]]]] = None,
            ids: Optional[List[str]] = None
    ) -> None:
        """
        Index multiple embeddings and documents with associated metadata into the database.

        Args:
            documents (List[List[str]]): List of documents to index. Each document is a list of string chunks.
            embeddings (List[List[float]]): List of embeddings for each document’s chunks.
            metadata (Optional[List[List[Dict[str, Any]]]]): List of metadata dicts associated with the documents.
            ids (Optional[List[str]]): List of custom IDs for the documents.
        """
        pass

    @abstractmethod
    def query_db(self, query_embedding: list, top_k: int = 1) -> Tuple[List[str], List[Dict]]:
        """Query the database with an embedding and return the top_k results.

        Args:
            query_embedding (list): Embedding form of the query.
            top_k (int): Number of results to return.


        Returns:
            context (List): List of documents retrieved from the database.
            metadata (Dict): List of all metadata retrieved from the database.
        """
        pass