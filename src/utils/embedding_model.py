from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """
    Abstract class for embedding models as part of the RAG pipeline.

    This class defines the interface for embedding models that are used in the RAG pipeline.

    Attributes:
        None

    Required Methods:
        generate_embeddings(text: str) -> list: Generate embeddings for the given text.

    Author: Srihari Raman
    """
    @abstractmethod
    def generate_embeddings(self, text: str) -> list:
        """Generate embeddings for the given text."""
        pass
