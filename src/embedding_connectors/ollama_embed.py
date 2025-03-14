import ollama
from src.utils.embedding_model import EmbeddingModel

class OllamaEmbed(EmbeddingModel):
    """
    Class to generate embeddings using embedding models available on the Ollama platform.
    """
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def generate_embeddings(self, text: str) -> list:
        """
        Generate embeddings for the given text using the Nomic API.
        :param text: Text to generate embeddings for.
        :return: Embeddings for the given text.
        """
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return response["embedding"]

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []