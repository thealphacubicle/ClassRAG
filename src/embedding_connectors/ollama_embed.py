import ollama
from src.utils.embedding_model import EmbeddingModel
import numpy as np
from ollama._types import ResponseError

class OllamaEmbed(EmbeddingModel):
    """
    Class to generate embeddings using embedding models available on the Ollama platform.
    """
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def generate_embeddings(self, text: str, **kwargs) -> np.ndarray:
        """
        Generate embeddings for the given text using the Nomic API.
        :param text: Text to generate embeddings for.
        :return: Embeddings for the given text.
        """
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return np.array(response["embedding"])

        # Catch the error if the model is pulled and run the command and retry
        except ResponseError as re:
            if re.status_code == 404:
                print(f"Model {self.model_name} not found. Attempting to pull the model.")
                try:
                    ollama.pull(self.model_name)
                    response = ollama.embeddings(model=self.model_name, prompt=text)
                    return np.array(response["embedding"])

                except Exception as e:
                    print(f"Error pulling model: {e}")
                    return np.array([])

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.array([])