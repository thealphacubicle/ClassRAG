"""
Define the Ollama connector which can load models available through the Ollama API. This is an implementation of the
 LLMModel class.
"""
from src.utils.llm_model import LLMModel
import ollama
from ollama._types import ResponseError


class OllamaLLM(LLMModel):
    """
    Initialize the OllamaLLM connector with the given model name. This loads from the local Ollama API.
    """
    def __init__(self, model_name: str = "tinyllama:latest") -> None:
        """
        Initialize the OllamaLLM connector with the given model name. This loads from the local Ollama API. It uses
        TinyLlama by default.

        :param prompt: The prompt to use for the model (required to pass)
        :param model_name: The name of the model to use. Default is "tinyllama:latest".
        """
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response based on the provided prompt using the Ollama API.
        :param prompt: The prompt to generate a response for.
        :return: The generated response.
        """
        try:
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]

        except ResponseError as re:
            if re.status_code == 404:
                print(f"Model {self.model_name} not found. Attempting to pull the model.")
                try:
                    ollama.pull(self.model_name)
                    print(f"Model {self.model_name} successfully pulled.")
                    response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
                    return str(response["message"]["content"])
                except Exception as e:
                    print(f"Error pulling model: {e}")
                    return str("")

        except Exception as e:
            print(f"Error generating response: {e}")
            return str("")
