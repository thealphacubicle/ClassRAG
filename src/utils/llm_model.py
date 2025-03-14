from abc import ABC, abstractmethod


class LLMModel(ABC):
    """
    Abstract class for a language model. This class should be subclassed to implement a specific language model.

    Attributes:
        None

    Required Methods:
        generate_response (text: str) -> str: Generate a response based on the provided prompt.

    Author: Srihari Raman
    """
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response based on the provided prompt.

        :param prompt: The prompt to generate a response for.
        :return: The generated response.
        """
        pass