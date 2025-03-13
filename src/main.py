"""
Main script to run the project

Author: Srihari Raman, Alexander Zhen, Shreesh Dassarkar
"""
from src import *
from src.db_connectors.chroma import ChromaConnector
# from embedding_connectors import MiniLMConnector, InstructorXLConnector, NomicEmbedConnector
# from llm_connectors import MistralLLMConnector, OllamaLLMConnector
from src.pipeline import RAG

if __name__ == "__main__":

    # Create all connections to the databases
    chroma_db = ChromaConnector()

    # Index the documents into all 3 vector databases



    # Create a RAG object
    rag = RAG()

