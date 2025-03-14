"""
Main script to run the project

Author: Srihari Raman, Alexander Zhen, Shreesh Dassarkar
"""
from src.db_connectors.chroma_connector import ChromaConnector
from src.db_connectors.redis_connector import RedisConnector
from src.pipeline import RAG
from src.embedding_connectors.ollama_embed import OllamaEmbed
from src.llm_connectors.ollama_llm import OllamaLLM


if __name__ == "__main__":
    # Initialize the embedding model
    embedding_model = OllamaEmbed(model_name="nomic-embed-text")

    # Initialize the LLM model
    llm = OllamaLLM(model_name="tinyllama:latest")

    # Initialize the vector databases
    chroma_db = ChromaConnector()
    redis_db = RedisConnector()
    dbs = [redis_db, chroma_db]

    # Index sample data into both vector databases
    for db in dbs:
        print("Indexing sample data into database:", db.__class__.__name__)

        documents = ["Cookies are yummy", "Trucks are fast", "Yummy is defined as a taste that is delicious.",
                     "Fast is defined as moving quickly."]
        embeddings = [embedding_model.generate_embeddings(doc) for doc in documents]
        metadata = [{"file": "file1", "page": 1, "chunk": 1}, {"file": "file2", "page": 2, "chunk": 2},
                    {"file": "file3", "page": 3, "chunk": 3}, {"file": "file4", "page": 4, "chunk": 4}]

        db.index_embeddings(documents, embeddings, metadata)

        print("Embeddings indexed successfully.")

        # Create the RAG pipeline
        rag_pipeline = RAG(embedding_model, db, llm)

        # Example query
        query = "What is yummy?"
        response = rag_pipeline.run(query, top_k=1)
        print("Response:", response)

        print("\n\n")

    print("Experiment completed successfully.")

