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
    embedding_models = [OllamaEmbed(model_name=model_name) for model_name in
                        ['nomic-embed-text']]

    # Initialize the LLM model
    llms = [OllamaLLM(model_name="tinyllama:latest"), OllamaLLM(model_name="deepseek-r1:1.5b"),
            OllamaLLM(model_name="gemma3:latest")]

    # Initialize the vector databases
    dbs = [ChromaConnector(), RedisConnector()]

    # Index sample data into both vector databases
    for db in dbs:
        for embedding_model in embedding_models:
            for llm in llms:
                print("Indexing sample data into database:", db.__class__.__name__)
                print("Using embedding model:", embedding_model.model_name)
                print("Using LLM model:", llm.model_name)

                documents = ["Cookies are yummy", "Trucks are fast", "Yummy is defined as a taste that is delicious.",
                             "Fast is defined as moving quickly."]
                embeddings = [embedding_model.generate_embeddings(doc) for doc in documents]
                metadata = [{"file": "file1", "page": 1, "chunk": 1, "model_type": embedding_model.model_name},
                            {"file": "file2", "page": 2, "chunk": 2, "model_type": embedding_model.model_name},
                            {"file": "file3", "page": 3, "chunk": 3, "model_type": embedding_model.model_name},
                            {"file": "file4", "page": 4, "chunk": 4, "model_type": embedding_model.model_name}]

                db.index_embeddings(documents, embeddings, metadata)

                print("Embeddings indexed successfully.")

                # Create the RAG pipeline
                rag_pipeline = RAG(embedding_model, db, llm)

                # Example query
                query = "What is yummy?"
                response, metadata = rag_pipeline.run(query, top_k=1)
                print("Response:", response)
                print("\n\n")

                # Write the results to CSV
                path = "data/experiment_data"


    print("Experiment completed successfully.")

