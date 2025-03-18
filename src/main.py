"""
Main script to run the project

Author: Srihari Raman, Alexander Zhen, Shreesh Dassarkar
"""
import psutil
import os
from hashlib import sha256
import csv
from time import time
from src.db_connectors.chroma_connector import ChromaConnector
from src.db_connectors.redis_connector import RedisConnector
from src.db_connectors.qdrant_connector import QdrantConnector
from src.pipeline import RAG
from src.embedding_connectors.ollama_embed import OllamaEmbed
from src.llm_connectors.ollama_llm import OllamaLLM

if __name__ == "__main__":
    # Initialize the embedding model
    embedding_models = [OllamaEmbed(model_name=model_name) for model_name in ['nomic-embed-text']]

    # Initialize the LLM model
    llms = [
        OllamaLLM(model_name="tinyllama:latest"),
        OllamaLLM(model_name="deepseek-r1:1.5b"),
        OllamaLLM(model_name="gemma3:latest")
    ]

    # Initialize the vector databases
    dbs = [ChromaConnector(), RedisConnector(), QdrantConnector()]

    # Index sample data into both vector databases
    all_data = []
    for db in dbs:
        for embedding_model in embedding_models:
            for llm in llms:
                print("Indexing sample data into database:", db.__class__.__name__)
                print("Using embedding model:", embedding_model.model_name)
                print("Using LLM model:", llm.model_name)

                documents = [
                    "AVL Trees are self-balancing binary search trees.",
                    "B+ Trees are a type of self-balancing tree data structure that maintains sorted data.",
                    "Key-value pair databases store data as a collection of key-value pairs.",
                    "Document-based databases store data in document format, typically JSON or BSON.",
                    "Examples of document-based databases include MongoDB and CouchDB.",
                    "Non-relational databases are designed to handle large volumes of unstructured data."
                ]
                embeddings = [embedding_model.generate_embeddings(doc) for doc in documents]
                metadata = [
                    {"file": "avl_trees.pdf", "page": 1, "chunk": 1, "model_type": embedding_model.model_name,
                     "text": documents[0]},
                    {"file": "b_plus_trees.pdf", "page": 1, "chunk": 1, "model_type": embedding_model.model_name,
                     "text": documents[1]},
                    {"file": "key_value_databases.pdf", "page": 1, "chunk": 1, "model_type": embedding_model.model_name,
                     "text": documents[2]},
                    {"file": "document_based_databases.pdf", "page": 1, "chunk": 1, "model_type": embedding_model.model_name,
                     "text": documents[3]},
                    {"file": "document_based_databases.pdf", "page": 2, "chunk": 4, "model_type": embedding_model.model_name,
                     "text": documents[4]},
                    {"file": "non_relational_databases.pdf", "page": 1, "chunk": 1, "model_type": embedding_model.model_name,
                     "text": documents[5]}
                ]

                # Memory and time tracking for embedding indexing
                process = psutil.Process()
                mem_start = process.memory_info().rss
                st = time()

                db.index_embeddings(documents, embeddings, metadata)

                et = time()
                mem_end = process.memory_info().rss
                time_taken_to_index = et - st
                mem_taken_to_index = mem_end - mem_start

                print("Time taken to index:", time_taken_to_index, "seconds")
                print("Memory increased by", mem_taken_to_index / (1024 * 1024), "MB during indexing")
                print("Embeddings indexed successfully.")

                # Create the RAG pipeline
                rag_pipeline = RAG(embedding_model, db, llm)

                # Example query
                process = psutil.Process()
                mem_start = process.memory_info().rss
                st = time()

                query = "What are document-based databases, and give examples of document-based databases."
                response, query_metadata = rag_pipeline.run(query, top_k=1)

                et = time()
                mem_end = process.memory_info().rss
                time_taken_to_rag = et - st
                mem_taken_to_rag = mem_end - mem_start

                print("Response:", response)
                print("RAG took", time_taken_to_rag, "seconds")
                print("Memory increased by", mem_taken_to_rag / (1024 * 1024), "MB during RAG execution")
                print("\n\n")

                all_data.append({
                    "embedding_model": embedding_model.model_name,
                    "llm_model": llm.model_name,
                    "db_type": db.__class__.__name__,
                    "query": query,
                    "query_id": str(hash(query)),
                    "response": response,
                    "time_taken_to_index": time_taken_to_index,
                    "mem_taken_to_index": mem_taken_to_index / (1024 * 1024),
                    "time_taken_to_rag": time_taken_to_rag,
                    "mem_taken_to_rag": mem_taken_to_rag / (1024 * 1024)
                })

    # Ensure the directory exists
    os.makedirs(os.path.join(os.getcwd(), "../data/experiment_data"), exist_ok=True)
    path = os.path.join(os.getcwd(), "../data/experiment_data/experiment_results.csv")

    # Write/append the results to CSV
    COLS = [
        "embedding_model",
        "llm_model",
        "db_type",
        "query",
        "query_id",
        "response",
        "time_taken_to_index",
        "mem_taken_to_index",
        "time_taken_to_rag",
        "mem_taken_to_rag"
    ]

    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        if os.stat(path).st_size == 0:
            writer.writeheader()

        for row in all_data:
            writer.writerow(row)

    print("Experiment completed successfully.")
