"""
Main script to run the project

Author: Srihari Raman, Alexander Zhen, Shreesh Dassarkar
"""
import psutil
import os
import csv
from time import time
from src.db_connectors.chroma_connector import ChromaConnector
from src.db_connectors.redis_connector import RedisConnector
from src.db_connectors.qdrant_connector import QdrantConnector
from src.pipeline import RAG
from src.embedding_connectors.ollama_embed import OllamaEmbed
from src.llm_connectors.ollama_llm import OllamaLLM
import src.utils.preprocessing as tp

if __name__ == "__main__":
    # Initialize the embedding model
    embedding_models = [OllamaEmbed(model_name=model_name) for model_name in ["bge-large",
                                                                              "snowflake-arctic-embed2",
                                                                              "mxbai-embed-large"]]
    # Initialize the LLM model
    llms = [
        OllamaLLM(model_name="tinyllama:latest"),
        OllamaLLM(model_name="deepseek-r1:1.5b"),
        OllamaLLM(model_name="gemma3:latest")
    ]

    # Initialize the vector databases
    dbs = [
        ChromaConnector(),
        RedisConnector(),
        QdrantConnector()
        ]

    # Initialize prompts to the LLM
    base_prompts = [
        """You are an AI assistant. Use the following context to answer the query as accurately as possible.""",
        """You are an AI assistant. Use the following context to answer the query as accurately as possible.
        Directly answer my question using pertinent information from the context.""",
        """You are a genius AI assistant. Use the following context to answer the query as accurately as possible.""",
        """You are a genius AI assistant. Use the following context to answer the query as accurately as possible.
        Directly answer my question using pertinent information from the context.""",
        "Use the following context to answer the query as accurately as possible.",
        "Use the following context to answer the query as accurately as possible. "
        "Directly answer my question using pertinent information from the context."
    ]

    queries = [
        "What is an AVL tree?",
        "How are document databases different from relational databases?",
        "What is an elephant?"
    ]

    # Index sample data into both vector databases
    all_data = []
    for db in dbs:
        for embedding_model in embedding_models:
            for llm in llms:
                print("Indexing sample data into database:", db.__class__.__name__)
                print("Using embedding model:", embedding_model.model_name)
                print("Using LLM model:", llm.model_name)

                # Extract text from the raw data source (PDFs)
                chunks_for_each_document, metadata_for_chunks = tp.chunk_text(tp.extract_data("../data/raw_data"),
                                                                              100, 50)

                # Add the embedding model name to the metadata
                for doc in metadata_for_chunks:
                    for chunk in doc:
                        chunk["model_type"] = embedding_model.model_name

                # Generate embeddings for each chunk in each document.
                embedding_for_chunks = [
                    [embedding_model.generate_embeddings(chunk) for chunk in document]
                    for document in chunks_for_each_document
                ]

                # Memory and time tracking for embedding indexing
                process = psutil.Process()
                mem_start = process.memory_info().rss
                st = time()

                # Index the embeddings into the database
                db.index_embeddings(chunks_for_each_document, embedding_for_chunks, metadata_for_chunks)

                et = time()
                mem_end = process.memory_info().rss
                time_taken_to_index = et - st
                mem_taken_to_index = mem_end - mem_start

                print("Time taken to index:", time_taken_to_index, "seconds")
                print("Memory increased by", mem_taken_to_index / (1024 * 1024), "MB during indexing")
                print("Embeddings indexed successfully.")
    #
                # Create the RAG pipeline
                rag_pipeline = RAG(embedding_model, db, llm)

                for prompt in base_prompts:

                    # Example query
                    process = psutil.Process()
                    mem_start = process.memory_info().rss
                    st = time()

                    query = "What do DB systems aim to minimize with relation to HDD and SDD?"
                    response, query_metadata = rag_pipeline.run(query, base_prompt=prompt, top_k=1)

                    et = time()
                    mem_end = process.memory_info().rss
                    time_taken_to_rag = et - st
                    mem_taken_to_rag = mem_end - mem_start

                    print("*"*50)
                    print("Prompt:", prompt)
                    print("Response:", response)
                    print("RAG took", time_taken_to_rag, "seconds")
                    print("Memory increased by", mem_taken_to_rag / (1024 * 1024), "MB during RAG execution")
                    print("\n\n")

                    all_data.append({
                        "embedding_model": embedding_model.model_name,
                        "llm_model": llm.model_name,
                        "db_type": db.__class__.__name__,
                        'base_prompt': prompt,
                        'base_prompt_id': str(hash(prompt)),
                        "query": query,
                        "query_id": str(hash(query)),
                        "response": response,
                        "num_documents": len(chunks_for_each_document),
                        "total_chunks": sum(len(doc) for doc in chunks_for_each_document),
                        "time_taken_to_index": time_taken_to_index,
                        "mem_taken_to_index": mem_taken_to_index / (1024 * 1024),
                        "time_taken_to_rag": time_taken_to_rag,
                        "mem_taken_to_rag": mem_taken_to_rag / (1024 * 1024)
                    })

    # Ensure the directory exists
    os.makedirs(os.path.join(os.getcwd(), "../data/experiment_data"), exist_ok=True)
    path = os.path.join(os.getcwd(), "../data/experiment_data/experiment_results.csv")

    # Write/append the results to CSV
    COLS = list(all_data[0].keys())

    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        if os.stat(path).st_size == 0:
            writer.writeheader()

        for row in all_data:
            writer.writerow(row)

    print("Experiment completed successfully.")
