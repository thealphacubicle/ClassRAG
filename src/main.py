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
        """You are a helpful AI assistant. Use the following context to answer the query as accurately as possible.""",
        "You are a smart AI assistant well-versed in all things computer science. "
        "Use the following context to answer the query as accurately as possible.",
        """You are a genius in all things related to databases and computer science. Use the following context to 
        answer the query as accurately as possible."""
    ]

    # Index sample data into both vector databases
    all_data = []
    for db in dbs:
        for embedding_model in embedding_models:
            for llm in llms:
                print("Indexing sample data into database:", db.__class__.__name__)
                print("Using embedding model:", embedding_model.model_name)
                print("Using LLM model:", llm.model_name)

                # Two documents, each split into two coherent chunks.
                chunks_for_each_document = [
                    [
                        "AVL Trees are self-balancing binary search trees.",
                        "They maintain balance through rotations and structured insertions."
                    ],
                    [
                        "B+ Trees are a type of self-balancing tree data structure.",
                        "They are widely used in databases for efficient indexing."
                    ]
                ]

                # Generate embeddings for each chunk in each document.
                # The resulting structure is a list (per document) of lists (per chunk) of embedding vectors.
                embedding_for_chunks = [
                    [embedding_model.generate_embeddings(chunk) for chunk in document]
                    for document in chunks_for_each_document
                ]

                # Create metadata for each chunk.
                # Here we use list comprehensions to generate metadata entries that correspond to each chunk.
                metadata_for_chunks = [
                    [
                        {
                            "file_name": "sample_data.pdf",
                            "chunk_number": idx + 1,  # Using 1-indexing for readability.
                            "chunk_size": len(chunk),  # Example: use the length of the chunk.
                            "chunk_overlap": 50,
                            "model_type": embedding_model.model_name
                        }
                        for idx, chunk in enumerate(chunks_for_each_document[0])
                    ],
                    [
                        {
                            "file_name": "b+_trees_data.pdf",
                            "chunk_number": idx + 1,
                            "chunk_size": len(chunk),
                            "chunk_overlap": 50,
                            "model_type": embedding_model.model_name
                        }
                        for idx, chunk in enumerate(chunks_for_each_document[1])
                    ]
                ]

                # Memory and time tracking for embedding indexing
                process = psutil.Process()
                mem_start = process.memory_info().rss
                st = time()

                db.index_embeddings(chunks_for_each_document, embedding_for_chunks, metadata_for_chunks)

                et = time()
                mem_end = process.memory_info().rss
                time_taken_to_index = et - st
                mem_taken_to_index = mem_end - mem_start

                print("Time taken to index:", time_taken_to_index, "seconds")
                print("Memory increased by", mem_taken_to_index / (1024 * 1024), "MB during indexing")
                print("Embeddings indexed successfully.")

                # Create the RAG pipeline
                rag_pipeline = RAG(embedding_model, db, llm)

                for prompt in base_prompts:

                    # Example query
                    process = psutil.Process()
                    mem_start = process.memory_info().rss
                    st = time()

                    query = "What are AVL trees?"
                    response, query_metadata = rag_pipeline.run(query, base_prompt=prompt, top_k=1)

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
