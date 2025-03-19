# Experiment Notes

This document outlines the setup, methodology, and collected data from our experiments using different components of the **Retrieval-Augmented Generation (RAG)** pipeline. The primary goal was to evaluate how various databases, embedding models, and language models impact retrieval quality and response generation.

---

## Overview

We designed our experiments to get an understanding of the following domains:

1. **How do different vector databases affect retrieval performance?**  
2. **Which language model produces the most accurate and contextually relevant responses?**  
3. **How do embedding models and chunking strategies influence the overall quality of the RAG output?**

To address these questions, we systematically varied multiple components in the pipeline and logged performance metrics, including runtime and memory usage.

---

## Setup

### 1. Vector Databases

We tested three different vector databases:

- **Chroma**  
- **Redis Stack**  
- **Qdrant**

Each database was responsible for indexing the embeddings and performing similarity searches to retrieve the most relevant chunks of text.

### 2. LLM Models

We employed three different language models to generate responses:

- **tinyllama**  
- **deepseek-r1:1.5b**  
- **gemma3**

Each model was evaluated on the same queries and data to compare response quality and performance.

### 3. Embedding Models

For converting text into vector representations, we tested three embedding models:

- **bge-large**  
- **snowflake-arctic-embed2**  
- **mxbai-embed-large**

These models produce 1024-dimensional vectors, which were then indexed in one of the vector databases.

### 4. Chunking & Preprocessing

We experimented with various chunking strategies (e.g., fixed-length, semantic-based) and preprocessing methods (e.g., token cleaning, lowercasing) to determine their impact on retrieval accuracy and final LLM responses.

---

## Data Collection

During each experiment, we captured the following metrics:

1. **embedding_model**  
   Name of the embedding model used to create vector representations.

2. **llm_model**  
   Language model that generated the responses.

3. **db_type**  
   Type (class name) of the database connector (e.g., `RedisConnector`, `ChromaConnector`, `QdrantConnector`).

4. **base_prompt**  
   The primary prompt or instruction text used to guide the LLM.

5. **base_prompt_id**  
   A unique identifier (hash) of the `base_prompt`.

6. **query**  
   The user’s input question or prompt.

7. **query_id**  
   A unique identifier (hash) of the `query`.

8. **response**  
   The LLM-generated output based on the retrieved chunks.

9. **num_documents**  
   The total number of documents (or document sets) considered for retrieval.

10. **total_chunks**  
    The sum of all text chunks created from the documents.

11. **time_taken_to_index**  
    The time (in seconds) required to index the documents.

12. **mem_taken_to_index**  
    Memory usage (in MB) during the indexing process.

13. **time_taken_to_rag**  
    The time (in seconds) for the RAG pipeline to retrieve relevant chunks and generate the final response.

14. **mem_taken_to_rag**  
    Memory usage (in MB) during the RAG retrieval and generation process.

---

## Logging & Analysis

All experimental runs were logged to capture the metrics listed above. We stored these logs in CSV files (e.g., `experiment_results.csv`) and analyzed them using Python notebooks located in the `notebooks/` directory.

Key points of our analysis include:

- **Comparative Performance**: Evaluating how each database, embedding model, and LLM combination performed in terms of speed, memory, and output quality.
- **Chunking Strategy Impact**: Assessing whether different chunk sizes or preprocessing methods led to more relevant or concise LLM responses.
- **Scalability Considerations**: Observing how memory and time scales as the number of documents or chunk sizes increase.

---

## Observations & Next Steps

1. **Database Trade-offs**  
   Each vector database exhibited unique performance characteristics under different loads and data sizes. Selecting the optimal database may depend on specific project requirements (e.g., speed vs. memory footprint).

2. **Model Suitability**  
   Not all LLMs performed equally well on every dataset. Larger models often provided more contextually rich responses but at a higher computational cost.

3. **Embedding Quality**  
   Embedding models influenced retrieval precision. Embeddings that better capture semantic relationships tended to improve the relevance of retrieved chunks.

4. **Future Work**  
   - **Additional Databases**: Investigate other vector databases like Milvus or Faiss.  
   - **More Embeddings**: Incorporate different embedding techniques (e.g., domain-specific or multilingual).  
   - **Refined Chunking**: Explore advanced chunking approaches (e.g., sentence-level chunking, hybrid chunking).  
   - **Benchmarking**: Implement standardized benchmarks for more robust performance comparisons.

---
