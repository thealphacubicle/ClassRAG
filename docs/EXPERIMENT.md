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

We followed an overlay-based chunking strategy with thresholds of 100 and 200 words. Chunks were stored in a 2-D list, with each sublist representing the chunks from each document. Relevant metadata for each chunk was also passed through into a 2-D list.

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

1. **Importance of prompt engineering** 

An immportant thing we noticed was that different LLMs respond differently to minor changes in prompts. For instance, some models performed better when more supportive and encouraging language was used in the prompt. Others responded better when asked to be direct or upon specific instructions. In any case, the way a question is asked is also very significant. The output varied significantly when questions were more strictly defined versus when the questions encouraged more "creative" thinking.

2. **Model Suitability**

Not all LLMs performed equally well. The larger models performed a lot better in handling the "grey areas" that arose from certain questions, while the smaller models were a lot more context dependent, so they did not perform well with ambiguity.

3. **Embedding and Database Quality**

The choice of database and embedding model was not as significant as other factors, but did have some effects on memory, operating speed, and quality of answers.

4. **Future Work**  
   - **Additional Databases**: Investigate other vector databases like Milvus or Faiss.  
   - **More Embeddings**: Incorporate different embedding techniques (e.g., domain-specific or multilingual).  
   - **Refined Chunking**: Explore advanced chunking approaches (e.g., sentence-level chunking, hybrid chunking).  
   - **Benchmarking**: Implement standardized benchmarks for more robust performance comparisons.

---
