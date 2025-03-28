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

We followed an overlay-based chunking strategy with thresholds of 100 and 200 words. Chunks were stored in a 2-D list, with each sublist representing the chunks from each document. Relevant metadata for each chunk was also passed through into a 2-D list. This approach ensured better context retention across chunks while reducing information loss at boundaries. Additionally, by maintaining metadata alongside each chunk, we improved the accuracy of retrieval and relevance in responses.

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

One key takeaway from our project was how much the wording of a prompt affects the response. Different LLMs reacted differently to small changes in phrasing. Some models performed better when prompts were more supportive and encouraging, while others responded best to direct and specific instructions. The way a question was asked also had a big impact on the output. When questions were strictly defined, responses were more structured and precise, while open-ended questions led to more creative answers. This showed that getting the right prompt is just as important as the model itself.

2. **Model Suitability**

Not all models handled complex questions the same way. Larger models were much better at dealing with uncertainty and ambiguity, while smaller models needed very clear and specific context to generate useful responses. If a prompt was too vague, smaller models often produced incomplete or overly literal answers. This made it clear that choosing the right model depends on the needs of the application—whether depth of understanding or speed and efficiency is more important.

3. **Embedding and Database Quality**

The choice of database and embedding model played a role as well, though it was less important than prompt design and model selection. Some embeddings improved retrieval accuracy, helping the model provide better answers. The database setup also influenced memory usage and response speed, but these factors were only noticeable when working with large amounts of data. While optimizing these technical aspects improved performance, we found that structuring information well and crafting effective prompts had a much bigger impact on results.

4. **Future Work**  
   - **Additional Databases**: Investigate other vector databases like Milvus or Faiss.  
   - **More Embeddings**: Incorporate different embedding techniques (e.g., domain-specific or multilingual).  
   - **Refined Chunking**: Explore advanced chunking approaches (e.g., sentence-level chunking, hybrid chunking).  
   - **Benchmarking**: Implement standardized benchmarks for more robust performance comparisons.

---
