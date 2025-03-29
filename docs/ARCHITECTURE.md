# Architecture

This document provides an overview of our **Retrieval-Augmented Generation (RAG)** architecture. The goal of this design is to enable a flexible, plug-and-play approach to building different RAG pipelines using interchangeable components.

---

## Overview

Our RAG system is structured to allow the seamless combination of different database connectors, embedding models, and LLMs. Think of each connector as a "lego block" that can be swapped out for another, enabling rapid experimentation and customization of your pipeline.

### Key Objectives

1. **Modular Design**: Each connector handles a specific task within the RAG pipeline. This modularity promotes clarity and maintainability.
2. **Extensibility**: New connectors (for databases, embedding models, or LLMs) can be added with minimal changes to existing code.
3. **Reusability**: Connectors can be mixed and matched to create diverse RAG configurations without rewriting core logic.

---


### Main Components

1. **`data/`**  
   - Contains datasets, raw data, and experiment-related data (e.g., CSV logs).

2. **`docs/`**  
   - Houses any additional documentation or references.

3. **`notebooks/`**  
   - Contains Jupyter notebooks used for analysis and experimentation.

4. **`src/`**  
   - Core source code for the project.
   - **`db_connectors/`**: Database connector implementations (Redis, Chroma, Qdrant).
   - **`embedding_connectors/`**: Embedding model connectors (e.g., OllamaEmbed).
   - **`llm_connectors/`**: Language model connectors (e.g., OllamaLLM).
   - **`utils/`**: Utility functions and abstract classes.
   - **`pipeline.py`**: Core pipeline logic for connecting the different components.
   - **`main.py`**: Entry point or orchestrator for the RAG pipeline.
   - **`test.py`**: Sample file to run 1 specific RAG architecture

5. **`docker-compose.yml`**  
   - Container orchestration file (e.g., for running Redis or other services).

6. **`requirements.txt`**  
   - Python dependencies.

---

## RAG Architecture

### 1. Plug-and-Play Connectors

Our RAG pipeline is composed of three primary connector types, each fulfilling a critical role:

1. **DBConnector**  
   - Defines how embeddings are indexed and how similarity searches are performed.  
   - Returns the *k* most relevant chunks of information to the LLM.

2. **EmbeddingConnector**  
   - Generates 1024-dimensional vector representations of text or data.
   - Ensures consistency in how text is embedded before being stored or retrieved.

3. **LLMConnector**  
   - Responsible for generating the final responses.
   - Consumes both the query and the retrieved chunks to produce contextually rich answers.

### 2. Concrete Implementations

Within each connector category, we have specific implementations:

#### DBConnectors
- **RedisConnector**  
  Uses Redis Stack for vector storage and similarity searches.

- **ChromaConnector**  
  Leverages the Chroma vector database for efficient embedding indexing.

- **QdrantConnector**  
  Employs the Qdrant vector search engine for retrieval tasks.

#### EmbeddingConnectors
- **OllamaEmbed**  
  Connects to the Ollama distribution to produce embeddings.  
  > **Note:** Ollama provides on-device language models, which can be leveraged for various NLP tasks.

#### LLMConnectors
- **OllamaLLM**  
  Interfaces with Ollama-based language models to generate textual responses.

---

## Pipeline Flow

Below is a simplified view of how a query moves through the system:

1. **Input Query**  
   A user provides a question or prompt.

2. **Embedding**  
   The query is converted into a vector representation via an **EmbeddingConnector**.

3. **Similarity Search**  
   The **DBConnector** takes the query embedding and retrieves the top *k* most relevant text chunks from the database.

4. **LLM Generation**  
   The **LLMConnector** (e.g., `OllamaLLM`) uses the retrieved chunks alongside the original query to generate a contextually informed response.

5. **Output**  
   The final answer is returned to the user.

---

## Benefits of This Architecture

1. **Flexibility**  
   Swap out databases or models with minimal changes to the codebase.

2. **Scalability**  
   Each connector can be optimized independently. For instance, you might upgrade your DBConnector to a more powerful vector database without changing the rest of the pipeline.

3. **Ease of Experimentation**  
   Researchers and developers can quickly test how different databases or models affect retrieval quality and response generation.

4. **Code Reusability**  
   Common logic (e.g., chunking, vector transformations) can be centralized and reused across different connector implementations.

---

## Future Extensions

1. **Additional DBConnectors**  
   - Integration with other vector databases (e.g., Milvus, Faiss).

2. **More Embedding Models**  
   - Hugging Face Transformers, custom embeddings, or specialized domain embeddings.

3. **Diverse LLMs**  
   - OpenAI GPT, local Large Language Models, or domain-specific fine-tuned models.

4. **Advanced Pipeline Management**  
   - Workflow orchestration tools, caching layers, and more sophisticated indexing strategies.

---
