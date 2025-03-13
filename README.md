## Introduction

Welcome to **DS4300-Class-RAG**, a Retrieval-Augmented Generation (RAG) system designed with a modular, plug-and-play architecture. This project leverages multiple vector databases (Redis, Chroma, Milvus) and local Large Language Models (LLMs) to deliver high-quality, context-aware responses. By following the instructions below, you can quickly set up your environment, run the databases in Docker, and test a sample workflow.

## Repository Structure
```
DS4300-Class-RAG/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/
│   ├── db_connectors/
│   │   ├── __init__.py
│   │   └── chroma.py  # Sample script to demonstrate Chroma usage
│   ├── embedding_connectors/
│   │   ├── __init__.py
│   │   └── ...  # Concrete embedding model connectors
│   ├── llm_connectors/
│   │   ├── __init__.py
│   │   └── ...  # Concrete LLM connectors
│   └── utils/
│       ├── __init__.py
│       ├── db_model.py  # Abstract base class for DB connectors
│       ├── embedding_model.py
│       ├── llm_model.py
│       ├── pipeline.py  # Core pipeline orchestrating RAG
│       └── main.py  # Potential entry point to run pipeline
├── docker-compose.yml  # Defines containers for Redis, Chroma, and Milvus
├── requirements.txt  # Python dependencies
└── README.md  # Project documentation
```

In this repository:

- **`data/`**: Placeholder directory for storing raw or processed data files.  
- **`docs/`**: Documentation related to the project (reports, design documents, etc.).  
- **`notebooks/`**: Jupyter notebooks for experimentation and prototyping.  
- **`scripts/`**: Utility scripts or command-line tools.  
- **`src/`**: Main source code, subdivided into:
  - **`db_connectors/`**: Concrete implementations for different databases (e.g., `chroma.py`, `redis.py`, `milvus.py`).  
  - **`embedding_connectors/`**: Classes for embedding models.  
  - **`llm_connectors/`**: Classes for local LLM integrations.  
  - **`utils/`**: Contains abstract base classes (`db_model.py`, `embedding_model.py`, `llm_model.py`), the pipeline orchestrator (`pipeline.py`), and a sample entry point (`main.py`).  
- **`docker-compose.yml`**: Defines containerized services for Redis, Chroma, and Milvus.  
- **`requirements.txt`**: Python dependencies for the project.  
- **`README.md`**: This documentation file, providing setup instructions and usage details.

## Project Setup
## Project System Setup

### 1. Create a Conda Environment

1. **Install Conda (if not already installed):**  
   Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).


2. **Create & Activate a New Environment:**  
   Create an environment named `ds4300-rag` with Python 3.12.9:
   ```bash
   conda create -n ds4300-rag python=3.12.9
   conda activate ds4300-rag
    ```
   
### 2. Clone the Repository
1. **Clone the repository:**  
   ```bash
   git clone <repo-url>
    ```

### 3. Install Dependencies
1. **Install the dependencies from project root folder:**
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Ensure the dependencies are installed:**
   ```bash
   pip list
   ```

### 4. Start the Docker Containers
1. **Start the Docker containers for Redis, Chroma, and Milvus:**
   ```bash
   docker-compose up -d
   ```
5. **Ensure the containers are running:**
   ```bash
   docker ps
   ```
   
### 5. TO CHANGE: Run the pipeline
1. **Run the sample pipeline:**
   ```bash
   python src/db_connectors/chroma.py
   ```

### 6. Remove Docker Containers
1. **Stop and remove the Docker containers (NO PERSISTENCE):**
   ```bash
   docker-compose down -v
   ```
   
2. **To ensure persistence when stopping and starting the containers:**
   ```bash
   docker-compose down
   ```