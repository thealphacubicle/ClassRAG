from src.db_connectors.redis_connector import RedisConnector
from src.pipeline import RAG
from src.embedding_connectors.ollama_embed import OllamaEmbed
from src.llm_connectors.ollama_llm import OllamaLLM
import src.utils.preprocessing as tp


if __name__ == '__main__':
    NEED_TO_INDEX = True

    # Define parameters
    db = RedisConnector()
    embedding_model = OllamaEmbed(model_name="bge-large")
    llm = OllamaLLM(model_name="gemma3:latest")

    # If you need to index, index into database
    if NEED_TO_INDEX:
        chunks_for_each_document, metadata_for_chunks = tp.chunk_text(tp.extract_data("../data/raw_data"),
                                                                      100, 50)

        # Index into database
        embedding_for_chunks = [
            [embedding_model.generate_embeddings(chunk) for chunk in document]
            for document in chunks_for_each_document
        ]

        db.index_embeddings(chunks_for_each_document, embedding_for_chunks, metadata_for_chunks)

    else:
        print("Data already indexed. Skipping indexing.")


    # Run RAG
    rag_pipeline = RAG(embedding_model, db, llm)
    query = """What is disk-based indexing and why is it important for database systems?"""
    prompt = ("Use the following context to answer the query as accurately as possible. Be as direct and concise as possible and answer within 2 sentences.")
    response, query_metadata = rag_pipeline.run(query, base_prompt=prompt, top_k=1)
    print("Response:", response)

    # Cosine simialrity between query and response
    print("Similarity:", query_metadata["response_similarity_to_query"])