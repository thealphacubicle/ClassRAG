from pdfminer.high_level import extract_text
from typing import List, Tuple, Dict, Any
import os

file_path = "/Users/srihariraman/Desktop/DS4300/DS4300_P02/DS4300-Class-RAG/data/raw_data"
def extract_data(file_path: str) -> str:
    """
    Extracts the text from the given
    :param file_path:
    :return: Raw text from the file
    """
    text_dict = {}
    files = [f for f in os.listdir(file_path) if f.lower().endswith(".pdf")]
    if files:
        for file in files:
            path = os.path.join(file_path, file)

            text = extract_text(path)
            text_dict[file] = text\
            
        return text_dict
    

def chunk_text(text_dict: str, chunk_size: int = 500, chunk_overlap: int = 50) -> tuple[
    list[list[str]], list[list[dict[str, int | Any]]]]:
    """
    Chunk the raw text into smaller pieces.
    :param raw_text: Raw text to be chunked.
    :param chunk_size: Size of each chunk.
    :param chunk_overlap: Overlap between chunks.
    """
    # For return types:
    # List of chunks is going to be a list of strings where each string is a chunk of text
    # I.E list_of_chunks = ["AVL Trees are" , "self-balancing binary" , "search trees."]

    # List of dictionaries is going to be a list of dictionaries where each dictionary is the metadata for the chunk
    """
    [{"file_name": "sample_data.pdf", "chunk_number": 1, "chunk_size": 250, "chunk_overlap": 50, "model_type": embedding_model.model_name},
                                        {"file_name": "sample_data.pdf", "chunk_number": 2, "chunk_size": 250, "chunk_overlap": 50, "model_type": embedding_model.model_name},
                                        {"file_name": "sample_data.pdf", "chunk_number": 3, "chunk_size": 250, "chunk_overlap": 50, "model_type": embedding_model.model_name}]
    """
    chunked = {}
    metadata = {}

    for filename, text in text_dict.items():
        words = text.split()
        chunks = []
        chunk_meta = []

        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

            chunk_meta.append({
                "file_name": filename,
                "chunk_number": len(chunks),
                "chunk_size": len(chunk),
                "chunk_overlap": chunk_overlap
            })

        chunked[filename] = chunks  
        metadata[filename] = chunk_meta

    return list(chunked.values()), list(metadata.values())

if __name__ == "__main__":
    txt = extract_data(file_path)
    chunk_list, metadata_list = chunk_text(txt, 500, 50)
    print(metadata_list[11])
    print(chunk_list[11])



