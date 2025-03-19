from pdfminer.high_level import extract_text
from typing import List, Tuple, Dict

def extract_data(file_path: str) -> str:
    """
    Extracts the text from the given
    :param file_path:
    :return: Raw text from the file
    """
    return extract_text(file_path)

def chunk_text(raw_text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Tuple[List[str], List[Dict[str, str]]]:
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
    return list_of_chunks, list_of_dictionaries
    pass

