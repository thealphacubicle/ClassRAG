from pdfminer.high_level import extract_text
from typing import List, Tuple, Dict
import os
import re

file_path = "C:\\Users\\thisi\\OneDrive\\Desktop\\DS4300-P02\\DS4300-Class-RAG\\data\\raw_data"

def clean_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Clean the input text by removing unnecessary characters and formatting.
    This includes:
    - Replacing multiple newlines with a single newline
    - Replacing bullet characters with a dash and a space
    - Removing any leading "o" at the beginning of a line
    - Removing punctuation (if specified)
    - Replacing multiple spaces or tabs with a single space
    - Removing extra spaces 

    :param text: The raw text to clean.
    :param remove_punctuation: If True, punctuation will be removed.
    :return: The cleaned text.
    """
    # Replace multiple newlines with a single newline to keep paragraphs separate
    text = re.sub(r'\n+', '\n', text)
    
    # Replace bullet characters with a dash and a space
    text = re.sub(r'[•–—]', '- ', text)
    
    # Remove any leading "o" at the beginning of a line
    text = re.sub(r'^\s*o\s+', '', text, flags=re.M)
    
    # Remove punctuation except for periods, commas, question marks, exclamation marks, and hyphens
    if remove_punctuation:
        text = re.sub(r'[^\w\s\.,?!-]', '', text)
    
    # Replace multiple spaces or tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove extra spaces at the start and end of each line
    lines = [line.strip() for line in text.splitlines()]
    
    # Add a period to any line that doesn't already end with punctuation
    new_lines = []
    for line in lines:
        # If the line is not empty and doesn't end with . ! or ?
        if line and not re.search(r'[.!?]$', line):
            line += "."
        new_lines.append(line)
    
    return "\n".join(new_lines)

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
            # Clean the extracted text before storing it
            cleaned_text = clean_text(text)
            text_dict[file] = cleaned_text
            
        return text_dict

def chunk_text(text_dict: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Tuple[List[str], List[Dict[str, str]]]:
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
                "File name": filename,
                "Chunk #": len(chunks),
                "Chunk size (char)": len(chunk),
                "Chunk overlap": chunk_overlap
            })

        chunked[filename] = chunks  
        metadata[filename] = chunk_meta

    return list(chunked.values()), list(metadata.values())

# Testing it out
txt = extract_data(file_path)
chunk_list, metadata_list = chunk_text(txt, 500, 50)
print(metadata_list[0])
print(chunk_list[0])

sample_text = """
   This is a sample text.   It includes extra spaces, bullet points like • and –,
and multiple newlines.

Random    text!
"""

cleaned = clean_text(sample_text)
print("Cleaned Text:\n", cleaned)



