import os
from pdfminer.high_level import extract_text
import json

# Chunkng parameters: WILL MODIFY AS NEEDED LATER
CHUNK_SIZE = 500  
CHUNK_OVERLAP = 50 

# Update file path as needed
data_dir = r"C:\Users\shree\DS4300-Class-RAG\data\raw_data"

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

pdf_chunks = {}

# Get all PDF files in the directory
files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

if files:
    for file in files:
        path = os.path.join(data_dir, file)

        # Extract text, chunk it, and store it
        text = extract_text(path)
        chunks = chunk_text(text)
        pdf_chunks[file] = chunks

        print(f"{file}: {len(chunks)} chunks created.")

else:
    print("No files found")

# Store the text as json
output_path = "chunked_text.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(pdf_chunks, f, ensure_ascii=False, indent=4)

print(f"Chunked text saved to {output_path}")


