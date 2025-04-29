# import json
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Load data
# with open('/../data/community_data.json', 'r') as f:
#     documents = json.load(f)  # Assume list of dicts like [{"text": "Birth customs..."}]

# # Load embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Embed documents
# texts = [doc['text'] for doc in documents]
# embeddings = model.encode(texts)

# # Create FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)  # Simple, fast index
# index.add(np.array(embeddings))

# # Save FAISS index
# faiss.write_index(index, '../embeddings/faiss_index.bin')

# # (Optional) Save ID mapping
# id_map = {str(i): text for i, text in enumerate(texts)}
# with open('../embeddings/id_mapping.json', 'w') as f:
#     json.dump(id_map, f)

# print("✅ FAISS Index created and saved!")
# print(f"Model device: {model.device}")


# import json
# import faiss
# import numpy as np
# import os
# from sentence_transformers import SentenceTransformer

# # Get the absolute path to the data directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(os.path.dirname(current_dir), 'data', 'community_data.json')
# embeddings_dir = os.path.join(os.path.dirname(current_dir), 'embeddings')

# # Create embeddings directory if it doesn't exist
# os.makedirs(embeddings_dir, exist_ok=True)

# # Load data
# with open(data_path, 'r') as f:
#     documents = json.load(f)

# # Load embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Embed documents
# texts = [doc['content'] for doc in documents]  # Changed from 'text' to 'content'
# embeddings = model.encode(texts)

# # Create FAISS index
# dimension = embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(embeddings))

# # Save FAISS index
# faiss.write_index(index, os.path.join(embeddings_dir, 'faiss_index.bin'))

# # Save ID mapping
# id_map = {str(i): text for i, text in enumerate(texts)}
# with open(os.path.join(embeddings_dir, 'id_mapping.json'), 'w') as f:
#     json.dump(id_map, f)

# print("✅ FAISS Index created and saved!")
# print(f"Model device: {model.device}")


import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Get the absolute path to the data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(current_dir), 'data', 'community_data.json')
embeddings_dir = os.path.join(os.path.dirname(current_dir), 'embeddings')

# Create embeddings directory if it doesn't exist
os.makedirs(embeddings_dir, exist_ok=True)

def chunk_content(documents, max_chunk_size=512):
    chunked_docs = []
    for doc in documents:
        content = doc['content']
        chunks = [c.strip() for c in content.split('.') if len(c.strip()) > 50]
        
        for chunk in chunks:
            chunked_docs.append({
                'community': doc['community'],
                'topic': doc['topic'],
                'subtopic': doc['subtopic'],
                'content': chunk + '.'
            })
    return chunked_docs

# Load data and model
with open(data_path, 'r') as f:
    documents = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Process and embed documents
chunked_documents = chunk_content(documents)
texts = [doc['content'] for doc in chunked_documents]
embeddings = model.encode(texts)

# Create and populate FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save FAISS index
faiss.write_index(index, os.path.join(embeddings_dir, 'faiss_index.bin'))

# Save ID mapping
id_map = {str(i): text for i, text in enumerate(texts)}
with open(os.path.join(embeddings_dir, 'id_mapping.json'), 'w') as f:
    json.dump(id_map, f)

print("✅ FAISS Index created and saved!")
print(f"Model device: {model.device}")