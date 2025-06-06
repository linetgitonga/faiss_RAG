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

# def chunk_content(documents, max_chunk_size=512):
#     chunked_docs = []
#     for doc in documents:
#         content = doc['content']
#         # Improve chunking for name-related content
#         chunks = []
#         sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 0]
        
#         current_chunk = []
#         current_length = 0
        
#         for sentence in sentences:
#             # Preserve name lists and their meanings in the same chunk
#             if 'name' in sentence.lower() or any(marker in sentence.lower() for marker in ['called', 'means', 'meaning']):
#                 if current_chunk:
#                     chunks.append(' '.join(current_chunk) + '.')
#                     current_chunk = []
#                 chunks.append(sentence + '.')
#             else:
#                 current_chunk.append(sentence)
#                 if len(' '.join(current_chunk)) >= max_chunk_size:
#                     chunks.append(' '.join(current_chunk) + '.')
#                     current_chunk = []
        
#         if current_chunk:
#             chunks.append(' '.join(current_chunk) + '.')
        
#         for chunk in chunks:
#             if len(chunk.strip()) > 50:  # Keep meaningful chunks only
#                 chunked_docs.append({
#                     'community': doc['community'],
#                     'topic': doc['topic'],
#                     'subtopic': doc['subtopic'],
#                     'content': chunk,
#                     'is_name_related': 'name' in chunk.lower()  # Flag for name-related content
#                 })
#     return chunked_docs

# # Update the ID mapping to include metadata
# id_map = {
#     str(i): {
#         'content': doc['content'],
#         'community': doc['community'],
#         'topic': doc['topic'],
#         'subtopic': doc['subtopic'],
#         'is_name_related': doc.get('is_name_related', False)
#     } for i, doc in enumerate(chunked_documents)
# }

# # Load data and model
# with open(data_path, 'r', encoding='utf-8') as f:
#     documents = json.load(f)

# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Process and embed documents
# chunked_documents = chunk_content(documents)
# texts = [doc['content'] for doc in chunked_documents]
# embeddings = model.encode(texts)

# # Create and populate FAISS index
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
        chunks = []
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 0]
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if 'name' in sentence.lower() or any(marker in sentence.lower() for marker in ['called', 'means', 'meaning']):
                if current_chunk:
                    chunks.append(' '.join(current_chunk) + '.')
                    current_chunk = []
                chunks.append(sentence + '.')
            else:
                current_chunk.append(sentence)
                if len(' '.join(current_chunk)) >= max_chunk_size:
                    chunks.append(' '.join(current_chunk) + '.')
                    current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk) + '.')
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                chunked_docs.append({
                    'community': doc['community'],
                    'topic': doc['topic'],
                    'subtopic': doc['subtopic'],
                    'content': chunk,
                    'is_name_related': 'name' in chunk.lower()
                })
    return chunked_docs

# Load data and model
with open(data_path, 'r', encoding='utf-8') as f:
    documents = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Process and embed documents
chunked_documents = chunk_content(documents)

# Update the ID mapping to include metadata
id_map = {
    str(i): {
        'content': doc['content'],
        'community': doc['community'],
        'topic': doc['topic'],
        'subtopic': doc['subtopic'],
        'is_name_related': doc.get('is_name_related', False)
    } for i, doc in enumerate(chunked_documents)
}

# Create embeddings
texts = [doc['content'] for doc in chunked_documents]
embeddings = model.encode(texts)

# Create and populate FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save FAISS index and ID mapping
faiss.write_index(index, os.path.join(embeddings_dir, 'faiss_index.bin'))
with open(os.path.join(embeddings_dir, 'id_mapping.json'), 'w', encoding='utf-8') as f:
    json.dump(id_map, f, ensure_ascii=False, indent=2)

print("✅ FAISS Index created and saved!")
print(f"Model device: {model.device}")
print(f"Number of chunks created: {len(chunked_documents)}")