# import faiss
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Load FAISS index and mapping
# index = faiss.read_index('../embeddings/faiss_index.bin')
# with open('../embeddings/id_mapping.json', 'r') as f:
#     id_map = json.load(f)

# # Load embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def search(query, k=3):
#     query_vec = model.encode([query])
#     distances, indices = index.search(np.array(query_vec), k)
#     results = [id_map[str(idx)] for idx in indices[0]]
#     return results

# if __name__ == "__main__":
#     while True:
#         user_query = input("\n🔍 Enter your question: ")
#         answers = search(user_query)
#         for ans in answers:
#             print(f"- {ans}")


import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# from llm_integration import LLMProcessor


# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_dir = os.path.join(os.path.dirname(current_dir), 'embeddings')
data_path = os.path.join(os.path.dirname(current_dir), 'data', 'community_data.json')
index_path = os.path.join(embeddings_dir, 'faiss_index.bin')
mapping_path = os.path.join(embeddings_dir, 'id_mapping.json')

# Load FAISS index and mapping
index = faiss.read_index(index_path)
with open(mapping_path, 'r') as f:
    id_map = json.load(f)



# Initialize LLM
# llm = LLMProcessor()


# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'text': id_map[str(idx)],
            'score': 1 - dist/2  # Convert L2 distance to similarity score
        })
    return results

def format_results(results):
    print("\nSearch Results:\n" + "="*50)
    for i, result in enumerate(results, 1):
        score = result['score']
        print(f"\n[Match {i}] Relevance: {score:.2%}")
        print("-"*50)
        print(result['text'])
        print("-"*50)

if __name__ == "__main__":
    print("RAG Search System - Type 'exit' to quit")
    while True:
        user_query = input("\n🔍 Enter your question: ")
        if user_query.lower() == 'exit':
            break
        answers = search(user_query)
        format_results(answers)






# # LLM INTEGRATION

# def process_query(query):
#     # Get relevant contexts
#     search_results = search(query)
    
#     # Combine contexts
#     context = "\n".join([r['text'] for r in search_results])
    
#  # Generate LLM response
#     llm_response = llm.generate_response(context, query)
    
#     return {
#         'answer': llm_response,
#         'sources': search_results
#     }

# if __name__ == "__main__":
#     print("RAG Search System - Type 'exit' to quit")
#     while True:
#         user_query = input("\n🔍 Enter your question: ")
#         if user_query.lower() == 'exit':
#             break
            
#         response = process_query(user_query)
#         print("\n🤖 Answer:")
#         print(response['answer'])
#         print("\n📚 Sources:")
#         for i, source in enumerate(response['sources'], 1):
#             print(f"\n[Source {i}] Relevance: {source['score']:.2%}")
#             print("-"*50)
#             print(source['text'])