# RAG (Retrieval Augmented Generation) System

## Overview
This project implements a document retrieval system focused on Luhya community cultural information using FAISS vector search and Sentence Transformers.

## Project Structure
```
RAG/
├── app/
│   ├── create_index.py    # Creates FAISS index from documents
│   └── query_index.py     # Handles user queries and retrieval
├── data/
│   └── community_data.json # Cultural information dataset
├── embeddings/            # Generated embeddings and indices
└── README.md
```

## Features
- Document chunking and embedding
- Fast similarity search using FAISS
- Relevance scoring for search results
- Command-line interface for queries

## Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```bash
# Create and activate virtual environment
python -m venv Virtualenv
.\Virtualenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```text
faiss-cpu==1.7.4
sentence-transformers==2.2.2
numpy==1.24.3
```

## Usage

1. Create the search index:
```bash
python app/create_index.py
```

2. Query the system:
```bash
python app/query_index.py
```

3. Enter your questions about Luhya cultural practices when prompted.
Type 'exit' to quit the program.

## Performance
- Embedding Model: `all-MiniLM-L6-v2`
- Vector Dimension: 384
- Search Speed: < 1 second per query

## Future Improvements
- [ ] Add LLM integration for answer generation
- [ ] Implement web interface
- [ ] Add more cultural data
- [ ] Improve chunking strategy
- [ ] Add multilingual support

## License
MIT License

## Contributors
- [Your Name]