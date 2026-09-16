# 👩‍💼 HR Policy Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

Users can upload an HR Policy PDF and ask questions about company policies. The application retrieves the most relevant sections from the uploaded document and uses Groq's `openai/gpt-oss-20b` model to generate an answer.

## 🚀 Features

- Upload HR Policy PDF
- Extract PDF text using PyMuPDF
- Split policy into searchable chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Generate answers using Groq
- Display retrieved sources
- Streamlit web interface
- No database required
- No local server required after deployment

## 🧠 RAG Pipeline

```text
HR Policy PDF
      ↓
PyMuPDF
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Sentence Transformers
      ↓
Embeddings
      ↓
FAISS Vector Index
      ↓
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Relevant Policy Chunks
      ↓
Groq
openai/gpt-oss-20b
      ↓
Final Answer
