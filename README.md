# AI Document Analyst

This project is a high-performance **RAG (Retrieval-Augmented Generation)** system that allows you to upload PDF documents and perform intelligent queries on their content in real time.

It leverages the power of **Llama 3** models through **Groq** infrastructure to deliver ultra-fast responses, and uses **LangChain** for orchestrating the data flow.

## Features

- **PDF Processing:** Intelligent loading and segmentation of documents.
- **Semantic Search:** Implementation of **FAISS** for efficient information retrieval.
- **Modern Interface:** Interactive UI built with **Gradio**.
- **Security:** Credential management using environment variables (`.env` file).

## Technologies Used

- **LLM:** Groq (Llama-3-8b-8192 or similar).
- **Embeddings:** `all-MiniLM-L6-v2` from Hugging Face.
- **Framework:** LangChain.
- **Vector Database:** FAISS.
- **Frontend:** Gradio.

## Requirements and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tu-usuario/AI_Document_Analyst.git
   cd AI_Document_Analyst