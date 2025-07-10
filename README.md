# PDF RAG QA System

**PDF RAG QA Tool** — Retrieve answers from any PDF using **FAISS**, **LangChain**, and your choice of local LLM: **Ollama (Llama3)** or **Qwen1.5**.

---

## 📌 Overview

This project is a **Retrieval-Augmented Generation (RAG)** pipeline for question answering over PDF documents.  
You can:
- Extract text from any PDF
- Chunk and embed the text with **HuggingFace embeddings**
- Store & search chunks with **FAISS**
- Generate context-based answers using either:
  - ✅ **Ollama version:** Runs **Llama3.2:3b** via local Ollama server
  - ✅ **Qwen version:** Runs **Qwen1.5-0.5B** directly via HuggingFace Transformers

All **offline**, no cloud calls needed!

---

## 🎯 Key Features

- 📄 PDF loader with LangChain `PyPDFLoader`
- ✂️ Smart chunking with `RecursiveCharacterTextSplitter`
- 🗂️ Local FAISS vector store for fast similarity search
- 🔍 Terminal-based query loop
- 🤖 Option to generate AI answers with your preferred LLM
- 🧩 Modular — choose Ollama backend or direct Transformers

---

## 📁 Repository Structure

pdf-rag-qa-system/
│
├── ollama_version/
│ ├── ollama_pdf_qa.py # Advanced Ollama-based RAG
│
├── qwen_version/
│ ├── qwen_pdf_qa.py # Qwen1.5 version


---

## ⚙️ Tech Stack

- **Python**
- **LangChain**
- **FAISS**
- **HuggingFace Embeddings**
- **Ollama (Llama3.2:3b)**
- **Qwen1.5-0.5B** via Transformers
- **PyPDFLoader**

---

