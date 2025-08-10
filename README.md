# LegalEagle 🦅 — AI-Powered Contract Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-161616?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-008080?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)

A sophisticated RAG (Retrieval-Augmented Generation) chatbot designed to demystify complex legal documents. Upload statutes, contracts, or case law and get answers in plain English, backed by direct citations from the source text.

---

## 🌟 Introduction

Navigating the dense language of legal documents is a significant challenge. LegalEagle tackles this by leveraging the power of Large Language Models (LLMs) combined with a robust RAG pipeline. Instead of relying on the LLM's generalized knowledge, LegalEagle grounds its responses in the specific documents you provide. This ensures answers are accurate, relevant, and trustworthy, making it an indispensable tool for lawyers, paralegals, and anyone needing to quickly understand legal text.



---

## ✨ Key Features

* **PDF Document Upload**: Seamlessly upload any legal document in PDF format.
* **Natural Language Queries**: Ask complex questions about the document in plain English.
* **Source-Grounded Answers**: Receive answers generated directly from the content of your document, minimizing hallucinations and errors.
* **Cited Sources**: Every answer is accompanied by expandable excerpts from the original text, allowing for immediate verification.
* **Intuitive Chat Interface**: A clean and simple UI built with Streamlit for an excellent user experience.
* **Robust Error Handling**: Resilient against API rate limits and cloud infrastructure race conditions.

---

## 🛠️ Tech Stack & Architecture

LegalEagle is built with a modern, scalable stack designed for AI applications.

* **Frontend**: Streamlit
* **Backend & Orchestration**: Python, LangChain
* **LLM & Embeddings**: OpenAI (GPT-4o, text-embedding-3-small)
* **Vector Database**: Pinecone (Serverless)
* **PDF Parsing**: PyMuPDF

### RAG Architecture

The application follows a classic and effective Retrieval-Augmented Generation pipeline:



1.  **Ingestion**: The user uploads a PDF document via the Streamlit interface.
2.  **Parsing & Chunking**: The PDF's text is extracted page by page using PyMuPDF. It's then split into smaller, semantically meaningful chunks using `langchain_text_splitters`.
3.  **Embedding**: Each text chunk is converted into a numerical vector representation (an embedding) using OpenAI's embedding model. This is done with a rate-limiting loop to accommodate free-tier API limits.
4.  **Indexing**: The embeddings and their corresponding text/metadata are stored in a Pinecone serverless vector index for efficient similarity searching.
5.  **Retrieval**: When a user asks a question, the query is embedded, and Pinecone is searched to find the most relevant text chunks from the document.
6.  **Generation**: The retrieved chunks and the original question are passed to an OpenAI LLM (GPT-4o) with a prompt instructing it to answer based *only* on the provided context.
7.  **Citation**: The final answer and the source chunks used to generate it are displayed to the user.

---

## 🚀 Getting Started

You can run this project locally by following these steps.

### Prerequisites

* Python 3.11+
* An [OpenAI API Key](https://platform.openai.com/api-keys) with a billing plan set up (to avoid rate limits).
* A [Pinecone API Key](https://www.pinecone.io/) (the free tier is sufficient).

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/legaleagle.git](https://github.com/your-username/legaleagle.git)
    cd legaleagle
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The `requirements.txt` file contains all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a file named `.env` in the root of the project folder and add your API keys:
    ```
    OPENAI_API_KEY="sk-..."
    PINECONE_API_KEY="your-pinecone-api-key"
    ```

### Running the Application

Launch the Streamlit app with the following command:
```bash
streamlit run app.py