# app.py

import streamlit as st
from dotenv import load_dotenv
import os
import time
import fitz  # PyMuPDF
import openai
import pinecone
# CORRECTED: Import specific exceptions from the pinecone.exceptions module
from pinecone.exceptions import ConflictException, NotFoundException

# Core LangChain components
from langchain.schema import Document
from langchain.chains import RetrievalQA

# Updated LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- UTILITY FUNCTIONS ---

def extract_document_data(pdf_file):
    """Extracts text and metadata from a PDF."""
    documents = []
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:
                documents.append(Document(page_content=text, metadata={'page': i + 1}))
    return documents

def get_text_chunks(documents):
    """Splits documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

# --- FINAL ROBUST LOGIC ---
def get_or_create_vectorstore(chunked_documents, index_name):
    """Initializes Pinecone and creates a vector store, handling all race conditions."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    try:
        pc.describe_index(index_name)
        index = pc.Index(index_name)
        # Add specific error handling for the delete operation
        try:
            st.info("Index already exists. Clearing existing data...")
            index.delete(deleteAll=True)
        except NotFoundException:
            # This can happen if the index is new and has no vectors/namespaces yet.
            # We can safely ignore it because our goal is an empty index.
            st.info("Index is new and already empty. Skipping clear.")
            pass
            
    except NotFoundException:
        st.info(f"Index not found. Attempting to create new index: {index_name}")
        try:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1')
            )
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            st.info("Index created. Waiting for it to be fully available...")
            time.sleep(10) # Extra delay for insurance
        except ConflictException:
            # This handles the race condition where the index was created
            # between the describe_index and create_index calls.
            st.info("Index was created by another process. Continuing.")
            pass

    index = pc.Index(index_name)
    st.info("Embedding document with rate-limiting... This will be slow.")
    progress_bar = st.progress(0, text="Embedding chunks...")

    for i, doc in enumerate(chunked_documents):
        try:
            embedded_doc = embeddings.embed_documents([doc.page_content])
            vector_id = f"doc_chunk_{i}"
            index.upsert(vectors=[(vector_id, embedded_doc[0], doc.metadata)])
            progress_text = f"Embedding chunk {i + 1}/{len(chunked_documents)}"
            progress_bar.progress((i + 1) / len(chunked_documents), text=progress_text)
            time.sleep(20)
        except openai.RateLimitError:
            st.error("OpenAI rate limit hit. Upgrade your OpenAI plan for faster processing.")
            return None
        except Exception as e:
            st.error(f"An error occurred during embedding: {e}")
            return None

    progress_bar.empty()
    vectorstore = LangChainPinecone(index, embeddings, "page_content")
    return vectorstore

def create_conversation_chain(vectorstore):
    """Creates a question-answering chain."""
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa_chain

# --- STREAMLIT UI ---
def main():
    load_dotenv()
    st.set_page_config(page_title="LegalEagle 🦅", page_icon="⚖️")
    st.header("LegalEagle — AI-Powered Contract Assistant 🦅")

    PINECONE_INDEX_NAME = "legaleagle"

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Your Document")
        pdf_doc = st.file_uploader("Upload your PDF and click 'Process'", type="pdf")
        if st.button("Process"):
            if pdf_doc is not None:
                with st.spinner("Processing document..."):
                    docs = extract_document_data(pdf_doc)
                    text_chunks = get_text_chunks(docs)
                vectorstore = get_or_create_vectorstore(text_chunks, PINECONE_INDEX_NAME)
                if vectorstore:
                    st.session_state.conversation = create_conversation_chain(vectorstore)
                    st.session_state.messages = []
                    st.success("Document processed! You can now ask questions.")
            else:
                st.warning("Please upload a PDF file first.")

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        if st.session_state.conversation:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"query": prompt})
                answer = response['result']
                source_docs = response['source_documents']
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Show Sources"):
                        for doc in source_docs:
                            st.info(f"**Source (Page {doc.metadata.get('page', 'N/A')}):**\n\n{doc.page_content}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("Please upload and process a document first.")

if __name__ == '__main__':
    main()