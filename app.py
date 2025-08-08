# app.py

import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import pinecone
import time

# Core LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# UPDATED: OpenAI components moved to langchain_openai
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# UPDATED: Pinecone integration moved to langchain_pinecone
from langchain_pinecone import Pinecone

# --- UTILITY FUNCTIONS ---

def extract_document_data(pdf_file):
    """
    Extracts text and metadata (page numbers) from a PDF file.
    Returns a list of LangChain Document objects.
    """
    documents = []
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text:  # Only add pages with text
                documents.append(Document(page_content=text, metadata={'page': i + 1}))
    return documents

def get_text_chunks(documents):
    """Splits documents into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def get_or_create_vectorstore(chunked_documents, index_name):
    """
    Creates or gets a vector store from text chunks.
    Deletes existing vectors in the index to ensure a fresh start.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Using a specific, cost-effective model
    
    # Initialize connection to Pinecone
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        st.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name, 
            dimension=1536, # OpenAI's text-embedding-3-small dimension
            metric='cosine',
            spec=pinecone.ServerlessSpec(cloud='aws', region='us-east-1') # Recommended spec
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pc.Index(index_name)
    st.info("Clearing previous document from memory...")
    index.delete(deleteAll=True)
    
    st.info("Embedding document...")
    vectorstore = Pinecone.from_documents(
        documents=chunked_documents, 
        embedding=embeddings, 
        index_name=index_name
    )
    return vectorstore

def create_conversation_chain(vectorstore):
    """Creates a question-answering chain."""
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- STREAMLIT UI ---

def main():
    load_dotenv()
    st.set_page_config(page_title="LegalEagle 🦅", page_icon="⚖️")
    st.header("LegalEagle — AI-Powered Contract Assistant 🦅")
    
    PINECONE_INDEX_NAME = "legaleagle-prod"
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Your Document")
        pdf_doc = st.file_uploader("Upload your PDF and click 'Process'", type="pdf")
        if st.button("Process"):
            if pdf_doc is not None:
                with st.spinner("Processing document... This may take a moment."):
                    docs = extract_document_data(pdf_doc)
                    text_chunks = get_text_chunks(docs)
                    vectorstore = get_or_create_vectorstore(text_chunks, PINECONE_INDEX_NAME)
                    st.session_state.conversation = create_conversation_chain(vectorstore)
                    st.session_state.messages = [] 
                    st.success("Document processed! You can now ask questions about it.")

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
                            st.info(f"**Source (Page {doc.metadata['page']}):**\n\n{doc.page_content}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("Please upload and process a document first.")

if __name__ == '__main__':
    main()