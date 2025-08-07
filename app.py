import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
import pinecone

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
    # The splitter can directly work on the list of Document objects
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def get_or_create_vectorstore(chunked_documents, index_name):
    """
    Creates or gets a vector store from text chunks.
    Deletes existing vectors in the index to ensure a fresh start.
    """
    embeddings = OpenAIEmbeddings()
    
    # Initialize connection to Pinecone
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        # Create a new index if it doesn't exist
        st.info(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name, 
            dimension=1536, # OpenAI's embedding dimension
            metric='cosine'
        )
        # We need to wait for the index to be ready
        import time
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pc.Index(index_name)
    
    # Clear out the index to start fresh with the new document
    st.info("Clearing previous document from memory...")
    index.delete(deleteAll=True)
    
    st.info("Embedding document...")
    # Use from_documents to ingest the chunks with their metadata
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
        search_kwargs={"k": 3} # Retrieve top 3 relevant chunks
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
    
    # Use a consistent index name
    PINECONE_INDEX_NAME = "legaleagle-prod"
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Your Document")
        pdf_doc = st.file_uploader("Upload your PDF and click 'Process'", type="pdf")
        if st.button("Process"):
            if pdf_doc is not None:
                with st.spinner("Processing document... This may take a moment."):
                    # 1. Extract Documents with metadata
                    docs = extract_document_data(pdf_doc)
                    
                    # 2. Chunk Documents
                    text_chunks = get_text_chunks(docs)
                    
                    # 3. Create Vector Store (this handles clearing and embedding)
                    vectorstore = get_or_create_vectorstore(text_chunks, PINECONE_INDEX_NAME)
                    
                    # 4. Create Conversation Chain and store in session
                    st.session_state.conversation = create_conversation_chain(vectorstore)
                    
                    # Reset chat history for the new document
                    st.session_state.messages = [] 
                    st.success("Document processed! You can now ask questions about it.")

    # Main chat interface
    st.divider()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your document..."):
        if st.session_state.conversation:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"query": prompt})
                answer = response['result']
                source_docs = response['source_documents']

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Show Sources"):
                        for doc in source_docs:
                            # The 'page' metadata is now correctly preserved
                            st.info(f"**Source (Page {doc.metadata['page']}):**\n\n{doc.page_content}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("Please upload and process a document first.")


if __name__ == '__main__':
    main()