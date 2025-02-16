import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import logging
from groq import InternalServerError
from groq import RateLimitError
from langchain_core.exceptions import LangChainException
import time
import paramiko



from dotenv import load_dotenv
load_dotenv()

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}  /* Hides the entire top-right menu */
        header {visibility: hidden;}  /* Hides the header */
        footer {visibility: hidden;}  /* Hides the footer */
    </style>
""", unsafe_allow_html=True)

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

# Define FAISS storage path
FAISS_INDEX_PATH = "./faiss_index"

VALID_USERS = ["5745401", "1234567", "7654321"]  # Add more as needed

# Function to authenticate user based on the list
def authenticate(username):
    if username in VALID_USERS:
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        return True
    else:
        return False

# Login Page
def login_page():
    st.title("Login")

    username = st.text_input("Enter your FedEx ID:")
    login_button = st.button("Login")

    if login_button:
        if authenticate(username):
            st.success("Authentication successful! Loading app...")
            st.rerun()  # Redirect to the main app
        else:
            st.error("Invalid FedEx ID. Please try again.")


# If authenticated, load the main RAG application
def rag_application():
    print("inside if auth")
    if "vector" not in st.session_state:

        st.session_state.embeddings=HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings':True}

    )
        # Check if FAISS index exists
        if os.path.exists(FAISS_INDEX_PATH):
            print("Loading FAISS index from storage...")
            st.session_state.vectors = FAISS.load_local(FAISS_INDEX_PATH, st.session_state.embeddings, allow_dangerous_deserialization=True)
        else:
            print("FAISS index not found. Creating new embeddings...")
            st.session_state.loader = PyPDFDirectoryLoader("./OreData")
            st.session_state.docs = st.session_state.loader.load()

            # Split text into smaller chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            print("Total documents loaded:", len(st.session_state.docs))
            print("Total chunks after splitting:", len(st.session_state.final_documents))
            print("starting FAISS indexing")
            # Store the document embeddings in FAISS
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            # Debug: Check how many documents are actually stored in FAISS
            print("Total documents indexed in FAISS:", len(st.session_state.vectors.index_to_docstore_id))
            print("done with FAISS indexing")

            # Save FAISS index
            st.session_state.vectors.save_local(FAISS_INDEX_PATH)
            print("FAISS index saved successfully.")


    st.title("ChatGroq Demo")
    llm=ChatGroq(groq_api_key=groq_api_key,
                model_name="mixtral-8x7b-32768")

    prompt=ChatPromptTemplate.from_template(
    """
    You must **only answer using the provided context**.
    If the answer is not found, say **"I don’t know"**.
    <context>
    {context}
    </context>
    Question: {input}
    Answer:
    """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10, "search_type": "mmr"})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    prompt=st.text_input("Input you prompt here")

    if prompt:
        start=time.process_time()
        try:
            response=retrieval_chain.invoke({"input":prompt})
        except (InternalServerError, LangChainException) as e:
            logging.error(f"Groq API error: {e}")
            response = {"answer": "Service unavailable. Please try again later.", "context": []}  # Graceful fallback
        except Exception as e:
            logging.critical(f"Unexpected error: {e}")  # Catch unexpected issues
            response = {"answer": "An unexpected error occurred. Please try again later.", "context": []}
        print("Response time :",time.process_time()-start)
        st.write(response['answer'])

        # Show an error message in the UI if something went wrong
        if response["answer"] in ["Service unavailable. Please try again later.", "An unexpected error occurred. Please try again later."]:
            st.error(response["answer"])  # Show an error message in red

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            st.write("Retrieved Documents:")
            for i, doc in enumerate(response["context"]):
                st.write(f"Document {i+1}: {doc.page_content}")
                st.write("--------------------------------")

# Show Login Page if not authenticated
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login_page()
else:
    rag_application()