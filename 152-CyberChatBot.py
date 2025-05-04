import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_huggingface import HuggingFaceEmbeddings


groq_api_key = st.secrets["GROQ_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """

    Answer the question based on the context provided only. If the context does not contain the answer, say "I don't know.
    <context>
    {context}
    <context>
    Question: {input}

    
    """)

def create_vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-Minilm-L6-v2")



        print(st.session_state.embeddings)

        st.session_state.loader = PyPDFDirectoryLoader("Research-Paper")

        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,

        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

        st.write(f"Loaded {len(st.session_state.docs)} documents.")
        st.write(f"Split into {len(st.session_state.final_documents)} chunks.")


user_prompt = st.text_input("Enter your question:")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector DB ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    
    retriever=st.session_state.vectors.as_retriever()

    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()

    response=retrieval_chain.invoke({'input':user_prompt})

    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
