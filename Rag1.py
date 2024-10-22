# -*- coding: utf-8 -*-
import streamlit as st
from PyPDF2 import PdfReader
from google.cloud import aiplatform
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

api_key = "your_api_key"
genai.configure(api_key=api_key)

emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = genai.GenerativeModel('gemini-1.5-flash')

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2.reshape(-1))
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def query_emb(query):
    query_embedding = emb.encode([query])
    return query_embedding

def chunk_emb(text_chunks):
    
    chunk_embeddings = emb.encode(text_chunks)
    return chunk_embeddings
    
def similar(text_chunks, query_embedding, top_k=5):
   
    similar_docs = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = cosine_similarity(chunk_embedding, query_embedding)
        similar_docs.append((text_chunks[i], similarity))

    similar_docs.sort(key=lambda x: x[1], reverse=True)
    return similar_docs[:top_k]

def question_text(retrieved_docs, query):
   

    model = genai.GenerativeModel('gemini-1.5-flash')

    # Combine relevant text chunks into a single string
    relevant_text = " ".join([doc for doc, _ in retrieved_docs])

    response = model.generate_content(f"The below information has been extracted from the links of a website . Please extract info and answer the question based on the info extracted :{query}:/n/n Text:{relevant_text}")
    return response.candidates[0].content.parts[0].text

st.title("PDF Question Answering Application")


uploaded_file = st.file_uploader("Choose a PDF file:", accept_multiple_files=False, type=['pdf'])

if uploaded_file is not None:
    st.write("File Received!")
    

    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content


    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

  
    chunk_embeddings = chunk_emb(texts)

    st.write("Choose your actions:")
    user_input = st.text_area("Put in your query:", value="")
    answer_button = st.button("Get Answer")
    
    if answer_button and user_input:
        # st.write("You entered:", user_input)
        query = user_input
        query_embedding = query_emb(query)  # Use pre-calculated embedding
        # Retrieve the most relevant document chunks based on the query
        retrieved_docs = similar(texts, query_embedding, top_k=5)
        answer = question_text(retrieved_docs,query)
        st.subheader("Answer")
        st.write(answer)
