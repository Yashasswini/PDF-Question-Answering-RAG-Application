PDF Question Answering Application

This project is a Streamlit application that leverages Generative AI (powered by Gemini), RAG (Retrieval-Augmented Generation), and LLMs (Large Language Models) to extract and answer questions from a PDF document. The application uses a combination of text embeddings, vector search, and generative AI to find relevant information in a document and generate precise answers to user queries.

Features:

1. Upload a PDF document.
2. Automatically extract text from the uploaded PDF.
3. Split the text into chunks for efficient processing.
4. Use sentence embeddings from sentence-transformers/all-MiniLM-L6-v2.
5. Perform a similarity search to retrieve the most relevant document chunks based on a user query.
6. Utilize Gemini Generative AI to generate a detailed response based on the extracted information.

Technology Stack:
1. Streamlit: Interactive UI for uploading PDFs and entering queries.
2. Generative AI (Gemini): Used to generate human-like responses.
3. Sentence-Transformers: Embedding model for text similarity (all-MiniLM-L6-v2).
4. FAISS: Vector store for efficient similarity search.
5. PyPDF2: To read PDF documents.
6. LangChain: Text splitting and processing.

How It Works:

1. Upload PDF: The application reads the uploaded PDF file and extracts the raw text using PyPDF2.
2. Text Chunking: The text is split into manageable chunks using LangChain’s text splitter.
3. Query Input: The user can input any query related to the PDF content.
4. Embedding & Similarity Search: The application encodes both the query and the text chunks into embeddings using SentenceTransformers, and computes cosine similarity to retrieve the top relevant chunks.
5. Answer Generation: Using the most relevant document chunks, the Gemini model generates an answer to the query.
