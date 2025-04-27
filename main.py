# Libraries
import os
import langchain
import langchain_community
import gdown
import faiss

# Downloading the file from Google Drive
# The file ID is the part of the URL after "id="
file_id = '1XQHTD0L55yXV8gLVFtAFXKrMbnI3YwnV'
url = f'https://drive.google.com/uc?id={file_id}'
file_path = gdown.download(url, 'downloaded_file.docx', quiet=False)

# Ingestion Pipeline
# Loading data from a docx file
from langchain_community.document_loaders import Docx2txtLoader  # Corrected case-sensitive import
loader = Docx2txtLoader(file_path)
data = loader.load()

# # Printing the content and metadata of the document
# for doc in data:
#     print(doc.page_content)  # Print the content of each document
#     print(doc.metadata)  # Print the metadata of each document

# Preprocessing the data
# Splitting the document into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    length_function=len # Using len function to calculate the length of the text 
)
texts = splitter.split_documents(data)
# print(len(texts))

#Loading the Model from huggingface for Embedding
from langchain_community.embeddings import HuggingFaceEmbeddings
embed_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
)
# Using FAISS for vector storage
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(texts, embed_model)
db.save_local("faiss_index")

#Retrieving the data Using MMR
query = "what is dom?"
results = db.max_marginal_relevance_search(query, k=4, filter=None)
# print(results[1].page_content)  # Print the content of the first result

from langchain.llms import HuggingFaceHub  

from langchain_community.llms import HuggingFaceHub
from langchain import HuggingFacePipeline
from transformers import pipeline

# Using a local Hugging Face model (no API key needed if local)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a text generation pipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, min_length=20, temperature=0.7)

# Wrap into LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)
from langchain.chains import RetrievalQA

# Load FAISS DB
new_db = FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)

# Create the RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=new_db.as_retriever(),  # Connect retriever
    chain_type="stuff"  # Simple RAG style
)
# Run the chain with a query
# query = "What is Directorate of Monitoring?"
response = qa_chain.run(query)
# print(response)

# 7. SETUP FASTAPI SERVER
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (frontend to connect)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    response = qa_chain.run(request.query)
    return {"response": response}

# Built by ALI ZULQARNAIN (AI/ML Developer) - 2025 @alizulqarnainirfan
# GitHub:alizulqarnainirfan
# LinkedIn:alizulqarnainirfan
# mail:aliirfanawan13@gmail.com