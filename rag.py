from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = TextLoader('data/sturgeon_corpus.txt')
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
document_chunks = splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(document_chunks, embeddings)