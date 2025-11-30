from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = TextLoader('data/sturgeon_corpus.txt')
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
document_chunks = splitter.split_documents(documents)