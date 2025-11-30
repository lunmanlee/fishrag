from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = TextLoader('data/sturgeon_corpus.txt')
documents = loader.load()