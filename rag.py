from encodings import search_function
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQA

import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = TextLoader('sturgeon_corpus.txt')
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
document_chunks = splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(document_chunks, embeddings)
retriever = vector_store.as_retriever(
  search_type="similarity",
  search_kwargs={"k": 5} # give the top 5 most revelant document
)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=retriever
)

query = "Who is Herman?"
response = qa_chain.invoke({"query": query})

print(response)