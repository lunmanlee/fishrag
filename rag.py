
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

def setup_rag_system():
  # Load the doc
  loader = TextLoader('sturgeon_corpus.txt')
  documents = loader.load()

  # Split the doc into chunks
  splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  document_chunks = splitter.split_documents(documents)

  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

  vector_store = FAISS.from_documents(document_chunks, embeddings)

  # Set up retriever
  retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # give the top 5 most revelant document
  )
  return retriever

async def get_rag_response(query: str):
  retriever = setup_rag_system()

  retrieved_docs = retriever.invoke(query)

  context = "\n".join([doc.page_content for doc in retrieved_docs])

  prompt = [f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"]

  generated_response = llm.generate(prompt)

  return generated_response