from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
import os
import warnings
warnings.filterwarnings("ignore")


# Load past missions into a vector store
def initialize_vector_store():
    if not os.path.exists("db"):
        os.makedirs("db")
    
    loader = TextLoader("Scanning_and_Motion_Planning.txt")  # Store past missions here
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(texts, OpenAIEmbeddings(model="text-embedding-ada-002"), persist_directory="db")
    vector_store.persist()
    
    return vector_store

# Retrieve relevant past missions
def retrieve_relevant_missions(vector_store, query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in retrieved_docs])

initialize_vector_store()