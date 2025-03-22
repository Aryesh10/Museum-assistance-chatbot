from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app= Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
HUGGINGFACE_API_KEY=os.environ.get('HUGGINGFACE_API_KEY')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['HUGGINGFACE_API_KEY']=HUGGINGFACE_API_KEY

embeggings= download_embeddings()

index_name='museumbot'

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeggings
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})

llm = Ollama(model="mistral")
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

qa_chain=create_stuff_documents_chain(llm, prompt)
rag_chain=create_retrieval_chain(retriever, qa_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__=='__main__':
    app.run(host="0.0.0.0", port=8000, debug= True)