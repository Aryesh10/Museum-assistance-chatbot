from src.helper import load_pdf_file, text_split, download_embeddings, load_excel_as_documents
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Load PDFs
extracted_data = load_pdf_file(data='data/pdfs')
text_chunks = text_split(extracted_data)

# Load Excel
excel_docs = load_excel_as_documents("data/database/museum_events.xlsx")

# Combine all
all_documents = text_chunks + excel_docs

# Download embeddings
embeddings = download_embeddings()

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'museumbot'

# Create index only if not exists (optional)
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Push all documents to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=all_documents,
    index_name=index_name,
    embedding=embeddings,
)
