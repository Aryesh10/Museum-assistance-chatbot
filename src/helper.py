import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openpyxl import load_workbook

# Load all PDFs in directory
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Load Excel event data as langchain documents
def load_excel_as_documents(path="data/museum_events.xlsx"):
    df = pd.read_excel(path)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(page_content=content, metadata={"source": "excel"})
        documents.append(doc)
    return documents

# Split large documents into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download HuggingFace embeddings
def download_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

# Book tickets in Excel by reducing available count
def book_tickets(event_name, num_tickets, path="data/museum_events.xlsx"):
    df = pd.read_excel(path)

    matched_rows = df[df['Event Name'].str.lower() == event_name.lower()]
    if matched_rows.empty:
        return f"❌ Event '{event_name}' not found."

    idx = matched_rows.index[0]
    available = df.at[idx, 'Available Tickets']

    if available < num_tickets:
        return f"⚠️ Only {available} tickets left for '{event_name}'. Please try booking fewer tickets."

    df.at[idx, 'Available Tickets'] -= num_tickets

    # Save the Excel file with updated ticket count
    df.to_excel(path, index=False)

    return f"✅ Successfully booked {num_tickets} ticket(s) for '{event_name}'. Enjoy the event!"
