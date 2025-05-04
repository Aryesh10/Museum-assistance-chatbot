import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openpyxl import load_workbook

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def load_excel_as_documents(path="data/museum_events.xlsx"):
    df = pd.read_excel(path)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(page_content=content, metadata={"source": "excel"})
        documents.append(doc)
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

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

    df.to_excel(path, index=False)

    return f"✅ Successfully booked {num_tickets} ticket(s) for '{event_name}'. Enjoy the event!"

def is_meta_question(user_input):
    meta_keywords = [
        "what can you do", "who are you", "help", "about you", "what do you do", "your name"
    ]
    user_input = user_input.lower()
    return any(keyword in user_input for keyword in meta_keywords)

def is_museum_related(user_input):
    museum_keywords = [
        "museum", "event", "ticket", "tour", "timing", "guide", "anthropology", "book", "lecture", "price","prices"
    ]
    user_input = user_input.lower()
    return any(word in user_input for word in museum_keywords)

def is_price_query(user_input):
    keywords = ["ticket price", "ticket prices", "entry fee", "how much", "cost", "what's the price", "pricing"]
    user_input = user_input.lower()
    return any(k in user_input for k in keywords)


def get_ticket_price_info(path="data/database/museum_events.xlsx"):
    df = pd.read_excel(path)
    prices = df[["Event Name", "Ticket Price"]].drop_duplicates()
    message = "🎟️ Here are the ticket prices for available events:\n\n"
    for _, row in prices.iterrows():
        message += f"- {row['Event Name']}: ₹{row['Ticket Price']}\n"
    return message.strip()


