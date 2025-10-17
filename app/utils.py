import io
import pandas as pd
import pdfplumber
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.embeddings import EMBED_MODEL_NAME # Import to use its tokenizer
from transformers import AutoTokenizer

# Load the tokenizer object from the model name
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)

# Use a tokenizer-aware text splitter for better semantic chunking
# This uses the tokenizer from the same model we use for embeddings
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, # Pass the loaded tokenizer object
    chunk_size=int(os.getenv('CHUNK_SIZE', '512')), # Chunk size in tokens
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '128')), # Overlap in tokens
    add_start_index=True,
)

def chunk_text(text: str):
    """
    Splits a long text into smaller chunks using a token-aware, recursive splitter.
    This method is superior to word-based splitting as it respects token boundaries
    and semantic units (paragraphs, sentences) better.
    """
    return text_splitter.split_text(text)

def extract_text_from_pdf_bytes(b: bytes):
    texts = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ''
            texts.append(t)
    return texts

def extract_texts_from_csv_bytes(b: bytes):
    df = pd.read_csv(io.BytesIO(b))
    
    # Basic data cleaning
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert all string columns to stripped strings and replace empty strings with NaN
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().replace(r'^\s*$', pd.NA, regex=True)
    
    # For Q&A format
    if set(['question','answer']).issubset(df.columns):
        # Remove rows where either question or answer is empty
        df = df.dropna(subset=['question', 'answer'])
        
        # Remove duplicate Q&A pairs
        df = df.drop_duplicates(subset=['question', 'answer'])
        
        processed_rows = []
        for _, row in df.iterrows():
            # Clean and format the text
            question = row['question'].strip()
            answer = row['answer'].strip()
            text = f"Q: {question}\nA: {answer}"
            
            metadata = row.to_dict()
            # remove the keys that are already part of the main text
            metadata.pop('question', None)
            metadata.pop('answer', None)
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if pd.notna(v)}
            # Ensure all metadata values are JSON serializable
            for k, v in metadata.items():
                if pd.isna(v):
                    metadata[k] = None
            
            processed_rows.append({'text': text, 'meta': metadata})
    else:
        # For non-Q&A format, concatenate all columns
        # Remove duplicate rows across all columns
        df = df.drop_duplicates()
        
        processed_rows = []
        for _, row in df.iterrows():
            # Filter out None and NaN values, then join the rest
            valid_values = [str(x).strip() for x in row.values if pd.notna(x) and str(x).strip()]
            if valid_values:  # Only add if there's actual content
                text = ' '.join(valid_values)
                processed_rows.append({'text': text, 'meta': {}})
    return processed_rows

def extract_texts_from_xlsx_bytes(b: bytes):
    df = pd.read_excel(io.BytesIO(b))
    
    # Basic data cleaning
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert all string columns to stripped strings and replace empty strings with NaN
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().replace(r'^\s*$', pd.NA, regex=True)
    
    # For Q&A format
    if set(['question','answer']).issubset(df.columns):
        # Remove rows where either question or answer is empty
        df = df.dropna(subset=['question', 'answer'])
        
        # Remove duplicate Q&A pairs
        df = df.drop_duplicates(subset=['question', 'answer'])
        
        processed_rows = []
        for _, row in df.iterrows():
            # Clean and format the text
            question = row['question'].strip()
            answer = row['answer'].strip()
            text = f"Q: {question}\nA: {answer}"
            
            metadata = row.to_dict()
            metadata.pop('question', None)
            metadata.pop('answer', None)
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if pd.notna(v)}
            # Ensure all metadata values are JSON serializable
            for k, v in metadata.items():
                if pd.isna(v):
                    metadata[k] = None
                elif isinstance(v, pd.Timestamp):  # Handle datetime values
                    metadata[k] = v.isoformat()
            
            processed_rows.append({'text': text, 'meta': metadata})
    else:
        # For non-Q&A format, concatenate all columns
        # Remove duplicate rows across all columns
        df = df.drop_duplicates()
        
        processed_rows = []
        for _, row in df.iterrows():
            # Filter out None and NaN values, then join the rest
            valid_values = [str(x).strip() for x in row.values if pd.notna(x) and str(x).strip()]
            if valid_values:  # Only add if there's actual content
                text = ' '.join(valid_values)
                processed_rows.append({'text': text, 'meta': {}})
    return processed_rows