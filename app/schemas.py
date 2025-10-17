from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    file_id: str
    filename: str
    chunks_added: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    language: str = "vi"  # Default is Vietnamese, supported: "vi" (Vietnamese), "en" (English), "ko" (Korean)

class SourceItem(BaseModel):
    chunk_id: str
    file_id: str
    score: float
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]