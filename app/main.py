from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import File as FileModel, Chunk as ChunkModel
from app.schemas import IngestResponse, QueryRequest, QueryResponse, SourceItem
from app.utils import extract_text_from_pdf_bytes, extract_texts_from_csv_bytes, extract_texts_from_xlsx_bytes, chunk_text
from app.embeddings import embed_texts, VECTOR_DIM
from app.engine_faiss import faiss_idx, FaissIndex
import os
import tempfile
import traceback
import numpy as np
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import time

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)

# Create a session with retry strategy
http_session = requests.Session()
http_session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
http_session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title='RAG Service')

THRESHOLD = float(os.getenv('THRESHOLD', '0.60'))
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '120'))  # 2 minutes timeout
OLLAMA_MAX_TOKENS = int(os.getenv('OLLAMA_MAX_TOKENS', '5120'))  # Limit response length

@app.post('/ingest', response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    global faiss_idx
    db: Session = SessionLocal()
    try:
        content = await file.read()
        filename = file.filename
        ftype = (filename.split('.')[-1] or '').lower()

        logging.info(f"Starting ingest for file: '{filename}' (type: {ftype})")

        file_row = FileModel(filename=filename, file_type=ftype)
        db.add(file_row)
        db.commit()
        db.refresh(file_row)
        logging.info(f"Created File record in DB with ID: {file_row.id}")

        chunks_to_add = []
        logging.info("Extracting and chunking text...")
        if ftype in ['pdf']:
            pages = extract_text_from_pdf_bytes(content)
            logging.info(f"Extracted {len(pages)} pages from PDF.")
            for pidx, page_text in enumerate(pages):
                if not page_text or page_text.strip()=='' :
                    continue
                chunks = chunk_text(page_text)
                for cidx, ctext in enumerate(chunks):
                    chunks_to_add.append(ChunkModel(file_id=file_row.id, text=ctext, page=pidx+1, offset=cidx))
        elif ftype in ['csv']:
            processed_rows = extract_texts_from_csv_bytes(content)
            logging.info(f"Extracted {len(processed_rows)} rows from CSV. Treating each Q&A row as a single chunk.")
            for row_data in processed_rows:
                # For structured Q&A data, we treat each row as a single, atomic chunk.
                # Do NOT split it further.
                chunks_to_add.append(ChunkModel(file_id=file_row.id, text=row_data['text'], meta=row_data['meta']))
        elif ftype in ['xls','xlsx']:
            processed_rows = extract_texts_from_xlsx_bytes(content)
            logging.info(f"Extracted {len(processed_rows)} rows from Excel file. Treating each Q&A row as a single chunk.")
            for row_data in processed_rows:
                # For structured Q&A data, we treat each row as a single, atomic chunk.
                chunks_to_add.append(ChunkModel(file_id=file_row.id, text=row_data['text'], meta=row_data['meta']))
        else:
            # try treat as text
            logging.info("Treating as plain text file.")
            s = content.decode('utf-8', errors='ignore')
            chunks = chunk_text(s)
            for cidx, ctext in enumerate(chunks):
                chunks_to_add.append(ChunkModel(file_id=file_row.id, text=ctext))

        if chunks_to_add:
            db.add_all(chunks_to_add)
            db.commit()
        
        logging.info("Finished saving chunks to the database.")

        # fetch all chunks for this file and create embeddings
        chunks = db.query(ChunkModel).filter(ChunkModel.file_id==file_row.id).all()
        texts = [c.text for c in chunks]
        logging.info(f"Fetched {len(texts)} chunks from DB for embedding.")

        if len(texts) > 0:
            vecs = embed_texts(texts)
            logging.info(f"Created {len(vecs)} embeddings.")
            # ensure vecs is numpy array and show shape for debugging
            try:
                vecs_arr = np.array(vecs, dtype='float32')
                if vecs_arr.ndim == 1:
                    vecs_arr = vecs_arr.reshape(1, -1)
                
                logging.info(f"Adding {vecs_arr.shape[0]} vectors of dimension {vecs_arr.shape[1]} to FAISS index.")
                # try to add; if dimension mismatch occurs, recreate index
                try:
                    faiss_idx.add(vecs_arr, [c.id for c in chunks])
                except ValueError as e:
                    # expected message from engine_faiss when dim mismatch
                    if 'does not match index dim' in str(e):
                        print('FAISS dim mismatch detected:', e)
                        # recreate index with correct dim and replace module-level instance
                        new_idx = FaissIndex(dim=vecs_arr.shape[1], path=faiss_idx.path)
                        import app.engine_faiss as engine_faiss
                        engine_faiss.faiss_idx = new_idx
                        faiss_idx = new_idx
                        faiss_idx.add(vecs_arr, [c.id for c in chunks])
                    else:
                        raise
                logging.info("Successfully added vectors to FAISS index.")
            except Exception:
                # re-raise to be handled by outer try/except and return 500
                logging.error("An error occurred during embedding or FAISS indexing.")
                raise
        
        logging.info(f"Ingest for file '{filename}' completed successfully. Total chunks added: {len(texts)}.")
        return IngestResponse(file_id=file_row.id, filename=filename, chunks_added=len(texts))
    except Exception as e:
        logging.error(f"An error occurred during ingest: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post('/query', response_model=QueryResponse)
def query(req: QueryRequest):
    db = SessionLocal()
    try:
        q = req.question
        top_k = req.top_k
        logging.info(f"Received query: '{q}', top_k: {top_k}")

        qvec = embed_texts([q])
        logging.info(f"Query embedded into vector of shape {qvec.shape}")

        # faiss_idx.search returns 1D arrays for a single query vector
        scores, ids = faiss_idx.search(qvec, top_k)
        logging.info(f"FAISS search returned scores: {scores}")
        logging.info(f"FAISS search returned ids: {ids}")
        sources = []

        # Process the 1D arrays of scores and ids directly
        for i in range(len(ids)):
            iid = int(ids[i])
            score = float(scores[i])

            # -1 is a common sentinel for no result
            if iid == -1:
                continue

            chunk_id = faiss_idx.id_map.get(iid)
            if not chunk_id:
                logging.warning(f"FAISS ID {iid} not found in id_map.")
                continue

            ch = db.query(ChunkModel).filter(ChunkModel.id == chunk_id).first()
            if not ch:
                logging.warning(f"Chunk ID {chunk_id} (from FAISS ID {iid}) not found in DB.")
                continue

            sources.append({
                'chunk_id': ch.id,
                'file_id': ch.file_id,
                'score': score,
                'text': ch.text[:1000]
            })
        
        logging.info(f"Found {len(sources)} relevant sources from DB.")
        logging.info(f"Found: {sources} ")

        # determine top score
        top_score = max([s['score'] for s in sources]) if sources else 0.0
        logging.info(f"Top score from sources: {top_score}, Threshold: {THRESHOLD}")
        if top_score < THRESHOLD:
            logging.info("Top score is below threshold. Returning no-answer response.")
            no_answer_responses = {
                "vi": "Xin lỗi, thông tin này không có trong tài liệu nội bộ của U-plus",
                "en": "Sorry, this information is not available in U-plus's internal documents",
                "ko": "죄송합니다. 이 정보는 U-plus의 내부 문서에서 찾을 수 없습니다"
            }
            return QueryResponse(answer=no_answer_responses.get(req.language, no_answer_responses["vi"]), sources=[])

        # build prompt for Ollama if available
        logging.info("Building prompt for LLM.")
        prompt_context = ''
        used_file_ids = set()
        for s in sources[:top_k]:
            prompt_context += f"FILE_ID: {s['file_id']} | CHUNK_ID: {s['chunk_id']}\n{s['text']}\n---\n"
            used_file_ids.add(s['file_id'])

        # Define responses for each language
        no_answer_responses = {
            "vi": "Xin lỗi, thông tin kỹ thuật hoặc dịch vụ này không có trong tài liệu nội bộ của U-plus. Vui lòng liên hệ với bộ phận hỗ trợ kỹ thuật của chúng tôi để được tư vấn chi tiết.",
            "en": "I apologize, but this technical information or service detail is not available in U-plus's internal documentation. Please contact our technical support team for detailed assistance.",
            "ko": "죄송합니다. 이 기술 정보 또는 서비스 세부 사항은 U-plus의 내부 문서에서 찾을 수 없습니다. 자세한 지원은 기술 지원팀에 문의해 주시기 바랍니다."
        }

        language_instructions = {
            "vi": "Trả lời bằng tiếng Việt. Hãy trả lời với tư cách chuyên gia viễn thông, sử dụng thuật ngữ chuyên ngành một cách chính xác, giải thích rõ ràng và chuyên nghiệp.",
            "en": "Answer in English as a telecommunications expert. Use industry terminology accurately, explain technical concepts clearly and professionally.",
            "ko": "통신 전문가로서 한국어로 답변하십시오. 산업 용어를 정확하게 사용하고 기술적 개념을 명확하고 전문적으로 설명하십시오."
        }

        prompt = f"""You are U-plus's specialized AI assistant for Telecommunications (Telecom). You are an expert in telecommunications technology, services, and industry standards. You must follow these rules strictly:

1. ROLE AND EXPERTISE:
   - You are a Telecom domain expert representing U-plus
   - Use telecommunications industry terminology accurately
   - Explain technical concepts clearly and professionally
   - Be knowledgeable about network infrastructure, mobile technologies, and telecom services

2. UNDERSTAND THE CONTEXT: 
   - The provided DOCUMENTS are in English or Korean
   - Content relates to U-plus's telecom services, infrastructure, and solutions
   - Technical specifications and industry standards may be referenced

3. ANSWER LANGUAGE: 
   - You must answer in {language_instructions.get(req.language, language_instructions["vi"])}
   - Maintain technical accuracy when translating telecom terminology
   - Use standardized industry terms in the target language

4. SOURCE USAGE:
   - ONLY use information from the provided DOCUMENTS below
   - If information cannot be found, respond: "{no_answer_responses.get(req.language, no_answer_responses["vi"])}"
   - Always cite sources using [FILE_ID] format at the end of the answer
   - When citing technical specifications, include relevant standards or protocols

5. ANSWER FORMAT:
   - Be professional, precise, and solution-oriented
   - For technical explanations:
     * Use structured formats with clear headings
     * Break down complex concepts into understandable points
     * Include relevant technical parameters when applicable
   - For service-related questions:
     * Highlight key features and benefits
     * Provide clear, actionable information
     * Reference specific service capabilities
   - Use appropriate technical diagrams or bullet points when helpful

6. TECHNICAL ACCURACY:
   - Maintain precision in technical details
   - Use correct telecom industry terminology
   - Verify specifications and parameters before including
   - Be explicit about technical limitations or requirements

7. SECURITY AND COMPLIANCE:
   - Maintain confidentiality of internal network infrastructure
   - Do not disclose sensitive technical specifications
   - Follow telecom industry compliance standards
   - For questions about competitors or sensitive topics, politely decline

8. TRANSLATION:
   - Accurately translate technical telecom terms
   - Maintain industry-standard terminology
   - Ensure technical meaning is preserved
   - Use local telecom market terminology when appropriate

DOCUMENTS:
{prompt_context}

QUESTION: {q}

{language_instructions.get(req.language, language_instructions["vi"])}:"""

        answer_text = None
        if OLLAMA_URL:
            try:
                logging.info(f"Sending request to LLM at {OLLAMA_URL}...")
                start_time = time.time()
                logging.info("Sending prompt to LLM (length: %d characters)", len(prompt))
                
                # Configure request parameters
                payload = {
                    "model": "gemma3:4b",
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": OLLAMA_MAX_TOKENS,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
                
                # Use session with retry strategy
                resp = http_session.post(
                    OLLAMA_URL,
                    json=payload,
                    timeout=OLLAMA_TIMEOUT
                )
                
                if resp.status_code == 200:
                    process_time = time.time() - start_time
                    answer_text = resp.json().get('response') or resp.text
                    logging.info(f"LLM response received in {process_time:.2f} seconds")
                    
                    # Log response stats
                    response_length = len(answer_text)
                    logging.info(f"Response length: {response_length} characters")
                    if process_time > OLLAMA_TIMEOUT / 2:
                        logging.warning(f"LLM response time ({process_time:.2f}s) is approaching timeout ({OLLAMA_TIMEOUT}s)")
                else:
                    logging.error(f"LLM request failed with status {resp.status_code}: {resp.text}")
                    answer_text = None
                    
            except requests.exceptions.Timeout:
                logging.error(f"LLM request timed out after {OLLAMA_TIMEOUT} seconds")
                answer_text = None
            except requests.exceptions.ConnectionError:
                logging.error("Failed to connect to LLM server. Check if Ollama is running.")
                answer_text = None
            except Exception as e:
                logging.error(f"Unexpected error calling LLM: {str(e)}", exc_info=True)
                answer_text = None
        if not answer_text:
            logging.info("Using fallback response as LLM is not available or failed.")
            # fallback: concisely return concatenated context and a short answer note
            answer_text = 'Dưới đây là thông tin tìm được: \n' + '\n'.join([s['text'][:500] for s in sources[:top_k]]) + "\n\n(Chú ý: Thiết lập OLLAMA_URL để tự động gọi model và tạo câu trả lời.)"

        src_items = [SourceItem(chunk_id=s['chunk_id'], file_id=s['file_id'], score=s['score'], text=s['text']) for s in sources[:top_k]]
        logging.info(f"Query for '{q}' completed successfully.")
        return QueryResponse(answer=answer_text, sources=src_items)
    except Exception as e:
        logging.error(f"An error occurred during query: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get('/status')
def status():
    return JSONResponse({'status': 'ok', 'faiss_index_path': faiss_idx.path, 'ndocs': len(faiss_idx.id_map)})