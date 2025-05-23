import os
import asyncio
import httpx
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import uvicorn
from typing import List, Optional, Union
import logging
from datetime import datetime
import traceback

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize Qdrant client with retries and timeout
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30
)

# Placeholder for DocumentProcessor
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

class DocumentProcessor:
    async def process(self, content: bytes, filename: str):
        tmp_path = None
        try:
            valid_extensions = {'.pdf', '.docx', '.txt', '.xlsx', '.xls'}
            if not any(filename.lower().endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Unsupported file format for {filename}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            converter = DocumentConverter()
            result = converter.convert(tmp_path, max_file_size=1024 * 512)

            markdown = result.document.export_to_markdown()
            doc = result.document
            chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5")
            chunk_iter = chunker.chunk(doc)
            chunks = [chunk.text.strip() for chunk in chunk_iter if hasattr(chunk, "text") and chunk.text and chunk.text.strip()]

            logger.info(f"Extracted {len(chunks)} chunks from {filename}")
            return type("Doc", (), {"text": chunks})()
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Docling processing failed: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {str(e)}")

processor = DocumentProcessor()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is required")

cohere_headers = {
    "Authorization": f"Bearer {COHERE_API_KEY}",
    "Content-Type": "application/json",
    "Cohere-Version": "2022-12-06"
}

documents = []

async def process_document(file: UploadFile) -> Optional[dict]:
    try:
        content = await file.read()
        doc = await processor.process(content, file.filename)
        return {
            "id": len(documents) + 1,
            "text": doc.text,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}\n{traceback.format_exc()}")
        return None
    finally:
        await file.close()

MAX_CHAR_LIMIT = 2000

def flatten_texts(texts: List[Union[str, List[str]]]) -> List[str]:
    flat = []
    for text in texts:
        if isinstance(text, list):
            flat.extend(text)
        elif isinstance(text, str):
            flat.append(text)
    return flat

def sanitize_texts(texts: List[Union[str, List[str]]]) -> List[str]:
    flat_texts = flatten_texts(texts)
    return [
        text.strip()
        for text in flat_texts
        if isinstance(text, str) and text.strip() and len(text.strip()) <= MAX_CHAR_LIMIT
    ]

async def get_embeddings(texts: List[str], batch_size: int = 5) -> List[List[float]]:
    sanitized_texts = sanitize_texts(texts)
    if not sanitized_texts:
        raise HTTPException(status_code=400, detail="No valid chunks to embed")

    embeddings = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(sanitized_texts), batch_size):
            batch = sanitized_texts[i:i + batch_size]
            try:
                response = await client.post(
                    "https://api.cohere.ai/v1/embed",
                    headers=cohere_headers,
                    json={
                        "texts": batch,
                        "model": "embed-english-v3.0",
                        "input_type": "search_document"
                    }
                )
                response.raise_for_status()
                batch_embeddings = response.json().get("embeddings")
                if not batch_embeddings:
                    raise ValueError("Empty embedding response")
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding error: {str(e)}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))
    return embeddings

async def index_documents(new_documents: List[dict]):
    global documents
    collection_name = "rag_collection"
    vector_size = 1024

    try:
        try:
            qdrant.get_collection(collection_name)
        except Exception:
            logger.info(f"Creating new collection: {collection_name}")
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )

        texts = [doc["text"] for doc in new_documents]
        embeddings = await get_embeddings(texts)

        points = [
            models.PointStruct(
                id=doc["id"],
                vector=embeddings[i],
                payload={
                    "text": doc["text"],
                    "source": doc["filename"],
                    "indexed_at": datetime.now().isoformat()
                }
            ) for i, doc in enumerate(new_documents)
        ]

        qdrant.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )

        documents.extend(new_documents)
        logger.info(f"Successfully indexed {len(new_documents)} documents")
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

async def retrieve_documents(query: str, limit: int = 3) -> List[str]:
    try:
        query_embedding = (await get_embeddings([query]))[0]
        results = qdrant.search(
            collection_name="rag_collection",
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        return [hit.payload["text"] for hit in results]
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}\n{traceback.format_exc()}")
        return []

async def generate_response(query: str, context: List[str]) -> str:
    flat_context = flatten_texts(context)
    combined_context = "\n\n".join(flat_context)

    if not combined_context:
        return "No relevant documents found to answer the query."

    prompt = f"""You are a helpful assistant. The context below is in markdown format. Use it to provide a concise and accurate answer to the query, ignoring markdown syntax like headings or lists unless relevant to the content.

Context:
{combined_context}

Query: {query}

Answer:"""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.cohere.ai/v1/generate",
                headers=cohere_headers,
                json={
                    "prompt": prompt,
                    "max_tokens": 200,
                    "temperature": 0.7,
                    "k": 0,
                    "p": 0.9,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
            )
            response.raise_for_status()
            return response.json()["generations"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation error: {str(e)}\n{traceback.format_exc()}")
            return "Failed to generate response."

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        tasks = [process_document(file) for file in files]
        processed_docs = await asyncio.gather(*tasks)
        new_docs = [doc for doc in processed_docs if doc]

        if not new_docs:
            raise HTTPException(status_code=400, detail="No valid documents processed")

        await index_documents(new_docs)
        return {"message": f"Successfully uploaded and indexed {len(new_docs)} documents."}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Upload and indexing failed.")

@app.post("/query")
async def query_documents(request: dict):
    try:
        query = request.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query text is required")

        retrieved = await retrieve_documents(query)
        answer = await generate_response(query, retrieved)
        return {"query": query, "answer": answer, "sources": retrieved}
    except Exception as e:
        logger.error(f"Query handling failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Query processing failed")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3000)),
        log_level="info",  # Changed from debug to info for production
        timeout_keep_alive=30  # Increased keep-alive timeout
    )