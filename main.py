import os
import asyncio
import httpx
import tempfile
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import uvicorn
from typing import List, Optional
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
    timeout=30  # Increased timeout
)

# Placeholder for DocumentProcessor (assuming docling supports Excel)
from docling.document_converter import DocumentConverter

class DocumentProcessor:
    async def process(self, content: bytes, filename: str):
        tmp_path = None
        try:
            # Validate file extension
            valid_extensions = {'.pdf', '.docx', '.txt', '.xlsx', '.xls'}
            if not any(filename.lower().endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Unsupported file format for {filename}")

            # Save the uploaded content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Use Docling to convert the document
            converter = DocumentConverter()
            result = converter.convert(tmp_path, max_file_size= 1024 * 512) 

            # Export to markdown
            markdown = result.document.export_to_markdown()
            logger.info(f"Processed {filename} to markdown")
            return type("Doc", (), {"text": markdown})()
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Docling processing failed: {str(e)}")
        finally:
            # Ensure temp file is removed
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {str(e)}")

processor = DocumentProcessor()

# Cohere API configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is required")

cohere_headers = {
    "Authorization": f"Bearer {COHERE_API_KEY}",
    "Content-Type": "application/json",
    "Cohere-Version": "2022-12-06"
}

# Global variable to store processed documents
documents = []

async def process_document(file: UploadFile) -> Optional[dict]:
    """Process a single uploaded file using DocumentProcessor"""
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

async def get_embeddings(texts: List[str], batch_size: int = 5) -> List[List[float]]:  # Reduced batch size
    """Get embeddings from Cohere API"""
    embeddings = []
    async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
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
                embeddings.extend(response.json()["embeddings"])
            except httpx.HTTPStatusError as e:
                logger.error(f"Cohere API error: {str(e)}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Cohere API error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during embedding: {str(e)}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    return embeddings

async def index_documents(new_documents: List[dict]):
    """Index documents in Qdrant"""
    global documents
    collection_name = "rag_collection"
    vector_size = 1024

    try:
        try:
            qdrant.get_collection(collection_name)
        except Exception as e:
            logger.info(f"Creating new collection: {collection_name}")
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
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
            )
            for i, doc in enumerate(new_documents)
        ]
        
        operation_info = qdrant.upsert(
            collection_name=collection_name,
            points=points,
            wait=True  # Wait for operation to complete
        )
        logger.info(f"Upsert operation info: {operation_info}")
        
        documents.extend(new_documents)
        logger.info(f"Successfully indexed {len(new_documents)} documents")
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

async def retrieve_documents(query: str, limit: int = 3) -> List[str]:
    """Retrieve relevant documents from Qdrant"""
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
        logger.error(f"Error retrieving documents: {str(e)}\n{traceback.format_exc()}")
        return []

async def generate_response(query: str, context: List[str]) -> str:
    """Generate response using Cohere's generate API"""
    if not context:
        return "No relevant documents found to answer the query."
    
    prompt = f"""You are a helpful assistant. The context below is in markdown format. Use it to provide a concise and accurate answer to the query, ignoring markdown syntax like headings or lists unless relevant to the content.

Context:
{"\n\n".join(context)}

Query: {query}

Answer:"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
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
        except httpx.HTTPStatusError as e:
            logger.error(f"Cohere generate error: {str(e)}\n{traceback.format_exc()}")
            return "Failed to generate response due to API error."
        except Exception as e:
            logger.error(f"Unexpected generate error: {str(e)}\n{traceback.format_exc()}")
            return "Failed to generate response due to unexpected error."

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads and index them in Qdrant"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        tasks = [process_document(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        new_documents = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Document processing error: {str(result)}")
            elif result is not None:
                new_documents.append(result)
        
        if not new_documents:
            raise HTTPException(status_code=400, detail="No valid documents processed")
        
        await index_documents(new_documents)
        
        return JSONResponse({
            "message": f"Successfully processed and indexed {len(new_documents)} documents",
            "filenames": [doc["filename"] for doc in new_documents]
        })
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/query")
async def handle_query(request: Request):
    """Handle user queries"""
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        context = await retrieve_documents(query)
        response = await generate_response(query, context)
        
        if not response or "failed" in response.lower():
            raise HTTPException(status_code=503, detail="Service temporarily unavailable. Please try again.")
            
        return JSONResponse({
            "query": query,
            "response": response,
            "context": context
        })
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Qdrant connection
        qdrant.get_collections()
        
        # Check Cohere connection
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.cohere.ai/v1/tokenize",
                headers=cohere_headers,
                params={"text": "test"}
            )
            response.raise_for_status()
            
        return {"status": "healthy", "qdrant": "connected", "cohere": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    try:
        required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "COHERE_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        # Test connections
        await health_check()
        
        logger.info("Application initialization complete")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3000)),
        log_level="info",  # Changed from debug to info for production
        timeout_keep_alive=30  # Increased keep-alive timeout
    )