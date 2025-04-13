import os
import faiss
import ollama
import uvicorn
import fitz  # PyMuPDF for PDFs
import numpy as np
import base64
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI(title="Cybersecurity Content Moderator")

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# FAISS Index
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
text_chunks = []
chunk_id_map = {}

# Ollama Models
TEXT_MODEL = "cyber-moderator-Wlm:7b"
VISION_MODEL = "cyber-vision-moderator-g3:4b"
# VISION_MODEL = "cyber-moderator-G3:12b"

# Create directory for extracted images
IMAGE_DIR = "extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def chunk_text(text: str) -> List[str]:
    """Splits text into paragraph-based chunks."""
    return [para.strip() for para in text.split("\n\n") if para.strip()]

def add_chunks_to_faiss(chunks: List[str]):
    """Embeds text chunks and stores them in FAISS for fast retrieval."""
    global text_chunks
    new_embeddings = []
    new_ids = []
    for i, chunk in enumerate(chunks):
        if chunk not in text_chunks:  # Avoid duplicate embeddings
            embedding = embedding_model.encode([chunk])
            new_embeddings.append(embedding)
            new_ids.append(len(text_chunks))
            text_chunks.append(chunk)
    
    if new_embeddings:
        faiss_index.add(np.array(new_embeddings, dtype=np.float32))
        for idx, chunk in zip(new_ids, chunks):
            chunk_id_map[idx] = chunk

def save_image(img_data: bytes, filename: str) -> str:
    """Saves image to the directory and returns the path."""
    path = os.path.join(IMAGE_DIR, filename)
    with open(path, "wb") as img_file:
        img_file.write(img_data)
    return path

def warm_up_model():
    """Warm up the Ollama models to avoid initialization delays."""
    try:
        ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": "ping"}])
        ollama.generate(model=VISION_MODEL, prompt="ping", images=[])
    except Exception as e:
        raise Exception(f"Error during model warm-up: {str(e)}")

# Call the warm-up function before starting the app
warm_up_model()

@app.post("/moderate-text/")
async def moderate_text(content: str = Form(...)):
    """Moderates pasted text using the text-based model."""
    try:
        chunks = chunk_text(content)
        add_chunks_to_faiss(chunks)
        
        results = []
        for chunk in chunks:
            response = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": chunk}])
            moderation_result = response.get("message", {}).get("content", "No response received.")
            
            results.append({"chunk": chunk, "moderation_result": moderation_result})
        
        return JSONResponse({"moderation_results": results})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/moderate-image/")
async def moderate_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        response = ollama.generate(
            model=VISION_MODEL,
            prompt="Analyze this image for cybersecurity threats.",
            images=[base64_image]
        )

        moderation_result = response.get("response") or response.get("message", {}).get("content", "No response received.")
        return {"moderation_result": moderation_result}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/moderate-pdf/")
async def moderate_pdf(file: UploadFile = File(...)):
    """Extracts text paragraphs and images from a PDF, then moderates them separately."""
    try:
        doc = fitz.open(stream=await file.read(), filetype="pdf")
        text_chunks = []
        image_paths = []

        # Extract text paragraphs & images
        for page_num, page in enumerate(doc):
            text_chunks.extend(chunk_text(page.get_text("text")))

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_path = save_image(img_bytes, f"page_{page_num+1}_img_{img_index+1}.png")
                image_paths.append(img_path)

        # Handle empty PDF case
        if not text_chunks and not image_paths:
            return JSONResponse({"error": "No text or images found in the PDF."}, status_code=400)

        # Process text chunks with TEXT_MODEL
        text_results = []
        for chunk in text_chunks:
            response = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": chunk}])
            text_results.append({
                "chunk": chunk,
                "moderation_result": response.get("message", {}).get("content", "No response received.")
            })

        # Process images with VISION_MODEL
        image_results = []
        for img_path in image_paths:
            with open(img_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')

            response = ollama.generate(
                model=VISION_MODEL,
                prompt="Analyze this image for cybersecurity threats.",
                images=[base64_image]
            )

            moderation_result = response.get("response") or response.get("message", {}).get("content", "No response received.")

            image_results.append({
                "image_path": img_path,
                "moderation_result": moderation_result
            })

        return JSONResponse({"text_moderation": text_results, "image_moderation": image_results})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
