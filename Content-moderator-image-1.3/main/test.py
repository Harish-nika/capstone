import os
import streamlit as st
from dotenv import load_dotenv
import requests
import json
import re
import time
from PIL import Image
import pytesseract
import base64
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = "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"
OLLAMA_TEXT_MODEL = "cyber-moderator-Wlm"
OLLAMA_VISION_MODEL = "cyber-vision-moderator-g3:4b"
CHUNK_SIZE = 100
CONFIDENCE_THRESHOLD = 0.7

# --- LLM & EMBEDDING MODELS ---
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- FAISS Setup for Text/Image Similarity ---
class ModerationDB:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.metadata = []

    def add(self, text, metadata):
        embedding = sentence_embedder.encode([text])
        self.index.add(np.array(embedding))
        self.metadata.append({"text": text, "meta": metadata})

    def search(self, query: str, top_k: int = 5):
        embedding = sentence_embedder.encode([query])
        D, I = self.index.search(np.array(embedding), top_k)
        return [(self.metadata[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.metadata)]

moderation_db = ModerationDB()

# --- PROMPT TEMPLATE ---
prompt = ChatPromptTemplate.from_template("""
Answer the question strictly based on the context.
<context>
{context}
</context>
Question: {input}
""")

# --- HELPER FUNCTIONS ---
def moderate_text_ollama(text):
    url = "http://localhost:11434/api/generate"
    prompt = f"""You are a cybersecurity content moderator. Analyze the following input and return valid JSON:
{{"label": <category>, "confidence": <float>, "reason": <why it's harmful>}}
Input: {text}"""
    try:
        response = requests.post(url, json={"model": OLLAMA_TEXT_MODEL, "prompt": prompt, "stream": False})
        response_text = response.json().get("response", "").strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        return json.loads(json_match.group(0)) if json_match else {}
    except Exception as e:
        return {"label": "unknown", "confidence": 0.0, "reason": f"Error: {e}"}

def moderate_image_ollama_with_ocr(img):
    try:
        ocr_text = pytesseract.image_to_string(img).strip()
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        vision_prompt = (
            "You are a cybersecurity moderation model. Analyze this image and determine if it contains any of the following: "
            "violence, threats, hate speech, self-harm indications, nudity, weapons, or illegal content. "
            "Provide the category, confidence score (0 to 1), and a short explanation."
        )

        response = requests.post("http://localhost:11434/api/generate", json={
            "model": OLLAMA_VISION_MODEL,
            "prompt": vision_prompt,
            "stream": False,
            "images": [encoded_image]
        })

        moderation_response = response.json().get("response", "⚠️ No response from vision model.")
        return ocr_text, moderation_response
    except Exception as e:
        return "", f"Error during image moderation: {str(e)}"

def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

def keyword_match(query, documents):
    matches = []
    for doc in documents:
        for line in doc.page_content.split("\n"):
            if re.search(query, line, re.IGNORECASE):
                matches.append(line)
    return matches

# --- STREAMLIT UI ---
st.set_page_config(page_title="ParentBot - Harmful Content Moderator", layout="wide")
st.title("🛡️ ParentBot - Harmful Content Moderator")

input_type = st.radio("Select Input Type:", ["📥 Copy-Paste Text", "🖼️ Upload Image", "📄 Upload WhatsApp Chat File"])

# --- COPY-PASTE TEXT ---
if input_type == "📥 Copy-Paste Text":
    user_input = st.text_area("Paste WhatsApp Message or any text")
    if st.button("Moderate Text") and user_input:
        for chunk in chunk_text(user_input):
            result = moderate_text_ollama(chunk)
            st.subheader("🧠 Moderation Result")
            st.json(result)

            if result.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                moderation_db.add(chunk, result)
                st.warning("⚠️ Harmful content detected and stored.")
                st.subheader("📌 Similar Messages")
                for r, score in moderation_db.search(chunk):
                    st.markdown(f"- {r['text']} (score: {score:.2f})")

# --- IMAGE UPLOAD ---
elif input_type == "🖼️ Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=False)
        text_from_img, result = moderate_image_ollama_with_ocr(img)

        st.subheader("🧠 Moderation Result")
        st.write(result)

        if text_from_img:
            st.subheader("📜 OCR Extracted Text")
            st.write(text_from_img)

# --- WHATSAPP CHAT FILE (.txt) ---
# --- WHATSAPP CHAT FILE (.txt) ---
elif input_type == "📄 Upload WhatsApp Chat File":
    uploaded_txt = st.file_uploader("Upload a `.txt` file", type="txt")
    if uploaded_txt:
        text = uploaded_txt.read().decode("utf-8")
        doc = Document(page_content=text, metadata={"source": uploaded_txt.name})

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents([doc])

        st.subheader("🧠 Moderation Results Per Chunk")

        for i, chunk in enumerate(chunks):
            chunk_text_data = chunk.page_content.strip()
            if not chunk_text_data:
                continue

            # Moderation prompt
            moderation_prompt = f"""
You are a cybersecurity content moderator. Analyze the following text and identify any harmful content such as hate speech, threats, nudity, violence, suicidal content, or illegal activity.

Return a JSON in the format:
{{
    "label": <category>,
    "confidence": <float 0.0-1.0>,
    "reason": <why it's harmful>
}}

Input: {chunk_text_data}
"""

            try:
                groq_response = llm.invoke(moderation_prompt)
                result = json.loads(re.search(r'\{.*\}', groq_response.content, re.DOTALL).group(0))

                st.markdown(f"### 🧩 Chunk {i+1}")
                st.code(chunk_text_data[:300] + ("..." if len(chunk_text_data) > 300 else ""), language='text')
                st.json(result)

                if result.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD:
                    moderation_db.add(chunk_text_data, result)
                    

                    st.subheader("📌 Similar Messages")
                    for r, score in moderation_db.search(chunk_text_data):
                        st.markdown(f"- {r['text']} (score: {score:.2f})")

            except Exception as e:
                st.error(f"Error processing chunk {i+1}: {e}")
# --- SIMILAR MESSAGE SEARCH (FAISS) ---
with st.expander("🔍 Search Similar Harmful Messages"):
    search_query = st.text_area("Enter a message or phrase to find similar flagged content", height=100)

    if st.button("Search Similar"):
        if search_query.strip():
            st.info("Searching for similar messages...")
            results = moderation_db.search(search_query)

            flagged_categories = {"hate speech", "threats", "nudity", "violence", "suicidal content", "illegal activity"}
            found = False

            for item, score in results:
                label = item["meta"].get("label", "").lower()
                if label in flagged_categories:
                    found = True
                    st.markdown(f"**🧷 Label:** `{label}` | **Confidence:** {item['meta'].get('confidence', 0):.2f}")
                    st.markdown(f"**📄 Text:** {item['text']}")
                    st.markdown(f"**🧠 Reason:** {item['meta'].get('reason')}")
                    st.markdown("---")

            if not found:
                st.success("✅ No similar harmful messages found in the selected categories.")
