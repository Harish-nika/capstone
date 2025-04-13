import os
import streamlit as st
from dotenv import load_dotenv
import requests
import json
import re
import time
import PyPDF2
import faiss
import joblib
import string
import pickle
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
from keras.models import load_model
from scipy.sparse import hstack
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = "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"
OLLAMA_TEXT_MODEL = "cyber-moderator-Wlm"
OLLAMA_VISION_MODEL = "cyber-vision-moderator-g3:4b"
CHUNK_SIZE = 60
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
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
    return text

def detect_harmful_mldl(test_comments):
    ensemble_model = joblib.load("/home/harish/toxicity_model/Toxicity-Classification-On-Social-Media-main/ensemblepackage/ensemble_model.pkl")
    tf_idf_vectorizer = joblib.load("/home/harish/toxicity_model/Toxicity-Classification-On-Social-Media-main/ensemblepackage/tf_idf_vectorizer.pkl")
    lstm_model = load_model("/home/harish/toxicity_model/Toxicity-Classification-On-Social-Media-main/lstmpackage/lstm_model.h5")
    with open("/home/harish/toxicity_model/Toxicity-Classification-On-Social-Media-main/lstmpackage/tokenizer.pickle", 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    test_tf_idf = tf_idf_vectorizer.transform(test_comments)
    ensemble_pred_proba = ensemble_model.predict_proba(test_tf_idf)
    
    max_length = 200
    lstm_tokenized = tokenizer.texts_to_sequences(test_comments)
    lstm_padded = pad_sequences(lstm_tokenized, maxlen=max_length)
    lstm_predictions = lstm_model.predict(lstm_padded)
    
    logistic_save_dir = "/home/harish/toxicity_model/Toxicity-Classification-On-Social-Media-main/logisticregressionpackage"
    word_vectorizer = joblib.load(os.path.join(logistic_save_dir, "tf_idf_word_vectorizer.pkl"))
    char_vectorizer = joblib.load(os.path.join(logistic_save_dir, "tf_idf_char_vectorizer.pkl"))
    
    word_features = word_vectorizer.transform(test_comments)
    char_features = char_vectorizer.transform(test_comments)
    logistic_features = hstack([char_features, word_features])
    
    logistic_predictions = {}
    logistic_probabilities = {}
    class_names = ['Hate Speech', 'Suicidal Content', 'Unparliamentary Language', 'Threats', 'Terrorism-Related Content', 'Explicit & NSFW Content','Illegal Content']
    
    for label in class_names:
        model_path = os.path.join(logistic_save_dir, f'logistic_model_{label}.pkl')
        model = joblib.load(model_path)
        logistic_predictions[label] = model.predict(logistic_features)
        logistic_probabilities[label] = model.predict_proba(logistic_features)[:, 1]
    
    results = []
    def majority_voting(logistic, lstm, ensemble):
        vote_count = logistic + lstm + ensemble
        return "Harmful Content" if vote_count >= 2 else "Not Harmful"
    
    for idx, comment in enumerate(test_comments):
        ensemble_votes = sum(ensemble_pred_proba[idx] > 0.5)
        logistic_votes = sum([logistic_predictions[label][idx] > 0.5 for label in class_names])
        lstm_votes = sum(lstm_predictions[idx] > 0.5)
        final_decision = majority_voting(logistic_votes, lstm_votes, ensemble_votes)
        results.append((comment, final_decision))
    
    return results

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

        # Recommended overlap to avoid redundant duplicate chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents([doc])

        st.subheader("🧠 Moderation Results Per Chunk")

        for i, chunk in enumerate(chunks):
            chunk_text_data = chunk.page_content.strip()
            if not chunk_text_data:
                continue

            moderation_prompt = f"""
Input: {chunk_text_data}

You are a Cybersecurity Content Moderation AI.

Your job is to classify user-generated text (from platforms like Discord, Reddit, etc.) for potential harmful or inappropriate content.

You MUST only reply in valid JSON format — no greetings, explanations, or markdown. Output nothing except the JSON object.

---

You can classify content into **one or more** of these 13 categories:

1. **Hate Speech** – Offensive, derogatory, or discriminatory language targeting race, religion, gender, ethnicity, disability, or other protected traits.

2. **Unparliamentary Language** – Profanity, offensive slurs, or disrespectful speech violating acceptable decorum. Do NOT flag polite or casual everyday conversations.

3. **Threats** – Statements implying harm, violence, doxxing, or any form of intimidation.

4. **Suicidal Content** – Mentions of self-harm, suicidal ideation, or encouragement of self-harm.

5. **Terrorism-Related Content** – Support, promotion, planning, or justification of terrorist acts or extremist ideologies.

6. **Illegal Content** – Discussions of unlawful activities such as hacking, drug trafficking, or other crimes.

7. **Harassment** – Cyberbullying, repeated targeting, intimidation, or abusive behavior towards individuals or groups.

8. **Self-Harm Encouragement** – Any content that promotes, glorifies, or normalizes self-harm or suicidal behavior.

9. **Sexual Exploitation & Child Safety Violations** – Content that depicts, promotes, or facilitates child exploitation, non-consensual sexual acts, or abuse.

10. **Explicit & NSFW Content** – Pornographic, sexual, or highly explicit material unsuitable for general audiences.

11. **Misinformation** – False or misleading info presented as fact.

12. **Scam / Fraud** – Deceptive schemes meant to defraud or trick users.

13. **Cybersecurity Threats** – Content that spreads malware, phishing links, or attempts unauthorized system access.

---

**Moderation Rules:**
- DO NOT flag casual, polite, or friendly text unless it clearly contains harmful elements.
- Only classify content with confidence **>= 0.7**.
- Ignore borderline or ambiguous statements.

---

Your Output Format **must be** like this:

{{
  "classification": {{
    "category_name": {{
      "confidence_score": float (0.0 to 1.0),
      "justification": "Why this category applies to input text"
    }},
    
  }},
  "max_confidence_category": "most_confident_category_name_or_null",
  "final_verdict": "Harmful Content" or "Not Harmful Content",
  "safe_content": true or false
}}

If the input is safe and contains no issues, return:

{{
  "classification": {{}},
  "max_confidence_category": null,
  "final_verdict": "Not Harmful Content",
  "safe_content": true
}}

If the input is invalid (not quoted text or empty), return:

{{
  "error": "Invalid format. Provide the input as a quoted string."
}}
"""

            try:
                # Run the moderation prompt through LLM
                groq_response = llm.invoke(moderation_prompt)

                match = re.search(r'\{.*\}', groq_response.content, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in model response.")
                result = json.loads(match.group(0))

                st.markdown(f"### 🧩 Chunk {i+1}")
                st.code(chunk_text_data[:300] + ("..." if len(chunk_text_data) > 300 else ""), language='text')
                st.json(result)

                # Get max confidence
                confidences = [
                    details["confidence_score"]
                    for details in result.get("classification", {}).values()
                    if isinstance(details, dict)
                ]
                max_conf = max(confidences) if confidences else 0.0

                if max_conf >= CONFIDENCE_THRESHOLD:
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
