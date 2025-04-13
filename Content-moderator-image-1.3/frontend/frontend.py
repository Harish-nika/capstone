import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Cybersecurity Content Moderator", layout="wide")
st.title("🔍 Cybersecurity Content Moderator")

# Sidebar for input selection
st.sidebar.header("Select Input Type")
input_type = st.sidebar.radio("Choose an option:", ["Paste Text", "Upload Image"])

def fetch_moderation_response(url, data=None, files=None):
    try:
        if data:
            response = requests.post(url, data=data)
        elif files:
            response = requests.post(url, files=files)
        else:
            return None
        
        # Check if response is valid
        if response.status_code == 200:
            return response.json()  # Try to decode JSON response
        else:
            st.error(f"Error: Received status code {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None
    except ValueError:
        st.error("Error: Invalid response format (not JSON).")
        return None

if input_type == "Paste Text":
    st.subheader("📝 Enter Text for Moderation")
    text_input = st.text_area("Paste text here:", height=200)
    
    if st.button("Moderate Text"):
        if text_input.strip():
            response_data = fetch_moderation_response(f"{API_URL}/moderate-text/", data={"content": text_input})
            
            if response_data:
                results = response_data.get("moderation_results", [])
                if results:
                    st.write("### 🚀 Moderation Results:")
                    for res in results:
                        st.markdown(f"**Chunk:** {res['chunk']}")
                        st.code(res["moderation_result"], language="json")
                else:
                    st.warning("⚠️ No moderation results returned.")
            else:
                st.error("Failed to get moderation response.")
        else:
            st.warning("⚠️ Please enter text for moderation.")

# elif input_type == "Upload PDF":
#     st.subheader("📄 Upload PDF for Moderation")
#     uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
    
#     if uploaded_pdf and st.button("Moderate PDF"):
#         files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
#         response_data = fetch_moderation_response(f"{API_URL}/moderate-pdf/", files=files)

#         if response_data:
#             results = response_data
#             # Process text moderation results
#             if "text_moderation" in results and results["text_moderation"]:
#                 st.write("### 📝 Text Moderation Results:")
#                 for res in results["text_moderation"]:
#                     st.markdown(f"**Chunk:** {res['chunk']}")
#                     st.code(res["moderation_result"], language="json")
#             else:
#                 st.warning("⚠️ No text detected in the PDF.")

#             # Process image moderation results
#             if "image_moderation" in results and results["image_moderation"]:
#                 st.write("### 🖼️ Image Moderation Results:")
#                 for res in results["image_moderation"]:
#                     st.code(res["moderation_result"], language="json")

#                 st.write("### 📷 Extracted Images from PDF:")
#                 for res in results["image_moderation"]:
#                     if "image_path" in res:
#                         image_response = requests.get(f"{API_URL}/{res['image_path']}")  # Fetch image
#                         if image_response.status_code == 200:
#                             img = Image.open(io.BytesIO(image_response.content))
#                             st.image(img, caption=res["image_path"], use_column_width=False)
#                         else:
#                             st.warning(f"⚠️ Could not fetch image: {res['image_path']}")
#             else:
#                 st.warning("⚠️ No images found in the PDF.")
#         else:
#             st.error("Failed to get PDF moderation response.")

elif input_type == "Upload Image":
    st.subheader("📷 Upload Image for Moderation")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image and st.button("Moderate Image"):
        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), "image/png")}
        response_data = fetch_moderation_response(f"{API_URL}/moderate-image/", files=files)

        if response_data:
            result = response_data
            if "moderation_result" in result:
                st.write("### 🖼️ Image Moderation Result:")
                st.code(result["moderation_result"], language="json")
            else:
                st.error("⚠️ Missing 'moderation_result' in API response.")
            
            # Open image using PIL
            image = Image.open(uploaded_image)

            # Resize image while maintaining aspect ratio
            max_width = 400  # Adjust as needed
            original_width, original_height = image.size
            aspect_ratio = original_height / original_width
            new_height = int(max_width * aspect_ratio)  # Maintain aspect ratio

            image = image.resize((max_width, new_height))  # Resize while preserving aspect ratio

            # Display resized image
            st.image(image, caption="Uploaded Image", use_column_width=False)
        else:
            st.error("Failed to get image moderation response.")
