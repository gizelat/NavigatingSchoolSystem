import streamlit as st
import pandas as pd
import os
import base64
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from gtts import gTTS

# Load environment variables
load_dotenv(override=True)

st.set_page_config(
    page_title="Juvenile Justice & Disability Glossary",
    layout="wide",
    page_icon="🧩"
)

# ------------------ Load Data ------------------ #
@st.cache_data

def load_data():
    df = pd.read_excel("Juvenile Justice, Academic IEP, School System.xlsx")
    df.columns = df.columns.str.strip()
    df.fillna("", inplace=True)
    return df

data = load_data()

# ------------------ Gemini API ------------------ #
def ask_gemini(question: str, context: str, api_key: str) -> str:
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Answer the following question using ONLY the provided glossary context.\n\nGlossary:\n{context}\n\nQuestion:\n{question}"}
                    ]
                }
            ]
        }
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        response = requests.post(url, headers=headers, data=json.dumps(body))
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Error: Gemini API failed."
    except Exception as e:
        return f"Exception: {str(e)}"

# ------------------ Text-to-Speech ------------------ #
def generate_audio(text: str, lang: str = "en") -> str:
    try:
        tts = gTTS(text, lang=lang)
        path = f"temp_audio_{hash(text)}_{lang}.mp3"
        tts.save(path)
        return path
    except:
        return ""

# ------------------ Top Banner ------------------ #
col1, col2 = st.columns([1, 5])
with col1:
    logo_path = "Screenshot 2025-09-10 at 1.53.16 PM.png"
    if Path(logo_path).exists():
        st.image(logo_path, width=100)
with col2:
    st.markdown("""
    # Access 2 Community & Friendship
    ### Juvenile Justice & Disability Glossary
    """)

# ------------------ Search and Letter Filter (Top Horizontal) ------------------ #
st.markdown("### Navigation & Search")
col_search, col_letter = st.columns([2, 5])

with col_search:
    search_term = st.text_input("Search for a term")

with col_letter:
    selected_letter = st.radio("Jump to Letter:", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), horizontal=True, index=0)

# ------------------ Filtering ------------------ #
filtered_data = data[data['term_en'].str.startswith(selected_letter, na=False)]
if search_term:
    filtered_data = data[data['term_en'].str.contains(search_term, case=False, na=False)]

if filtered_data.empty:
    st.warning("No terms found.")

# ------------------ Layout Titles ------------------ #
col_eng, col_esp = st.columns(2)
with col_eng:
    st.markdown("<h3 style='text-align: center; color: red;'>ENGLISH</h3>", unsafe_allow_html=True)
with col_esp:
    st.markdown("<h3 style='text-align: center; color: red;'>ESPAÑOL</h3>", unsafe_allow_html=True)

# ------------------ Term-by-Term Paired Layout ------------------ #
for _, row in filtered_data.iterrows():
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("---")
        st.subheader(row['term_en'])

        audio_file = generate_audio(row['term_en'], lang="en")
        if os.path.exists(audio_file):
            with open(audio_file, 'rb') as f:
                st.markdown("🔊 Click the speaker to hear pronunciation:")
                st.audio(f.read(), format='audio/mp3')

        st.markdown(f"**Definition:** {row.get('description_en', '')}")

        if row.get("Also Known as"):
            related = [t.strip() for t in row["Also Known as"].split(',')]
            st.markdown("**Also Known As:**")
            st.write(", ".join(related))

    with col2:
        st.markdown("---")
        st.subheader(row['term_es'])

        audio_file_es = generate_audio(row['term_es'], lang="es")
        if os.path.exists(audio_file_es):
            with open(audio_file_es, 'rb') as f:
                st.markdown("🔊 Haga clic en el altavoz para oír la pronunciación:")
                st.audio(f.read(), format='audio/mp3')

        st.markdown(f"**Definición:** {row.get('description_es', '')}")

    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# ------------------ Ask Gemini Section ------------------ #
st.markdown("## ❓ Ask a Question | Hacer una Pregunta")
col_en_q, col_es_q = st.columns(2)
api_key = "AIzaSyCWdz438r7X1Tk_PlP49yrGhDWRPCOmfn4"  # Replace with your Gemini API key (do not share publicly)

with col_en_q:
    st.markdown("**Ask any question about this glossary (English):**")
    english_q = st.text_input("Type your question in English", key="eng_qa")
    if english_q and api_key:
        with st.spinner("Thinking..."):
            full_context = "\n\n".join([f"{row['term_en']}: {row['description_en']}" for _, row in data.iterrows()])
            en_response = ask_gemini(english_q, full_context, api_key)
            st.success(en_response)

with col_es_q:
    st.markdown("**Haz una pregunta sobre este glosario (Español):**")
    spanish_q = st.text_input("Escribe tu pregunta en español", key="esp_qa")
    if spanish_q and api_key:
        with st.spinner("Pensando..."):
            full_context_es = "\n\n".join([f"{row['term_es']}: {row['description_es']}" for _, row in data.iterrows()])
            es_response = ask_gemini(spanish_q, full_context_es, api_key)
            st.success(es_response)
