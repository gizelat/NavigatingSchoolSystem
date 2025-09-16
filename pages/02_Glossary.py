# pages/02_Glossary.py
import os
from pathlib import Path
import json
import requests
from io import BytesIO
from functools import lru_cache

import streamlit as st
import pandas as pd
from gtts import gTTS

# --------------------------- Page header --------------------------- #
st.set_page_config(page_title="Navigating the School System — Glossary", page_icon="🧩", layout="wide")

st.markdown("## Navigating the School System")
st.caption("Juvenile Justice, Academic IEP, & School System — Bilingual Glossary")

# --------------------------- Load data ----------------------------- #
@st.cache_data(show_spinner=False)
def load_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()
    df.fillna("", inplace=True)

    # Canonical columns (map from likely variants)
    col_map = {
        "term_en": ["term_en", "Term EN", "english_term", "term(english)", "term"],
        "description_en": ["description_en", "English Definition", "definition_en", "definition"],
        "term_es": ["term_es", "Term ES", "spanish_term", "termino_es", "término_es"],
        "description_es": ["description_es", "Spanish Definition", "definition_es", "descripcion_es"],
        # Synonyms (optional)
        "also_known_as_en": ["also_known_as_en", "also_known_as", "Also Known as", "Also Known As", "aka", "AKAs"],
        "also_known_as_es": ["also_known_as_es", "sinonimos", "sinónimos", "tambien_conocido_como"],
    }

    selected = {}
    for canon, candidates in col_map.items():
        for c in candidates:
            if c in df.columns:
                selected[canon] = c
                break

    # Required
    required = ["term_en", "description_en", "term_es", "description_es"]
    for r in required:
        if r not in selected:
            raise ValueError(f"Missing required column for {r}. Found columns: {df.columns.tolist()}")

    # Tidy frame
    out = pd.DataFrame({
        "term_en": df[selected["term_en"]],
        "description_en": df[selected["description_en"]],
        "term_es": df[selected["term_es"]],
        "description_es": df[selected["description_es"]],
        "also_known_as_en": df[selected["also_known_as_en"]] if "also_known_as_en" in selected else "",
        "also_known_as_es": df[selected["also_known_as_es"]] if "also_known_as_es" in selected else "",
    })
    return out

DATA_PATHS = [
    "Juvenile Justice, Academic IEP, School System.xlsx",
    "/mnt/data/Juvenile Justice, Academic IEP, School System.xlsx",
]
xlsx = next((p for p in DATA_PATHS if Path(p).exists()), None)

if not xlsx:
    st.error("Glossary Excel file not found.")
    st.stop()

data = load_data(xlsx).copy()
data.sort_values(by=["term_en", "term_es"], key=lambda s: s.str.lower(), inplace=True, ignore_index=True)
st.caption(f"✅ Glossary loaded ({len(data)} rows) • Source file: {Path(xlsx).name}")

# --------------------------- Gemini helper ------------------------- #
def ask_gemini(question: str, context: str) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini is not configured on the server."

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {
        "contents": [{
            "parts": [{
                "text": (
                    "You are answering ONLY from the provided GLOSSARY. "
                    "If the answer is not covered in the glossary, say you don't have that in the glossary "
                    "and suggest a few related terms.\n\n"
                    f"GLOSSARY:\n{context}\n\nQUESTION:\n{question}"
                )
            }]
        }]
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
        if r.status_code == 200:
            return r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip() or "No response."
        return f"Gemini error ({r.status_code}): {r.text[:200]}"
    except Exception as e:
        return f"Gemini exception: {e}"

# --------------------------- Audio (in-memory) ---------------------- #
@lru_cache(maxsize=512)
def _audio_for(text: str, lang: str) -> bytes | None:
    """Return MP3 bytes for pronunciation without writing to disk."""
    try:
        buf = BytesIO()
        gTTS(text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def _parse_synonyms(val: str) -> list[str]:
    if not isinstance(val, str) or not val.strip():
        return []
    raw = [p.strip() for p in val.replace(";", ",").split(",")]
    return [p for p in raw if p]

def _render_synonyms(title: str, items: list[str]):
    if not items:
        return
    chips = " ".join(
        f"<span style='display:inline-block;margin:4px 6px;padding:.15rem .5rem;border:1px solid #e5e7eb;border-radius:9999px;font-size:.9rem;'>{st.escape_markdown(x)}</span>"
        for x in items
    )
    st.markdown(f"**{title}:**<br/>{chips}", unsafe_allow_html=True)

# --------------------------- TABS UI ------------------------------- #
tab_browse, tab_ask = st.tabs(["📚 Browse Glossary (one term)", "❓ Ask Gemini"])

# --------------------------- Session state ------------------------- #
if "browse_idx" not in st.session_state:
    st.session_state.browse_idx = 0
if "last_filter_key" not in st.session_state:
    st.session_state.last_filter_key = ""

with tab_browse:
    st.markdown("### Navigation & Search")
    col_search, col_letter = st.columns([2, 5])

    with col_search:
        search_term = st.text_input("Search for a term (English)", placeholder="e.g., IEP")

    with col_letter:
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        default_letter_index = 0
        if 0 <= st.session_state.browse_idx < len(data):
            first_letter = data.loc[st.session_state.browse_idx, "term_en"][:1].upper()
            if first_letter in letters:
                default_letter_index = letters.index(first_letter)
        selected_letter = st.radio("Jump to Letter:", letters, horizontal=True, index=default_letter_index)

    # Build filtered list
    filtered = data.copy()
    if selected_letter:
        filtered = filtered[filtered["term_en"].str.startswith(selected_letter, na=False)]
    if search_term:
        filtered = filtered[filtered["term_en"].str.contains(search_term, case=False, na=False)]

    filtered.sort_values(by=["term_en", "term_es"], key=lambda s: s.str.lower(), inplace=True, ignore_index=True)

    # Reset index if filters changed
    filter_key = f"{selected_letter}|{(search_term or '').strip().lower()}"
    if filter_key != st.session_state.last_filter_key:
        st.session_state.browse_idx = 0
        st.session_state.last_filter_key = filter_key

    if filtered.empty:
        st.warning("No terms found with the current filter(s).")
        st.stop()

    # Clamp & pick row
    idx = max(0, min(st.session_state.browse_idx, len(filtered) - 1))
    row = filtered.iloc[idx]

    # One-term view
    left, right = st.columns(2)

    with left:
        st.markdown("### ENGLISH")
        st.subheader(row["term_en"])
        audio_en = _audio_for(row["term_en"], "en")
        if audio_en:
            st.audio(audio_en, format="audio/mp3")
        st.markdown(f"**Definition:** {row['description_en']}")
        _render_synonyms("Synonyms / Also Known As", _parse_synonyms(row.get("also_known_as_en", "")))

    with right:
        st.markdown("### ESPAÑOL")
        st.subheader(row["term_es"])
        audio_es = _audio_for(row["term_es"], "es")
        if audio_es:
            st.audio(audio_es, format="audio/mp3")
        st.markdown(f"**Definición:** {row['description_es']}")
        _render_synonyms("Sinónimos", _parse_synonyms(row.get("also_known_as_es", "")))

    st.markdown("---")

    # Pager controls
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        prev_disabled = (idx <= 0)
        if st.button("← Previous", disabled=prev_disabled, use_container_width=True):
            st.session_state.browse_idx = max(0, idx - 1)
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<strong>{idx + 1}</strong> of <strong>{len(filtered)}</strong> terms"
            f"</div>",
            unsafe_allow_html=True
        )
    with c3:
        next_disabled = (idx >= len(filtered) - 1)
        if st.button("Next →", disabled=next_disabled, use_container_width=True):
            st.session_state.browse_idx = min(len(filtered) - 1, idx + 1)
            st.rerun()

with tab_ask:
    st.markdown("### Ask a Question | Hacer una pregunta")
    col_en, col_es = st.columns(2)

    with col_en:
        q_en = st.text_input("Ask in English", key="ask_en")
        if q_en:
            with st.spinner("Thinking..."):
                ctx_en = "\n".join([f"- {r.term_en}: {r.description_en}" for r in data.itertuples(index=False)])
                ans_en = ask_gemini(q_en, ctx_en)
                st.success(ans_en)

    with col_es:
        q_es = st.text_input("Pregunta en español", key="ask_es")
        if q_es:
            with st.spinner("Pensando..."):
                ctx_es = "\n".join([f"- {r.term_es}: {r.description_es}" for r in data.itertuples(index=False)])
                ans_es = ask_gemini(q_es, ctx_es)
                st.success(ans_es)
