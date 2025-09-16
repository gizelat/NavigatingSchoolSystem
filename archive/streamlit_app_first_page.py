"""
Streamlit Landing Page — Access 2 Community & Friendship (Dictionary Split View)
-------------------------------------------------------------------------------
This script creates the first page of a bilingual Streamlit web application
serving as an educational resource on the disability and juvenile justice
landscape in Massachusetts — now with a **split view** at the bottom:
- **Left pane:** alphabetical list of terms
- **Right pane:** selected term displayed like a dictionary entry with a large title,
  speaker button (text‑to‑speech), and a phonetic/IPA toggle (placeholder-ready)

The page reads a bilingual glossary with columns: term_en, description_en, term_es, description_es.
If those headers differ in your file, you can extend the alias map below.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

import streamlit as st
import pandas as pd

# --------------------------- Page configuration --------------------------- #
st.set_page_config(
    page_title="Access 2 Community & Friendship",
    page_icon="🧩",
    layout="wide",
)

# ------------------------------ Copy (i18n) ------------------------------- #
LANGUAGES = {"English": "en", "Español": "es"}
COPY: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "Access 2 Community & Friendship",
        "tagline": "Empowering connections across disability services and juvenile justice in Massachusetts.",
        "description": (
            "This site provides plain‑language explanations of key terms at the intersection of disability "
            "services and the juvenile justice system in Massachusetts. It supports families, administrators, "
            "advocates, and students, offering bilingual resources to help you navigate complex systems."
        ),
        "question_label": "Ask a question or search a term",
        "language_label": "Preferred language",
        "user_type_label": "I am a…",
        "user_types": ["Parent/Guardian", "Administrator", "Advocate", "Student"],
        "list_title": "All Terms (A–Z)",
        "search_terms": "Filter terms…",
        "preview_title": "Glossary — Dictionary View",
        "no_gloss": "Glossary not available. Ensure the Excel file is present.",
        "phonetic": "Phonetic (Standard)",
        "ipa": "IPA",
        "speak": "Play pronunciation",
    },
    "es": {
        "title": "Access 2 Community & Friendship",
        "tagline": "Conectando y empoderando en servicios de discapacidad y justicia juvenil en Massachusetts.",
        "description": (
            "Este sitio ofrece explicaciones claras de términos clave en la intersección de los servicios de "
            "discapacidad y el sistema de justicia juvenil en Massachusetts. Está diseñado para familias, "
            "administradores, defensores y estudiantes, brindando recursos bilingües para ayudarte a navegar "
            "sistemas complejos."
        ),
        "question_label": "Haz una pregunta o busca un término",
        "language_label": "Idioma preferido",
        "user_type_label": "Yo soy…",
        "user_types": ["Padre/Madre/Tutor", "Administrador/a", "Defensor/a", "Estudiante"],
        "list_title": "Todos los términos (A–Z)",
        "search_terms": "Filtrar términos…",
        "preview_title": "Glosario — Vista de diccionario",
        "no_gloss": "El glosario no está disponible. Verifica el archivo de Excel.",
        "phonetic": "Fonético (estándar)",
        "ipa": "AFI",
        "speak": "Reproducir pronunciación",
    },
}

USER_TYPES = {
    "en": ["Parent/Guardian", "Administrator", "Advocate", "Student"],
    "es": ["Padre/Madre/Tutor", "Administrador/a", "Defensor/a", "Estudiante"],
}

# ------------------------------ Paths / Backend ---------------------------- #

def find_first(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

GLOSSARY_PATH = find_first([
    Path("Juvenile Justice, Academic IEP, School System.xlsx"),
    Path("/mnt/data/Juvenile Justice, Academic IEP, School System.xlsx"),
])

LOGO_PATH = find_first([
    Path("Screenshot 2025-09-10 at 1.53.16 PM.png"),
    Path("Screenshot 2025-09-10 at 1.53.16\u202fPM.png"),
    Path("/mnt/data/Screenshot 2025-09-10 at 1.53.16 PM.png"),
    Path("/mnt/data/Screenshot 2025-09-10 at 1.53.16\u202fPM.png"),
])

@st.cache_data(show_spinner=False)
def load_glossary(path: Path) -> pd.DataFrame:
    import re
    df = pd.read_excel(path)
    # Normalize and alias
    def norm(s: str) -> str:
        return (
            s.strip().lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        )
    cols = {norm(c): c for c in df.columns}
    alias = {
        "term_en": ["term_en", "english_term", "term_english", "term"],
        "description_en": ["description_en", "definition", "english_description", "meaning"],
        "term_es": ["term_es", "spanish_term", "termino_es", "término_es"],
        "description_es": ["description_es", "spanish_description", "descripcion_es", "definicion_es"],
    }
    picked = {}
    for key, cands in alias.items():
        found = next((cols.get(norm(c)) for c in cands if cols.get(norm(c))), None)
        if not found:
            raise ValueError(f"Missing required column: {key}")
        picked[key] = df[found].astype(str).str.strip()
    df2 = pd.DataFrame(picked).dropna(how="all")
    return df2

# ---------------------------- Session defaults ---------------------------- #
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"
if "selected_term" not in st.session_state:
    st.session_state.selected_term = None

# ------------------------------- Styles ----------------------------------- #
st.markdown(
    """
    <style>
      body { background: #f8f5ef; }
      .page-wrap { max-width: 1200px; margin: 0 auto; }
      .subtitle { color:#374151; font-style: italic; margin-bottom: 1rem; }
      .callout { background:#eef3ff; border:1px solid #d9e2ff; padding:1rem 1.2rem; border-radius:12px; }
      .split { display:grid; grid-template-columns: 1fr 2fr; gap:1.25rem; margin-top:1.25rem; }
      @media (max-width: 980px) { .split { grid-template-columns: 1fr; } }
      .left-pane { background:#fff; border:1px solid #eadfca; border-radius:12px; padding:1rem; height: 60vh; overflow:auto; }
      .right-pane { background:#fff; border:1px solid #eadfca; border-radius:12px; padding:1.25rem; min-height:60vh; }
      .term-chip { padding:.35rem .6rem; border-radius:999px; background:#f3f4f6; border:1px solid #e5e7eb; display:inline-block; margin:.2rem; font-family: Georgia, serif; cursor:pointer; }
      .term-chip:hover { background:#e5e7eb; }
      .term-active { background:#1f2937; color:#fff; border-color:#1f2937; }
      .entry-title { font-size: 3rem; font-weight: 800; margin: .2rem 0 .6rem 0; letter-spacing:.3px; }
      .pronounce { display:flex; align-items:center; gap:.75rem; margin-bottom:1rem; }
      .speak-btn { width:42px; height:42px; border-radius:999px; border:1px solid #cbd5e1; display:flex; align-items:center; justify-content:center; cursor:pointer; background:#fff; }
      .phonetic-box { border:1px solid #e5e7eb; background:#f9fafb; padding:.5rem .75rem; border-radius:12px; display:inline-block; }
      .pos { font-weight:700; font-size:1.2rem; margin-top:1.2rem; }
      .definition { font-size:1.1rem; line-height:1.6; margin-top:.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Sidebar: language ---------------------------- #
st.sidebar.markdown("### 🌐 / 🌎 Bilingual Settings")
ui_lang = st.sidebar.selectbox(
    "Language / Idioma",
    options=[("English", "en"), ("Español", "es")],
    index=0 if st.session_state.ui_lang == "en" else 1,
    format_func=lambda o: o[0],
)[1]
st.session_state.ui_lang = ui_lang
text = COPY[ui_lang]

# -------------------------------- Banner ---------------------------------- #
col_logo, col_title = st.columns([1, 3], gap="large")
with col_logo:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), use_container_width=True)
with col_title:
    st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)
    st.markdown(f"# {text['title']}")
    st.markdown(f"<div class='subtitle'><strong>{text['tagline']}</strong></div>", unsafe_allow_html=True)

# ------------------------------ Description ------------------------------- #
st.markdown(f"<div class='callout'>{text['description']}</div>", unsafe_allow_html=True)

# ------------------------------ Load Glossary ------------------------------ #

gloss_df: Optional[pd.DataFrame] = None
if GLOSSARY_PATH and GLOSSARY_PATH.exists():
    try:
        gloss_df = load_glossary(GLOSSARY_PATH)
    except Exception as e:
        st.error(f"Error loading glossary: {e}")
else:
    st.warning(text["no_gloss"])

# ------------------------------ Footer note ------------------------------- #
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br><div style='text-align:center;color:#9a8f7a'>Access 2 Community & Friendship — Massachusetts • Page 1</div>", unsafe_allow_html=True)

# Close page wrapper
st.markdown("</div>", unsafe_allow_html=True)
