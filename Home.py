# landing_page.py (App)
# Navigating the School System — Bilingual Educational Resource (MA)
# Landing page + subject-restricted chatbot (Gemini optional). Users never enter a key.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional, List, Tuple
import os
import re
import unicodedata
import streamlit as st
import pandas as pd

# --------------------------- Page configuration --------------------------- #
st.set_page_config(
    page_title="Navigating the School System | MA Resource",
    page_icon="🧩",
    layout="wide",
)

# ------------------------------ Copy (i18n) ------------------------------- #
COPY: Dict[str, Dict[str, str]] = {
    "en": {
        "site_title": "Navigating the School System",
        "subtitle": "Helping families, students, and educators understand the academic and special education process in Massachusetts.",
        "desc": (
            "This resource offers clear, bilingual explanations of key terms and concepts related "
            "to the school system, IEPs, and academic supports in Massachusetts. It is designed for "
            "parents, administrators, advocates, and students. Content is sourced from a bilingual "
            "glossary (English/Spanish)."
        ),
        "language_label": "Language",
        "question_label": "Ask a question or enter a term",
        "question_ph": "e.g., What is an IEP? / ¿Qué es un IEP?",
        "user_type_label": "I am a…",
        "continue_btn": "Continue",
        "footer": "Your selections help personalize future pages.",
        "gloss_loaded": "✅ Glossary loaded ({n} rows)",
        "gloss_missing": "⚠️ Glossary file not found. The app will run, but lookups will be limited.",
        "logo_missing": "⚠️ Logo not found. Add it to the folder to show the banner image.",
        "source_file": "Source file: {name}",
        "chat_header": "Ask about IEPs, academic supports, and the school system in Massachusetts",
        "off_topic": "I can only answer questions related to the school system and glossary terms. Try asking about a term from the glossary.",
        "answer_note": "Answers are informational and not legal advice.",
    },
    "es": {
        "site_title": "Navegando el Sistema Escolar",
        "subtitle": "Ayudando a familias, estudiantes y educadores a comprender el proceso académico y de educación especial en Massachusetts.",
        "desc": (
            "Este recurso ofrece explicaciones claras y bilingües de términos y conceptos clave "
            "relacionados con el sistema escolar, los IEP y los apoyos académicos en Massachusetts. "
            "Está diseñado para familias, administradores, defensores y estudiantes. El contenido "
            "proviene de un glosario bilingüe (inglés/español)."
        ),
        "language_label": "Idioma",
        "question_label": "Escribe una pregunta o término",
        "question_ph": "p. ej., ¿Qué es un IEP? / What is an IEP?",
        "user_type_label": "Yo soy…",
        "continue_btn": "Continuar",
        "footer": "Tus selecciones ayudarán a personalizar las páginas futuras.",
        "gloss_loaded": "✅ Glosario cargado ({n} filas)",
        "gloss_missing": "⚠️ No se encontró el archivo del glosario. La app funcionará con funciones limitadas.",
        "logo_missing": "⚠️ No se encontró el logotipo. Agrégalo a la carpeta para mostrar el banner.",
        "source_file": "Archivo de origen: {name}",
        "chat_header": "Haz preguntas sobre IEP, apoyos académicos y el sistema escolar en Massachusetts",
        "off_topic": "Solo puedo responder preguntas relacionadas con el sistema escolar y los términos del glosario. Intenta con un término del glosario.",
        "answer_note": "Las respuestas son informativas y no constituyen asesoría legal.",
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
    Path("/Users/gizelathomas/Desktop/DisabilityProject/Screenshot 2025-09-10 at 1.53.16 PM.png"),
    Path("/Users/gizelathomas/Desktop/DisabilityProject/Screenshot 2025-09-10 at 1.53.16\u202fPM.png"),
    Path("Screenshot 2025-09-10 at 1.53.16 PM.png"),
    Path("Screenshot 2025-09-10 at 1.53.16\u202fPM.png"),
    Path("/mnt/data/Screenshot 2025-09-10 at 1.53.16 PM.png"),
    Path("/mnt/data/Screenshot 2025-09-10 at 1.53.16\u202fPM.png"),
])

# --------------------------- Glossary ingest ------------------------------ #
@st.cache_data(show_spinner=False)
def load_glossary(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_").replace("-", "_")
    cols = {norm(c): c for c in df.columns}
    alias = {
        "term_en": ["term_en", "english_term", "term(english)", "term"],
        "description_en": ["description_en", "english_description", "definition_en", "definition"],
        "term_es": ["term_es", "spanish_term", "termino_es", "término_es"],
        "description_es": ["description_es", "spanish_description", "definition_es", "descripcion_es"],
    }
    out = {}
    for k, names in alias.items():
        found = next((cols.get(norm(n)) for n in names if cols.get(norm(n))), None)
        if not found:
            raise ValueError(f"Missing required column: {k}")
        out[k] = df[found].astype(str).str.strip()
    return pd.DataFrame(out).dropna(how="all").reset_index(drop=True)

# ---------------------------- Relevance & Answers -------------------------- #
# Strict domain guardrails: answers must be grounded in the Excel glossary only.
RELEVANCE_MIN = 3
STRICT_MODE = True

STOP_EN = {"what","is","an","a","the","and","or","of","in","to","for","about","do","does","are","be","on"}
STOP_ES = {"qué","que","es","un","una","el","la","los","las","y","o","de","en","para","sobre","hacer","hace","son","ser"}

# --- Aliases strengthened for IEP and common variants ---
ALIASES = {
    "iep": ["iep", "individualized education program", "programa de educación individualizado"],
    "eval": ["evaluation", "special education evaluation", "evaluación de educación especial"],
}

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[^\w\s]", " ", s)

def _tokens(s: str, lang: str) -> List[str]:
    toks = _normalize(s).split()
    stop = STOP_EN if lang == "en" else STOP_ES
    return [t for t in toks if t not in stop]

def pick_context(df: pd.DataFrame, query: str, lang: str, k: int = 6) -> Tuple[pd.DataFrame, float]:
    if df is None or df.empty or not query:
        return pd.DataFrame(), 0.0
    term_col = "term_en" if lang == "en" else "term_es"
    desc_col = "description_en" if lang == "en" else "description_es"

    q_tokens = _tokens(query, lang)
    # Expand known aliases (e.g., IEP)
    for tok in list(q_tokens):
        if tok in ALIASES:
            q_tokens.extend(_tokens(" ".join(ALIASES[tok]), lang))

    scores = []
    for i, row in df.iterrows():
        term = str(row[term_col])
        desc = str(row[desc_col])
        t_norm = _normalize(term)
        d_norm = _normalize(desc)
        t_tokens = set(t_norm.split())
        d_tokens = set(d_norm.split())

        # exact/startswith boosts
        exact = 100 if _normalize(query) == t_norm else 0
        starts = 50 if t_norm.startswith(_normalize(query)) else 0

        overlap_term = len(set(q_tokens) & t_tokens)
        overlap_desc = len(set(q_tokens) & d_tokens)

        # Extra weight for short acronym-like matches (e.g., "iep")
        acronym_boost = sum(40 for tok in q_tokens if len(tok) <= 4 and tok in t_tokens)

        score = exact + starts + 4*overlap_term + 1*overlap_desc + acronym_boost
        scores.append((i, score))

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    idx = [i for i, s in top if s > 0]
    coverage = float(top[0][1]) if top else 0.0
    return (df.loc[idx] if idx else pd.DataFrame()), coverage


def suggest_terms(df: pd.DataFrame, query: str, lang: str, limit: int = 6) -> List[str]:
    if df is None or df.empty:
        return []
    term_col = "term_en" if lang == "en" else "term_es"
    desc_col = "description_en" if lang == "en" else "description_es"
    q = _normalize(query or "")
    try:
        mask = df[term_col].str.lower().str.contains(q, na=False) | df[desc_col].str.lower().str.contains(q, na=False)
        hits = df.loc[mask, term_col].head(limit).tolist()
        if hits:
            return hits
    except Exception:
        pass
    ctx, _ = pick_context(df, query, lang, k=limit)
    return ctx[term_col].head(limit).tolist() if not ctx.empty else []

# ---------------------------- Gemini configuration ------------------------ #

def _maybe_configure_gemini(api_key: Optional[str]):
    try:
        import google.generativeai as genai  # pip install google-generativeai
    except Exception:
        return None
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        return genai
    except Exception:
        return None

HARDCODED_GEMINI_KEY = "AIzaSyCWdz438r7X1Tk_PlP49yrGhDWRPCOmfn4"  # placeholder; use secrets/env in prod

def _safe_secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

if HARDCODED_GEMINI_KEY and HARDCODED_GEMINI_KEY != "AIzaSyCWdz438r7X1Tk_PlP49yrGhDWRPCOmfn4":
    GEMINI_API_KEY = HARDCODED_GEMINI_KEY
else:
    GEMINI_API_KEY = _safe_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-1.5-flash"
gemini = _maybe_configure_gemini(GEMINI_API_KEY)

def answer_with_gemini(genai, model_name: str, query: str, ctx: pd.DataFrame, lang: str) -> str:
    term_col = "term_en" if lang == "en" else "term_es"
    desc_col = "description_en" if lang == "en" else "description_es"
    context_text = "\n".join([f"- {r[term_col]}: {r[desc_col]}" for _, r in ctx.iterrows()])
    system = (
        "You are an educational assistant for the Navigating the School System site. "
        "Answer ONLY using the CONTEXT facts provided (which were sourced from the site's Excel glossary). "
        "If the answer is not in CONTEXT, say you don't have that in the glossary and offer a few related terms. "
        "Never use outside knowledge and never provide legal advice. Keep responses concise and clear."
    )
    instruction = (
        f"Respond in {'Spanish' if lang=='es' else 'English'}. "
        "Use brief paragraphs and bullets when useful.\n"
        "When appropriate, include a short list of glossary terms you used.\n"
        "CONTEXT:\n" + context_text + "\n---\n" + f"USER QUESTION: {query}"
    )
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content([system, instruction])
        return (resp.text or "").strip()
    except Exception as e:
        return f"(Gemini error) {e}"

def answer_locally(query: str, ctx: pd.DataFrame, lang: str) -> str:
    term_col = "term_en" if lang == "en" else "term_es"
    desc_col = "description_en" if lang == "en" else "description_es"
    if ctx is None or ctx.empty:
        return COPY[lang]["off_topic"]
    t = str(ctx.iloc[0][term_col])
    d = str(ctx.iloc[0][desc_col])
    note = COPY[lang]["answer_note"]
    return f"**{t}** — {d}\n\n_{note}_"

# ---------------------------- Session defaults ---------------------------- #
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"
if "chat" not in st.session_state:
    st.session_state.chat: List[Tuple[str, str]] = []  # (role, content)

# --------------------------- Sidebar: language ---------------------------- #
with st.sidebar:
    st.markdown("### 🌐 / 🌎")
    ui_lang = st.selectbox(
        "Language / Idioma",
        options=[("English", "en"), ("Español", "es")],
        index=0 if st.session_state.ui_lang == "en" else 1,
        format_func=lambda o: o[0],
    )[1]
    st.session_state.ui_lang = ui_lang

t = COPY[st.session_state.ui_lang]

# ------------------------------- Styles ----------------------------------- #
st.markdown(
    """
    <style>
      .a2cf-subtitle { color:#334155; font-size:1.05rem; margin-bottom:0.5rem; }
      .a2cf-card {
          background:#ffffff; border:1px solid #e5e7eb; border-radius:14px;
          padding:1.1rem 1.2rem; box-shadow:0 1px 2px rgba(0,0,0,.04);
      }
      .a2cf-desc { color:#1f2937; line-height:1.55; }
      .chatbox { background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding: .75rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------- Banner ---------------------------------- #
col_logo, col_title = st.columns([1, 3], gap="large")
with col_logo:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.info(t["logo_missing"])
with col_title:
    st.markdown(f"# {t['site_title']}")
    st.markdown(f'<div class="a2cf-subtitle"><strong>{t["subtitle"]}</strong></div>', unsafe_allow_html=True)

st.divider()

# ------------------------------ Description ------------------------------- #
st.markdown(f'<div class="a2cf-card a2cf-desc">{t["desc"]}</div>', unsafe_allow_html=True)
st.write("")

# --------------------------- Load glossary (backend) ---------------------- #
gloss_df = None
if GLOSSARY_PATH and GLOSSARY_PATH.exists():
    try:
        gloss_df = load_glossary(GLOSSARY_PATH)
        st.caption(t["gloss_loaded"].format(n=len(gloss_df)) + " • " + t["source_file"].format(name=GLOSSARY_PATH.name))
    except Exception as e:
        st.error(f"Error loading glossary: {e}")
else:
    st.warning(t["gloss_missing"])

# ---------------------------- Gemini status badge ------------------------- #
gemini_ready = bool(gemini and GEMINI_API_KEY)
st.caption("🤖 Gemini: connected" if gemini_ready else "⚠️ Gemini: not connected")

# ------------------------------- User Inputs ------------------------------ #
with st.form("first_page_form", clear_on_submit=False):
    q_col, lang_col, role_col = st.columns([2, 1, 1])
    with q_col:
        question = st.text_input(t["question_label"], placeholder=t["question_ph"])
    with lang_col:
        pref_lang = st.selectbox(t["language_label"], ["English", "Español"],
                                 index=0 if st.session_state.ui_lang == "en" else 1)
    with role_col:
        user_type = st.selectbox(t["user_type_label"], USER_TYPES[st.session_state.ui_lang])
    submitted = st.form_submit_button(t["continue_btn"])

if submitted:
    st.session_state.ui_lang = "en" if pref_lang == "English" else "es"
    st.session_state.user_type = user_type
    st.success("Preferences saved.")

# --------------------------------- Q&A (form-driven) ---------------------- #
if submitted:
    user_msg = (question or "").strip()
    if user_msg:
        st.session_state.chat.append(("user", user_msg))
        ctx, coverage = pick_context(
            gloss_df if gloss_df is not None else pd.DataFrame(),
            user_msg, st.session_state.ui_lang, k=6
        )

        # If Gemini is ready and we have ANY context rows, use Gemini.
        if gemini and GEMINI_API_KEY and ctx is not None and not ctx.empty:
            reply = answer_with_gemini(gemini, MODEL_NAME, user_msg, ctx, st.session_state.ui_lang)
        else:
            # Otherwise, strict guardrails + local fallback
            on_topic = (ctx is not None and not ctx.empty and coverage >= RELEVANCE_MIN)
            if not on_topic and STRICT_MODE:
                terms = suggest_terms(gloss_df, user_msg, st.session_state.ui_lang, limit=6) if gloss_df is not None else []
                suggestion_text = ("\n\n**Related terms in the glossary:** " + ", ".join(terms)) if terms else ""
                reply = t["off_topic"] + suggestion_text
            else:
                reply = answer_locally(user_msg, ctx, st.session_state.ui_lang)

        st.session_state.chat.append(("assistant", reply))

# --------------------------- Chat history render -------------------------- #
if st.session_state.chat:
    with st.container():
        for role, content in st.session_state.chat:
            with st.chat_message(role):
                st.markdown(content)
        st.caption(t["answer_note"])

# ------------------------------ Debug panel (optional) -------------------- #
with st.expander("🔍 Debug: show matched context", expanded=False):
    if gloss_df is None:
        st.write("No glossary loaded.")
    elif any(r == "user" for r, _ in st.session_state.chat):
        last_user = [c for r, c in st.session_state.chat if r == "user"][-1]
        dbg_ctx, dbg_cov = pick_context(gloss_df, last_user, st.session_state.ui_lang, k=6)
        st.write(f"Coverage score: {dbg_cov}")
        st.dataframe(dbg_ctx.head(6))

# ------------------------------ Footer note ------------------------------- #
st.caption(t["footer"])
