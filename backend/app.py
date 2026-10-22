import os
import re
import io
import gc
import uuid
import string
import logging
import time
from typing import List, Optional
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import asyncio
import random
import json as _json

import threading

# ---- Chroma ----
import chromadb
from chromadb.config import Settings

# ---- PDF + OCR ----
import pymupdf as fitz  # PyMuPDF
from PIL import Image
try:
    import pytesseract
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False

# ---- BM25 ----
from rank_bm25 import BM25Okapi
LEX_CORPUS: List[str] = []
LEX_META: List[dict] = []
BM25 = None

# ---- Torch / Reranker ----
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================== Config ==================
load_dotenv()
CHROMA_HOST          = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT          = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION           = os.getenv("COLLECTION", "kb_docs")
EMBED_MODEL          = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
GEM_BASE             = os.getenv("GEM_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEM_KEY              = os.getenv("GEMINI_API_KEY") or os.getenv("GEM_KEY")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DEFAULT_TOP_K        = int(os.getenv("DEFAULT_TOP_K", "6"))
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.8-flash")
ENABLE_RERANKER      = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
RERANKER_MODEL       = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_TOP_K       = int(os.getenv("RERANKER_TOP_K", "6"))
RERANKER_MAXLEN      = int(os.getenv("RERANKER_MAXLEN", "512"))
ENABLE_WATCHER       = os.getenv("ENABLE_WATCHER", "false").lower() == "true"

if not GEM_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY en el entorno")
if not GEM_BASE.endswith("/"):
    raise RuntimeError("GEM_BASE debe terminar con '/'. Ej: https://.../v1beta/openai/")

# ================== App & Middleware ==================
app = FastAPI(title="RAG Backend – Denso + Léxico + Gemini/Ollama")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag-pro")

# ================== Chroma ==================
chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(allow_reset=True)
)
collection = chroma_client.get_or_create_collection(name=COLLECTION)

# ================== Lazy-load embedder ==================
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

# ================== Reranker ==================
def _pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@lru_cache(maxsize=1)
def get_reranker():
    tok = AutoTokenizer.from_pretrained(RERANKER_MODEL, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    mdl.eval()
    mdl.to(_pick_device())
    return tok, mdl

def rerank_blocks(query: str, blocks: list, keep_top_k: int = None, max_length: int = None) -> list:
    if not ENABLE_RERANKER or not blocks:
        return blocks
    keep    = keep_top_k if keep_top_k is not None else RERANKER_TOP_K
    max_len = max_length  if max_length  is not None else RERANKER_MAXLEN
    try:
        tok, mdl = get_reranker()
        dev = _pick_device()
        scores = []
        with torch.no_grad():
            for b in blocks:
                inputs = tok(
                    query, b,
                    truncation=True, max_length=max_len,
                    padding="max_length", return_tensors="pt"
                ).to(dev)
                out = mdl(**inputs).logits.squeeze()
                score = float(out.item()) if out.dim() == 0 else float(out[0].item())
                scores.append((score, b))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [b for (_, b) in scores[:max(1, min(keep, len(scores)))]]
    except Exception as e:
        logger.warning(f"[reranker] error: {e}")
        return blocks[:keep]

# ================== RRF ==================
def rrf_fusion(dense_blocks: list, lex_blocks: list, k: int = 60, top_k: int = 6) -> list:
    scores = {}
    for rank, block in enumerate(dense_blocks):
        scores[block] = scores.get(block, 0) + 1 / (k + rank)
    for rank, block in enumerate(lex_blocks):
        scores[block] = scores.get(block, 0) + 1 / (k + rank)
    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in reranked[:top_k]]

# ================== Modelos de datos ==================
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    model: str = Field(default=DEFAULT_GEMINI_MODEL)
    temperature: float = Field(default=0.6, ge=0.0, le=1.0)

class LexicalRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)

class HybridRetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=50)

class RerankTestRequest(BaseModel):
    query: str
    blocks: list
    top_k: int = Field(default=6, ge=1, le=50)

class OpenAIChatRequest(BaseModel):
    model: str = DEFAULT_GEMINI_MODEL
    messages: list
    temperature: float = 0.6
    stream: bool = False
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

# ================== Detección de fases (mejorada) ==================
# Keywords más precisos para evitar que "roma" marque todo como fase 4
PHASE_PATTERNS = {
    1: [
        "fase 1", "fase i ", "fase i.", "primera fase",
        "nucleación", "nacimiento de los príncipes",
        "siglo viii", "siglo vii", "siglo vi",
        "período orientalizante", "periodo orientalizante",
        "primeras aristocracias", "jefatura",
    ],
    2: [
        "fase 2", "fase ii ", "fase ii.", "segunda fase",
        "expansión territorial", "clientelas",
        "siglo v a.n.e", "siglo v a.c",
        "consolidación aristocrática", "oppida iniciales",
        "formación del estado", "emergencia del estado",
    ],
    3: [
        "fase 3", "fase iii ", "fase iii.", "tercera fase",
        "territorios políticos", "estados iberos",
        "siglo iv a.n.e", "siglo iii a.n.e",
        "oppida principales", "ciudad-estado ibera",
        "plena cultura ibérica", "apogeo ibero",
    ],
    4: [
        "fase 4", "fase iv ", "fase iv.", "cuarta fase",
        "desaparición de los estados iberos",
        "hibridación de la sociedad ibera",
        "bajo el poder de roma", "conquista romana",
        "segunda guerra púnica", "batalla de baecula",
        "romanización ibera", "municipalización de vespasiano",
        "208 a.n.e", "206 a.n.e",
    ],
}

def detect_phase(text: str) -> Optional[int]:
    t = text.lower()
    for phase, keywords in PHASE_PATTERNS.items():
        if any(k in t for k in keywords):
            return phase
    return None

# ================== Utilidades texto ==================
def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def _chunk(text: str, max_chars=1200, overlap=150, min_len=80) -> List[str]:
    text = text.strip()
    n = len(text)
    if n == 0:
        return []
    out, start = [], 0
    while start < n:
        end = min(start + max_chars, n)
        part = text[start:end].strip()
        if len(part) > min_len:
            out.append(part)
        if end == n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return out

def _ocr_page(page, dpi=120, lang="spa+eng") -> str:
    if not TESSERACT_OK:
        return ""
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    return _clean_text(pytesseract.image_to_string(img, lang=lang, config="--oem 1 --psm 6") or "")

def build_context_blocks(docs: List[str], metas: List[dict]) -> List[str]:
    blocks = []
    for d, m in zip(docs, metas):
        src   = m.get("source", "desconocido")
        pg    = m.get("page", "?")
        phase = m.get("phase")
        phase_tag = f" [Fase {phase}]" if phase else ""
        blocks.append(f"[{src} p.{pg}{phase_tag}]\n{d.strip()}")
    return blocks

def filter_blocks(blocks: list, intent: str) -> list:
    if intent not in ("definition", "historical_summary"):
        return blocks
    blacklist = ["bibliografía", "catálogo", "revista", "archivo español"]
    return [b for b in blocks if not any(x in b.lower() for x in blacklist)]

# ================== Query expansion e intent ==================
def expand_query(query: str) -> list:
    q = query.lower()
    expansions = [query]

    if "oppidum" in q or "oppida" in q:
        expansions += [
            "oppidum ibérico definición función urbana",
            "oppida iberos asentamiento fortificado",
        ]

    if any(k in q for k in ["fases", "proceso histórico", "resume", "etapas",
                              "periodización", "principales fases"]):
        expansions += [
            # Fase 1
            "nucleación ibera nacimiento príncipes orientalizante",
            "primera fase iberos Jaén siglo VIII VII VI",
            "jefatura aristocracia ibera origen formación",
            # Fase 2
            "segunda fase iberos expansión clientelas siglo V",
            "consolidación aristocrática oppida formación estado ibero",
            # Fase 3
            "tercera fase iberos estados territoriales siglo IV III",
            "apogeo ibero ciudad estado oppida principales Jaén",
            "plena cultura ibérica Jaén campiña",
            # Fase 4
            "cuarta fase romanización hibridación iberos Roma",
            "segunda guerra púnica Baecula conquista romana Jaén",
            "municipalización Vespasiano iberos desaparición estados",
            # General
            "Arturo Ruiz periodización historia iberos Jaén",
            "proceso histórico ibero Jaén cronología",
        ]

    if any(k in q for k in ["qué es", "que es", "define", "definición", "concepto"]):
        expansions.append("definición arqueológica " + query)

    return list(dict.fromkeys(expansions))

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["qué es", "que es", "define", "definición", "concepto"]):
        return "definition"
    if any(k in q for k in ["resume", "resumen", "principales fases", "fases del proceso",
                              "proceso histórico", "etapas", "periodización"]):
        return "historical_summary"
    return "general"

# ================== Prompt builder ==================
def build_prompt(query: str, blocks: list, intent: str = "general",
                 max_ctx_chars: int = 14000) -> str:
    ctx_parts, total = [], 0
    for b in blocks:
        if total + len(b) > max_ctx_chars:
            break
        ctx_parts.append(b)
        total += len(b)
    ctx = "\n\n".join(ctx_parts)

    if intent == "definition":
        instruction = (
            "Redacta una definición clara, sintética y académica del concepto preguntado. "
            "Empieza con «Un/Una X es…». No incluyas información narrativa secundaria."
        )
    elif intent == "historical_summary":
        instruction = (
            "Eres un especialista en arqueología ibera. El proceso histórico ibero en Jaén "
            "se divide en CUATRO fases cronológicas bien definidas según la historiografía "
            "(Arturo Ruiz y Manuel Molinos). "
            "Con el contexto documental proporcionado:\n"
            "1. Identifica y describe las CUATRO fases en orden cronológico.\n"
            "2. Para cada fase indica: nombre completo, cronología aproximada, "
            "características sociopolíticas y económicas clave.\n"
            "3. Si en el contexto no aparece información explícita de alguna fase, "
            "indícalo brevemente pero NO omitas la fase del esquema.\n"
            "4. Usa un encabezado claro para cada fase (Fase 1, Fase 2, etc.).\n"
            "El objetivo es un resumen académico completo de las cuatro fases."
        )
    else:
        instruction = (
            "Responde de forma detallada y académica usando el contexto proporcionado. "
            "Cita las fuentes al final."
        )

    return (
        f"{instruction}\n\n"
        "=== CONTEXTO DOCUMENTAL ===\n"
        f"{ctx}\n\n"
        "=== PREGUNTA ===\n"
        f"{query}\n\n"
        "Responde en español. Cita fuentes entre corchetes [archivo p.página]."
    )

# ================== Ollama helpers ==================
async def _get_ollama_models() -> list:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                return [
                    {"id": m["name"], "object": "model", "owned_by": "ollama", "created": 0}
                    for m in r.json().get("models", [])
                ]
    except Exception as e:
        logger.warning(f"[ollama] no disponible: {e}")
    return []

async def _call_ollama_chat(model: str, messages: list, temperature: float,
                             stream: bool = False) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature}
    }
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

def _is_ollama_model(model_id: str) -> bool:
    return not any(model_id.startswith(p) for p in ("gemini-", "models/gemini"))

def _get_gemini_models() -> list:
    return [
        {"id": "gemini-3.8-flash", "object": "model", "owned_by": "google", "created": 0},
        {"id": "gemini-3.7-flash", "object": "model", "owned_by": "google", "created": 0},
        {"id": "gemini-3.6-flash", "object": "model", "owned_by": "google", "created": 0},
        {"id": "gemini-3.5-flash", "object": "model", "owned_by": "google", "created": 0},
    ]

# ================== LLM calls ==================
async def call_gemini_chat_completions(model: str, prompt: str,
                                        temperature: float = 0.6) -> str:
    headers = {"Authorization": f"Bearer {GEM_KEY}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Eres un asistente experto en arqueología ibera."},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature
    }
    max_attempts, base = 3, 1.0
    async with httpx.AsyncClient(timeout=120) as s:
        for attempt in range(1, max_attempts + 1):
            try:
                r = await s.post(f"{GEM_BASE}chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if (status == 429 or 500 <= status < 600) and attempt < max_attempts:
                    wait = base * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"{status} {e.response.text}") from e
            except httpx.RequestError as e:
                if attempt < max_attempts:
                    await asyncio.sleep(base * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2))
                    continue
                raise RuntimeError(f"RequestError: {e}") from e
    raise RuntimeError("Gemini no respondió tras reintentos")

async def call_llm_with_fallback(model: str, prompt: str,
                                  temperature: float = 0.6) -> str:
    """Intenta Gemini; si falla por cuota/error, usa Ollama automáticamente."""
    if not _is_ollama_model(model):
        try:
            return await call_gemini_chat_completions(model, prompt, temperature)
        except RuntimeError as e:
            logger.warning(f"[LLM] Gemini falló: {str(e)[:120]}. Intentando Ollama...")

    # Fallback o modelo Ollama directo
    ollama_models = await _get_ollama_models()
    if not ollama_models:
        raise RuntimeError("Gemini falló y Ollama no tiene modelos disponibles.")

    preferred = ["llama3.1:8b", "llama3:8b", "phi3:mini", "gemma2:2b"]
    fallback_model = ollama_models[0]["id"]
    for pref in preferred:
        if any(m["id"] == pref for m in ollama_models):
            fallback_model = pref
            break

    logger.info(f"[LLM] Usando Ollama: {fallback_model}")
    messages = [
        {"role": "system", "content": "Eres un asistente experto en arqueología ibera."},
        {"role": "user",   "content": prompt}
    ]
    return await _call_ollama_chat(fallback_model, messages, temperature)

# ================== BM25 ==================
def _tok(s: str) -> list:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"\w+", s, flags=re.UNICODE)

def build_bm25_from_chroma(batch: int = 1000, cap: Optional[int] = 8000) -> int:
    global LEX_CORPUS, LEX_META, BM25
    LEX_CORPUS, LEX_META = [], []
    offset = 0
    while True:
        res  = collection.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        docs  = res.get("documents", [])
        metas = res.get("metadatas", [])
        if not docs:
            break
        for d, m in zip(docs, metas):
            if d and len(d) >= 80:
                LEX_CORPUS.append(d)
                LEX_META.append(m)
                if cap and len(LEX_CORPUS) >= cap:
                    break
        if cap and len(LEX_CORPUS) >= cap:
            break
        offset += batch
    if LEX_CORPUS:
        BM25 = BM25Okapi([_tok(d) for d in LEX_CORPUS])
    return len(LEX_CORPUS)

# ================== Recuperación por fases (garantizada) ==================
def fetch_phase_blocks(phases: list = [1, 2, 3, 4],
                       chunks_per_phase: int = 3) -> list:
    """
    Recupera chunks etiquetados con phase=N directamente desde Chroma.
    Garantiza cobertura de todas las fases en resúmenes históricos.
    """
    phase_blocks = []
    for phase in phases:
        try:
            res = collection.get(
                where={"phase": phase},
                limit=chunks_per_phase,
                include=["documents", "metadatas"]
            )
            docs  = res.get("documents", [])
            metas = res.get("metadatas", [])
            for d, m in zip(docs, metas):
                src = m.get("source", "?")
                pg  = m.get("page", "?")
                phase_blocks.append(f"[{src} p.{pg} [Fase {phase}]]\n{d.strip()}")
        except Exception as e:
            logger.warning(f"[fetch_phase_blocks] fase {phase}: {e}")
    return phase_blocks

# ================== ENDPOINTS ==================
@app.get("/health")
def health():
    return {"ok": True}

# ---------- Ingesta ----------
@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...), ocr: bool = Form(False)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan PDFs")
    pdf_bytes = await file.read()
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="PDF inválido")

    documents, metadatas = [], []

    for page_idx, page in enumerate(doc):
        text = _clean_text(page.get_text() or "")
        if not text and ocr:
            text = _ocr_page(page)
        if not text:
            continue
        for chunk in _chunk(text):
            phase = detect_phase(chunk)
            meta  = {"source": file.filename, "page": page_idx + 1}
            if phase is not None:
                meta["phase"] = phase
            documents.append(chunk)
            metadatas.append(meta)

    if not documents:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF")

    embedder   = get_embedder()
    embeddings = embedder.encode(documents, normalize_embeddings=True).tolist()
    collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in documents]
    )
    page_count = len(doc)
    doc.close()
    return {"ok": True, "pages": page_count, "chunks": len(documents)}

# ---------- Retrieve (denso) ----------
@app.post("/retrieve", operation_id="rag_retrieve")
def retrieve(req: RetrieveRequest):
    emb  = get_embedder()
    qemb = emb.encode([req.query], normalize_embeddings=True).tolist()
    res  = collection.query(
        query_embeddings=qemb,
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs   = res.get("documents", [[]])[0]
    metas  = res.get("metadatas", [[]])[0]
    blocks = build_context_blocks(docs, metas)
    blocks = rerank_blocks(req.query, blocks, keep_top_k=req.top_k)
    return {
        "query": req.query, "top_k": req.top_k,
        "count": len(blocks), "contexts": blocks,
        "distances": res.get("distances", [[]])[0]
    }

# ---------- Answer (RAG completo) ----------
@app.post("/answer", operation_id="rag_answer")
async def answer(req: AnswerRequest):

    intent  = detect_intent(req.query)
    queries = expand_query(req.query)

    # ── Ajustes de recuperación según intención ──────────────────
    if intent == "historical_summary":
        dense_k = 30
        lex_k   = 30
        final_k = 24  # fijo: suficiente para cubrir 4 fases con detalle
    elif intent == "definition":
        dense_k = 8
        lex_k   = 12
        final_k = 5
    else:
        dense_k = 12
        lex_k   = 12
        final_k = req.top_k

    # ── Bloques garantizados por fase (solo en resúmenes históricos) ──
    guaranteed_blocks = []
    if intent == "historical_summary":
        guaranteed_blocks = fetch_phase_blocks(phases=[1, 2, 3, 4], chunks_per_phase=3)
        logger.info(f"[answer] Bloques garantizados por fase: {len(guaranteed_blocks)}")

    # ── Dense retrieval con query expansion ──────────────────────
    emb = get_embedder()
    dense_blocks_all = []
    seen_dense = set()

    for q in queries:
        qemb = emb.encode([q], normalize_embeddings=True).tolist()
        res  = collection.query(
            query_embeddings=qemb,
            n_results=dense_k,
            include=["documents", "metadatas", "distances"]
        )
        docs  = res["documents"][0]
        metas = res["metadatas"][0]
        for blk in build_context_blocks(docs, metas):
            key = blk[:120]
            if key not in seen_dense:
                seen_dense.add(key)
                dense_blocks_all.append(blk)

    # ── Lexical (BM25) ────────────────────────────────────────────
    lex_blocks = []
    if BM25:
        qtok   = _tok(req.query)
        scores = BM25.get_scores(qtok)
        idx    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:lex_k]
        lex_blocks = [
            f"[{LEX_META[i].get('source','?')} p.{LEX_META[i].get('page','?')}]\n{LEX_CORPUS[i]}"
            for i in idx
        ]
    else:
        logger.warning("[answer] BM25 no disponible — solo dense retrieval")

    # ── Fusión RRF ────────────────────────────────────────────────
    fused = rrf_fusion(dense_blocks_all, lex_blocks, k=60, top_k=final_k)

    # ── Filtro de ruido ───────────────────────────────────────────
    fused = filter_blocks(fused, intent)

    # ── Combinar: garantizados primero + fusionados (sin duplicar) ─
    seen_final = set()
    final_blocks = []

    # Primero los garantizados (aseguran cobertura de todas las fases)
    for blk in guaranteed_blocks:
        key = blk[:120]
        if key not in seen_final:
            seen_final.add(key)
            final_blocks.append(blk)

    # Luego los del RRF
    for blk in fused:
        key = blk[:120]
        if key not in seen_final:
            seen_final.add(key)
            final_blocks.append(blk)

    # ── Reranker: solo sobre los bloques NO garantizados ─────────
    # (los garantizados se preservan siempre)
    if len(final_blocks) > len(guaranteed_blocks):
        reranked_tail = rerank_blocks(
            req.query,
            final_blocks[len(guaranteed_blocks):],
            keep_top_k=max(final_k - len(guaranteed_blocks), 4)
        )
        final_blocks = final_blocks[:len(guaranteed_blocks)] + reranked_tail

    if not final_blocks:
        return {
            "query": req.query,
            "answer": "No encuentro evidencia suficiente en el contexto.",
            "contexts": [], "model": req.model
        }

    # ── Prompt y LLM ─────────────────────────────────────────────
    prompt = build_prompt(req.query, final_blocks, intent)

    try:
        text = await call_llm_with_fallback(req.model, prompt, req.temperature)
    except Exception as e:
        from fastapi.responses import JSONResponse
        logger.exception("Error en LLM")
        return JSONResponse(status_code=502, content={
            "query": req.query,
            "answer": f"Error al consultar el LLM: {type(e).__name__}: {e}",
            "contexts": final_blocks, "model": req.model
        })

    # ── Añadir fuentes ────────────────────────────────────────────
    sources, seen_src = [], set()
    for b in final_blocks:
        b = (b or "").strip()
        if b.startswith('[') and ']' in b:
            tag = b.split(']')[0] + ']'
            if tag not in seen_src:
                seen_src.add(tag)
                sources.append(tag)
    if sources:
        text += "\n\n---\n**Fuentes**\n" + "\n".join("- " + s for s in sources)

    return {"query": req.query, "answer": text, "contexts": final_blocks, "model": req.model}

# ---------- Retrieve lexical ----------
@app.post("/retrieve_lexical")
def retrieve_lexical(req: LexicalRequest):
    if not BM25:
        raise HTTPException(status_code=500, detail="BM25 no disponible.")
    qtok   = _tok(req.query)
    scores = BM25.get_scores(qtok)
    idx    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:req.top_k]
    blocks = [f"[{LEX_META[i].get('source','?')} p.{LEX_META[i].get('page','?')}] {LEX_CORPUS[i]}"
              for i in idx]
    return {"query": req.query, "top_k": req.top_k, "contexts": blocks}

# ---------- Retrieve hybrid ----------
@app.post("/retrieve_hybrid")
def retrieve_hybrid(req: HybridRetrieveRequest):
    emb  = get_embedder()
    qemb = emb.encode([req.query], normalize_embeddings=True).tolist()
    res  = collection.query(
        query_embeddings=qemb,
        n_results=max(req.top_k, 12),
        include=["documents", "metadatas", "distances"]
    )
    blocks_dense = build_context_blocks(res["documents"][0], res["metadatas"][0])

    if not BM25:
        raise HTTPException(status_code=500, detail="BM25 no disponible.")
    qtok        = _tok(req.query)
    scores      = BM25.get_scores(qtok)
    idx         = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max(req.top_k, 12)]
    blocks_lex  = [f"[{LEX_META[i].get('source','?')} p.{LEX_META[i].get('page','?')}] {LEX_CORPUS[i]}"
                   for i in idx]

    fused = rrf_fusion(blocks_dense, blocks_lex, k=60, top_k=req.top_k)
    fused = rerank_blocks(req.query, fused, keep_top_k=req.top_k)
    return {"query": req.query, "top_k": req.top_k, "contexts": fused}

# ---------- Rebuild BM25 ----------
@app.post("/rebuild_bm25")
def rebuild_bm25(batch: int = 1000, cap: Optional[int] = 8000):
    t0 = time.time()
    try:
        docs = build_bm25_from_chroma(batch=batch, cap=cap)
        return {"ok": BM25 is not None, "docs_indexed": docs,
                "seconds": round(time.time() - t0, 2), "cap": cap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo BM25: {e}")

@app.get("/bm25_status")
def bm25_status():
    return {"ready": BM25 is not None, "docs_indexed": len(LEX_CORPUS)}

# ---------- Rerank test ----------
@app.post("/rerank_test")
def rerank_test(req: RerankTestRequest):
    try:
        out = rerank_blocks(req.query, req.blocks, keep_top_k=req.top_k)
        return {"query": req.query, "top_k": req.top_k, "contexts": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-ranker error: {e}")

# ---------- Retrieve debug ----------
@app.post("/retrieve_debug")
def retrieve_debug(req: RetrieveRequest):
    queries = expand_query(req.query)
    emb     = get_embedder()
    dense   = []
    for q in queries:
        qemb = emb.encode([q], normalize_embeddings=True).tolist()
        res  = collection.query(
            query_embeddings=qemb, n_results=req.top_k,
            include=["documents", "metadatas", "distances"]
        )
        for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            dense.append({"query": q, "source": m.get("source"), "page": m.get("page"),
                          "phase": m.get("phase"), "distance": s, "text": d[:300]})
    lex = []
    if BM25:
        qtok   = _tok(req.query)
        scores = BM25.get_scores(qtok)
        idx    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:req.top_k]
        for i in idx:
            lex.append({"score": scores[i], "source": LEX_META[i].get("source"),
                        "page": LEX_META[i].get("page"),
                        "phase": LEX_META[i].get("phase"), "text": LEX_CORPUS[i][:300]})
    guaranteed = fetch_phase_blocks()
    return {"expanded_queries": queries, "dense_results": dense,
            "lexical_results": lex, "guaranteed_phase_blocks": len(guaranteed)}

# ================== OpenAI-compatible bridge (Open WebUI) ==================
@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": _get_gemini_models() + await _get_ollama_models()}

def _extract_user_query(messages: list) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"]
            return str(content)
    return ""

def _build_rag_system_prompt(rag_blocks: list) -> str:
    ctx = "\n\n".join(rag_blocks)
    return (
        "Eres un asistente experto en arqueología ibera. "
        "Responde usando principalmente el siguiente contexto documental. "
        "Si la pregunta no está cubierta, indícalo.\n\n"
        f"=== CONTEXTO ===\n{ctx}\n=== FIN CONTEXTO ==="
    )

def _openai_response(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def openai_chat_completions(req: OpenAIChatRequest):
    query = _extract_user_query(req.messages)
    if not query:
        raise HTTPException(status_code=400, detail="No se encontró mensaje de usuario.")

    intent  = detect_intent(query)
    queries = expand_query(query)

    # RAG retrieval (mismo pipeline que /answer)
    rag_blocks = []
    try:
        emb = get_embedder()

        guaranteed = []
        if intent == "historical_summary":
            guaranteed = fetch_phase_blocks(phases=[1, 2, 3, 4], chunks_per_phase=2)

        dense_all, seen = [], set()
        for q in queries[:5]:  # limitar queries en bridge para velocidad
            qemb = emb.encode([q], normalize_embeddings=True).tolist()
            res  = collection.query(query_embeddings=qemb, n_results=12,
                                    include=["documents", "metadatas", "distances"])
            for blk in build_context_blocks(res["documents"][0], res["metadatas"][0]):
                key = blk[:120]
                if key not in seen:
                    seen.add(key)
                    dense_all.append(blk)

        lex_bl = []
        if BM25:
            sc  = BM25.get_scores(_tok(query))
            idx = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:12]
            lex_bl = [f"[{LEX_META[i].get('source','?')} p.{LEX_META[i].get('page','?')}]\n{LEX_CORPUS[i]}"
                      for i in idx]

        fused = rrf_fusion(dense_all, lex_bl, k=60, top_k=req.top_k)
        fused = filter_blocks(fused, intent)

        seen2 = set()
        for blk in guaranteed + fused:
            key = blk[:120]
            if key not in seen2:
                seen2.add(key)
                rag_blocks.append(blk)

        rag_blocks = rerank_blocks(query, rag_blocks, keep_top_k=max(req.top_k, 10))
    except Exception as e:
        logger.warning(f"[bridge/rag] {e}")

    # Construir mensajes enriquecidos
    enriched = []
    has_sys  = any(m.get("role") == "system" for m in req.messages)
    if rag_blocks:
        rag_sys = _build_rag_system_prompt(rag_blocks)
        if has_sys:
            for m in req.messages:
                if m.get("role") == "system":
                    enriched.append({"role": "system", "content": m["content"] + "\n\n" + rag_sys})
                else:
                    enriched.append(m)
        else:
            enriched = [{"role": "system", "content": rag_sys}] + list(req.messages)
    else:
        enriched = list(req.messages)

    # Llamada al LLM
    try:
        if _is_ollama_model(req.model):
            answer_text = await _call_ollama_chat(req.model, enriched, req.temperature)
        else:
            prompt_parts = []
            for m in enriched:
                role    = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p["text"] for p in content
                                       if isinstance(p, dict) and p.get("type") == "text")
                prefix = {"system": "[Sistema]", "assistant": "[Asistente]"}.get(role, "[Usuario]")
                prompt_parts.append(f"{prefix}: {content}")
            answer_text = await call_gemini_chat_completions(
                req.model, "\n\n".join(prompt_parts), req.temperature)
    except Exception as e:
        logger.exception("[bridge] Error LLM")
        raise HTTPException(status_code=502, detail=f"Error LLM: {type(e).__name__}: {e}")

    # Fuentes
    if rag_blocks:
        sources, seen_s = [], set()
        for b in rag_blocks:
            if b.startswith('[') and ']' in b:
                tag = b.split(']')[0] + ']'
                if tag not in seen_s:
                    seen_s.add(tag)
                    sources.append(tag)
        if sources:
            answer_text += "\n\n---\n**Fuentes**\n" + "\n".join("- " + s for s in sources)

    return _openai_response(req.model, answer_text)

# ================== Watcher ==================
if ENABLE_WATCHER:
    from watcher import start_watcher
    threading.Thread(target=start_watcher, daemon=True).start()
