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

try:
    from backend import retrieval as retrieval_core
    from backend.llm_mode import LLMMode
    from backend.api.composition import build_grounded_image_composition
    from backend.api.chat import (
        ChatDependencies,
        ChatCompletionRequest,
        OpenAIChatRequest,
        build_rag_system_prompt as _build_rag_system_prompt,
        create_chat_router,
        extract_user_query as _extract_user_query,
        openai_response as _openai_response,
    )
except ImportError:  # Docker runs this module as top-level ``app``.
    import retrieval as retrieval_core
    from llm_mode import LLMMode
    from api.composition import build_grounded_image_composition
    from api.chat import (
        ChatDependencies,
        ChatCompletionRequest,
        OpenAIChatRequest,
        build_rag_system_prompt as _build_rag_system_prompt,
        create_chat_router,
        extract_user_query as _extract_user_query,
        openai_response as _openai_response,
    )

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
# DEFAULT_PROVIDER=ollama (todo local, por defecto) | gemini (de pago).
LLM_MODE             = LLMMode.from_environ(os.environ)
OLLAMA_DEFAULT_MODEL = LLM_MODE.ollama_model
DEFAULT_CHAT_MODEL   = LLM_MODE.default_chat_model
# Contexto de Ollama: el valor por defecto (4096) trunca en silencio el contexto RAG.
OLLAMA_NUM_CTX       = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
ENABLE_RERANKER      = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
RERANKER_MODEL       = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_TOP_K       = int(os.getenv("RERANKER_TOP_K", "6"))
RERANKER_MAXLEN      = int(os.getenv("RERANKER_MAXLEN", "512"))
ENABLE_WATCHER       = os.getenv("ENABLE_WATCHER", "false").lower() == "true"
# BM25 vive en memoria: se construye al arrancar y tras cada ingesta.
# BM25_CAP limita los fragmentos indexados (0 = sin límite).
ENABLE_BM25_AUTOBUILD = os.getenv("ENABLE_BM25_AUTOBUILD", "true").lower() == "true"
BM25_CAP             = int(os.getenv("BM25_CAP", "50000"))

if LLM_MODE.requires_gemini_key and not GEM_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY en el entorno (DEFAULT_PROVIDER=gemini)")
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
    return retrieval_core.rrf_fusion(dense_blocks, lex_blocks, k=k, top_k=top_k)

# ================== Modelos de datos ==================
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)

class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    model: str = Field(default=DEFAULT_CHAT_MODEL)
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
    return retrieval_core.build_context_blocks(docs, metas)

def filter_blocks(blocks: list, intent: str) -> list:
    return retrieval_core.filter_blocks(blocks, intent)

# ================== Query expansion e intent ==================
def expand_query(query: str) -> list:
    return retrieval_core.expand_query(query)

def detect_intent(query: str) -> str:
    return retrieval_core.detect_intent(query)

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
        "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX}
    }
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]

def _is_ollama_model(model_id: str) -> bool:
    # En modo local nunca se llama a Gemini, aunque el cliente pida un gemini-*.
    return LLM_MODE.routes_to_ollama(model_id)

def _get_gemini_models() -> list:
    if LLM_MODE.is_local:
        return []
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
    """Intenta Gemini; si falla por cuota/error, usa Ollama automáticamente.

    En modo local (DEFAULT_PROVIDER=ollama) va directo a Ollama.
    """
    if not _is_ollama_model(model):
        try:
            return await call_gemini_chat_completions(model, prompt, temperature)
        except RuntimeError as e:
            logger.warning(f"[LLM] Gemini falló: {str(e)[:120]}. Intentando Ollama...")

    # Fallback o modelo Ollama directo
    ollama_models = await _get_ollama_models()
    if not ollama_models:
        raise RuntimeError("Gemini falló y Ollama no tiene modelos disponibles.")

    preferred = [LLM_MODE.resolve_local_model(model), OLLAMA_DEFAULT_MODEL,
                 "llama3.1:8b", "llama3:8b", "phi3:mini", "gemma2:2b"]
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
    return retrieval_core.tokenize(s)

def build_bm25_from_chroma(batch: int = 1000, cap: Optional[int] = None) -> int:
    global LEX_CORPUS, LEX_META, BM25
    cap = (BM25_CAP or None) if cap is None else (cap or None)
    corpus, meta, index = retrieval_core.build_bm25_from_collection(
        collection, BM25Okapi, batch=batch, cap=cap
    )
    # Swap all three at once so concurrent queries never see a half-built index.
    LEX_CORPUS, LEX_META, BM25 = corpus, meta, index
    if cap and len(corpus) >= cap:
        logger.warning(
            f"[BM25] Se alcanzó BM25_CAP={cap}: hay fragmentos sin indexar léxicamente."
        )
    return len(corpus)


def _rebuild_bm25_safely(reason: str) -> None:
    """Rebuild BM25 without ever failing the caller (startup or ingest)."""
    t0 = time.time()
    try:
        docs = build_bm25_from_chroma()
        logger.info(f"[BM25] Índice listo ({reason}): {docs} fragmentos en {time.time() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"[BM25] No se pudo construir el índice ({reason}): {e}")

# ================== Recuperación por fases (garantizada) ==================
def fetch_phase_blocks(phases: list = [1, 2, 3, 4],
                       chunks_per_phase: int = 3) -> list:
    """
    Recupera chunks etiquetados con phase=N directamente desde Chroma.
    Garantiza cobertura de todas las fases en resúmenes históricos.
    """
    return retrieval_core.fetch_phase_blocks(
        collection, logger, phases=phases, chunks_per_phase=chunks_per_phase
    )

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
    if ENABLE_BM25_AUTOBUILD:
        _rebuild_bm25_safely(f"ingesta de {file.filename}")
    return {"ok": True, "pages": page_count, "chunks": len(documents),
            "bm25_docs_indexed": len(LEX_CORPUS)}

# ---------- Retrieve (denso) ----------
@app.post("/retrieve", operation_id="rag_retrieve")
def retrieve(req: RetrieveRequest):
    blocks, distances = retrieval_core.retrieve_dense(
        req.query,
        req.top_k,
        get_embedder=get_embedder,
        collection=collection,
        rerank_blocks=rerank_blocks,
    )
    return {
        "query": req.query, "top_k": req.top_k,
        "count": len(blocks), "contexts": blocks,
        "distances": distances
    }

# ---------- Answer (RAG completo) ----------
@app.post("/answer", operation_id="rag_answer")
async def answer(req: AnswerRequest):
    intent, final_blocks = retrieval_core.retrieve_answer_blocks(
        req.query,
        req.top_k,
        get_embedder=get_embedder,
        collection=collection,
        bm25=BM25,
        lex_corpus=LEX_CORPUS,
        lex_meta=LEX_META,
        rerank_blocks=rerank_blocks,
        logger=logger,
    )

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
    blocks = retrieval_core.retrieve_lexical(
        req.query,
        req.top_k,
        bm25=BM25,
        lex_corpus=LEX_CORPUS,
        lex_meta=LEX_META,
    )
    return {"query": req.query, "top_k": req.top_k, "contexts": blocks}

# ---------- Retrieve hybrid ----------
@app.post("/retrieve_hybrid")
def retrieve_hybrid(req: HybridRetrieveRequest):
    fused = retrieval_core.retrieve_hybrid(
        req.query,
        req.top_k,
        get_embedder=get_embedder,
        collection=collection,
        bm25=BM25,
        lex_corpus=LEX_CORPUS,
        lex_meta=LEX_META,
        rerank_blocks=rerank_blocks,
        missing_bm25_error=lambda: HTTPException(
            status_code=500, detail="BM25 no disponible."
        ),
    )
    return {"query": req.query, "top_k": req.top_k, "contexts": fused}

# ---------- Rebuild BM25 ----------
@app.post("/rebuild_bm25")
def rebuild_bm25(batch: int = 1000, cap: Optional[int] = None):
    t0 = time.time()
    try:
        docs = build_bm25_from_chroma(batch=batch, cap=cap)
        return {"ok": BM25 is not None, "docs_indexed": docs,
                "seconds": round(time.time() - t0, 2),
                "cap": (BM25_CAP or None) if cap is None else (cap or None)}
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
    return retrieval_core.retrieve_debug_data(
        req.query,
        req.top_k,
        get_embedder=get_embedder,
        collection=collection,
        bm25=BM25,
        lex_corpus=LEX_CORPUS,
        lex_meta=LEX_META,
        logger=logger,
    )

# ================== OpenAI-compatible bridge (Open WebUI) ==================
def _retrieve_chat_blocks(query: str, top_k: int) -> list:
    return retrieval_core.retrieve_chat_blocks(
        query,
        top_k,
        get_embedder=get_embedder,
        collection=collection,
        bm25=BM25,
        lex_corpus=LEX_CORPUS,
        lex_meta=LEX_META,
        rerank_blocks=rerank_blocks,
        logger=logger,
    )


# Grounded image generation is composed only when GROUNDED_IMAGE_ENABLED or
# GROUNDED_IMAGE_SHADOW is true; otherwise both values are None and the text
# route is unchanged.  Configuration is parsed strictly at startup.
image_orchestrator, image_metrics_router = build_grounded_image_composition(
    os.environ,
    retrieve_blocks=lambda query, top_k: _retrieve_chat_blocks(query, top_k),
    retrieval_model_id=EMBED_MODEL,
    api_key=GEM_KEY,
)
logger.info("[LLM] modo=%s modelo_por_defecto=%s", LLM_MODE.provider, DEFAULT_CHAT_MODEL)
if image_orchestrator is not None:
    logger.info("[grounded-image] composed (stage=%s)", image_orchestrator.metrics.stage.value)

chat_routes = create_chat_router(
    ChatDependencies(
        get_gemini_models=lambda: _get_gemini_models(),
        get_ollama_models=lambda: _get_ollama_models(),
        retrieve_blocks=lambda query, top_k: _retrieve_chat_blocks(query, top_k),
        is_ollama_model=lambda model: _is_ollama_model(model),
        call_ollama_chat=lambda model, messages, temperature: _call_ollama_chat(
            LLM_MODE.resolve_local_model(model), messages, temperature
        ),
        call_gemini_chat=lambda model, prompt, temperature: call_gemini_chat_completions(
            model, prompt, temperature
        ),
        logger=logger,
        image_orchestrator=image_orchestrator,
    ),
    default_model=DEFAULT_CHAT_MODEL,
    default_top_k=DEFAULT_TOP_K,
)
app.include_router(chat_routes.router)
if image_metrics_router is not None:
    app.include_router(image_metrics_router)

# Keep legacy imports used by scripts and tests while route ownership lives in
# backend.api.chat.
list_models = chat_routes.list_models
openai_chat_completions = chat_routes.openai_chat_completions

# ================== BM25 al arrancar ==================
# En segundo plano para no retrasar el arranque; mientras tanto la
# recuperación híbrida degrada a solo densa, como antes.
if ENABLE_BM25_AUTOBUILD:
    threading.Thread(
        target=_rebuild_bm25_safely, args=("arranque",), daemon=True
    ).start()

# ================== Watcher ==================
if ENABLE_WATCHER:
    from watcher import start_watcher
    threading.Thread(target=start_watcher, daemon=True).start()
