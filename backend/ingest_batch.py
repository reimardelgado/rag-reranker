# ingest_batch.py
import os, sys, uuid, re, gc
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import settings

COLLECTION = settings.COLLECTION

client = chromadb.HttpClient(
    host=settings.CHROMA_HOST,
    port=settings.CHROMA_PORT,
    settings=Settings(allow_reset=True)
)

# Tamaños de lote

# Tamaños de lote (ajústalos si hace falta)
MAX_CHARS   = 1200
OVERLAP     = 150
BATCH       = 128    # nº de chunks por lote de embeddings/add

def info(msg: str): print(f"[INFO] {msg}")
def warn(msg: str): print(f"[WARN] {msg}")
def err(msg: str):  print(f"[ERROR] {msg}")

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def chunk(text: str, max_chars=1200, overlap=150):
    text = text.strip()
    n = len(text)
    chunks = []
    if n == 0:
        return chunks

    start = 0
    while start < n:
        end = min(start + max_chars, n)
        part = text[start:end].strip()

        # filtra trozos muy cortos
        if len(part) > 50:
            chunks.append(part)

        # si llegamos al final, salimos (evita bucle)
        if end == n:
            break

        # calcula el siguiente inicio con solape
        next_start = end - overlap

        # GARANTIZA progreso (evita que next_start <= start)
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

def load_pdf_pages(path: str):
    import pymupdf as fitz  # PyMuPDF
    with fitz.open(path) as doc:
        total = len(doc)
        info(f"Total de páginas: {total}")
        for i in range(total):
            try:
                page = doc[i]
                raw  = page.get_text("text") or ""
                yield i+1, clean_text(raw)
            except Exception as e:
                warn(f"Página {i+1}: error PyMuPDF ({e}). Continúo...")
                yield i+1, ""

def add_batch_to_chroma(collection, embedder, docs: List[str], metas: List[dict], ids: List[str]):
    if not docs:
        return 0
    vecs = embedder.encode(docs, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
    # Liberamos memoria de estos lotes
    added = len(docs)
    docs.clear(); metas.clear(); ids.clear()
    gc.collect()
    return added

def main(pdf_path: str):
    p = Path(pdf_path)
    if not p.exists() or not p.is_file():
        err("Ruta al PDF inválida.")
        return

    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    info(f"Colección OK: {COLLECTION}")

    # Modelo embeddings
    info("Cargando modelo de embeddings...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    fname = p.name
    batch_docs, batch_metas, batch_ids = [], [], []

    pages_total = 0
    pages_with_text = 0
    chunks_total = 0
    added_total  = 0

    # Proceso página a página
    for page_num, page_text in load_pdf_pages(str(p)):
        pages_total += 1
        if page_text.strip():
            pages_with_text += 1
            for c in chunk(page_text):
                batch_docs.append(c)
                batch_metas.append({"source": fname, "page": page_num})
                batch_ids.append(str(uuid.uuid4()))
                chunks_total += 1

                # Si llenamos un lote -> embeddings + add
                if len(batch_docs) >= BATCH:
                    try:
                        added = add_batch_to_chroma(collection, embedder, batch_docs, batch_metas, batch_ids)
                        added_total += added
                        info(f"Añadidos {added} chunks (acumulado: {added_total}).")
                    except Exception as e:
                        warn(f"Fallo al subir lote (pág.{page_num}). Sigo. Detalle: {e}")
                        # Limpiar el lote que falló para seguir
                        batch_docs.clear(); batch_metas.clear(); batch_ids.clear()
                        gc.collect()

    # Último lote si quedó algo pendiente
    if batch_docs:
        try:
            added = add_batch_to_chroma(collection, embedder, batch_docs, batch_metas, batch_ids)
            added_total += added
            info(f"Añadidos {added} chunks (acumulado: {added_total}).")
        except Exception as e:
            warn(f"Fallo al subir último lote. Detalle: {e}")

    info(f"Páginas totales: {pages_total}, con texto: {pages_with_text}")
    if added_total == 0:
        warn("No se añadieron chunks. Posible PDF escaneado o errores en todas las páginas.")
    else:
        info(f"Ingestados {added_total} chunks desde {fname}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ingest_batch.py <ruta_al_pdf>")
        sys.exit(1)
    main(sys.argv[1])