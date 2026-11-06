# watch_and_ingest.py
# Monitorea C:\RAG_Project\backend\docs y lanza ingestas incrementales sin duplicados.
import hashlib, json, os, queue, signal, subprocess, sys, threading, time
from pathlib import Path
from config import settings
import os


try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("[ERROR] Falta 'watchdog'. Instala: pip install watchdog")
    sys.exit(1)

BACKEND  = Path("/app")
DOCS     = Path(settings.DOCS_PATH)
STATE    = BACKEND / "ingest_state.json"
PY       = "python"
INGEST   = str(BACKEND / "ingest_ocr_batch.py")


# Backend FastAPI para trigger de BM25
API_BASE = "http://127.0.0.1:8088"
REBUILD  = f"{API_BASE}/rebuild_bm25?batch=1000"  # el tope lo fija BM25_CAP en el backend

EXTS = {".pdf", ".PDF"}

def file_key(p: Path) -> str:
    try:
        st = p.stat()
    except FileNotFoundError:
        return ""
    return f"{st.st_size}-{int(st.st_mtime)}"

def ensure_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"files": {}, "last_rebuild": 0}

def persist_state(s):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)

def run_ingest(pdf_path: Path) -> int:
    # Llama a tu script con deduplicación de chunks y upsert (ver parche más abajo)
    args = [
        PY, INGEST, str(pdf_path),
        "--dedup-chunks",
        "--chroma-host", settings.CHROMA_HOST,
        "--chroma-port", str(settings.CHROMA_PORT),
        "--collection", settings.COLLECTION,
        "--embed-model", settings.EMBED_MODEL,
        "--max-chars", "1200", "--overlap", "150",
        # OCR opcional si lo necesitas por defecto:
        # "--enable-ocr", "--ocr-lang", "spa+eng", "--ocr-psm", "3", "--ocr-oem", "1"
    ]
    print(f"[INGEST] {pdf_path.name}")
    return subprocess.call(args, cwd=str(BACKEND), shell=False)

def rebuild_bm25():
    try:
        import urllib.request
        with urllib.request.urlopen(REBUILD, timeout=10) as r:
            print("[BM25] rebuild trigger ->", r.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        print("[BM25] rebuild fallo:", e)

class DebounceHandler(FileSystemEventHandler):
    def __init__(self, work_q: queue.Queue):
        self.work_q = work_q
    def on_any_event(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix in EXTS:
            self.work_q.put(p)

def worker_loop():
    state = ensure_state()
    changed = set()
    last_trigger = 0
    while True:
        try:
            p = work_q.get(timeout=2)
        except queue.Empty:
            # cada 30s, si hubo cambios ingestado, dispara rebuild
            if changed and (time.time() - last_trigger > 30):
                rebuild_bm25()
                last_trigger = time.time()
                changed.clear()
            continue

        if not p.exists():
            continue
        key = file_key(p)
        if not key:
            continue

        rel = str(p.relative_to(DOCS))
        prev = state["files"].get(rel)
        if prev == key:
            continue  # sin cambios

        # pequeño debounce: espera a que deje de crecer
        size1 = p.stat().st_size
        time.sleep(1.0)
        size2 = p.stat().st_size
        if size2 != size1:
            # archivo aún copiándose
            time.sleep(2.0)

        rc = run_ingest(p)
        if rc == 0:
            state["files"][rel] = file_key(p)
            persist_state(state)
            changed.add(rel)
        else:
            print(f"[ERROR] Ingesta falló para {rel} (rc={rc})")

def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    handler = DebounceHandler(work_q)
    observer.schedule(handler, str(DOCS), recursive=True)
    observer.start()
    print(f"[WATCH] Vigilando: {DOCS}")
    try:
        worker_loop()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    work_q = queue.Queue()
    main()