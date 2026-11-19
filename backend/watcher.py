"""Vigila la carpeta de documentos e ingiere cada PDF nuevo con POST /ingest.

Por defecto usa ``PollingObserver``: en Docker Desktop para Windows los
archivos copiados desde el host a un bind mount no generan eventos inotify,
así que el ``Observer`` nativo nunca los ve. El sondeo funciona en cualquier
sistema de archivos a cambio de revisar la carpeta cada pocos segundos.

La deduplicación vive en /ingest (por nombre y por huella), así que el
escaneo inicial y las copias repetidas no duplican fragmentos.
"""
import logging
import os
import threading
import time
from typing import Callable, Optional

import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

logger = logging.getLogger("rag-pro.watcher")

DOCS_DIR = os.getenv("DOCS_PATH", "/app/docs")
INGEST_ENDPOINT = os.getenv("WATCHER_INGEST_URL", "http://127.0.0.1:8088/ingest")
HEALTH_ENDPOINT = INGEST_ENDPOINT.rsplit("/", 1)[0] + "/health"


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() == "true"


USE_POLLING = _env_bool("WATCHER_POLLING", True)
POLL_INTERVAL = float(os.getenv("WATCHER_POLL_INTERVAL", "5"))
OCR = _env_bool("WATCHER_OCR", True)
SCAN_ON_START = _env_bool("WATCHER_SCAN_ON_START", True)
STABLE_CHECKS = 3          # tamaños iguales seguidos para dar la copia por terminada
STABLE_INTERVAL = 2.0      # segundos entre comprobaciones
STABLE_TIMEOUT = 600.0     # una copia de más de 10 min se da por fallida
INGEST_TIMEOUT = 1800      # OCR de un PDF grande puede tardar


def is_pdf(path: str) -> bool:
    name = os.path.basename(path)
    # Los temporales de copia/descarga (~$x.pdf, .x.pdf) se ignoran.
    return name.lower().endswith(".pdf") and not name.startswith(("~", "."))


def wait_until_stable(path: str, checks: int = STABLE_CHECKS,
                      interval: float = STABLE_INTERVAL,
                      timeout: float = STABLE_TIMEOUT,
                      sleep: Callable[[float], None] = time.sleep,
                      clock: Callable[[], float] = time.monotonic) -> bool:
    """True cuando el tamaño deja de cambiar; False si el archivo desaparece o expira."""
    deadline = clock() + timeout
    last, same = -1, 0
    while clock() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size > 0 and size == last:
            same += 1
            if same >= checks - 1:
                return True
        else:
            same = 0
        last = size
        sleep(interval)
    return False


class DocHandler(FileSystemEventHandler):
    def __init__(self, post: Callable = requests.post, ocr: bool = OCR,
                 stable: Callable[[str], bool] = wait_until_stable) -> None:
        super().__init__()
        self._post = post
        self._ocr = ocr
        self._stable = stable
        self._lock = threading.Lock()
        self._in_flight: set = set()

    def on_created(self, event):
        if not event.is_directory:
            self.ingest(event.src_path)

    def on_moved(self, event):
        # Navegadores y algunos copiadores escriben un temporal y lo renombran.
        if not event.is_directory:
            self.ingest(event.dest_path)

    def ingest(self, path: str) -> Optional[dict]:
        if not is_pdf(path):
            return None
        with self._lock:
            if path in self._in_flight:
                return None
            self._in_flight.add(path)
        try:
            if not self._stable(path):
                logger.warning("[WATCHER] %s no terminó de copiarse; se omite", path)
                return None
            name = os.path.basename(path)
            logger.info("[WATCHER] Ingiriendo %s (ocr=%s)", name, self._ocr)
            with open(path, "rb") as f:
                resp = self._post(
                    INGEST_ENDPOINT,
                    files={"file": (name, f, "application/pdf")},
                    data={"ocr": str(self._ocr).lower()},
                    timeout=INGEST_TIMEOUT,
                )
            if resp.status_code != 200:
                logger.error("[WATCHER] %s falló: HTTP %s %s",
                             name, resp.status_code, resp.text[:300])
                return None
            body = resp.json()
            if body.get("skipped"):
                logger.info("[WATCHER] %s ya estaba ingerido (%s)", name, body.get("reason"))
            else:
                logger.info("[WATCHER] %s ingerido: %s páginas, %s fragmentos",
                            name, body.get("pages"), body.get("chunks"))
            return body
        except Exception:
            logger.exception("[WATCHER] Error ingiriendo %s", path)
            return None
        finally:
            with self._lock:
                self._in_flight.discard(path)


def scan_existing(handler: DocHandler, docs_dir: str = DOCS_DIR) -> int:
    """Pasa por /ingest los PDF ya presentes (los añadidos con el backend parado)."""
    count = 0
    for name in sorted(os.listdir(docs_dir)):
        path = os.path.join(docs_dir, name)
        if os.path.isfile(path) and is_pdf(path):
            handler.ingest(path)
            count += 1
    return count


def wait_for_backend(get: Callable = requests.get, attempts: int = 60,
                     interval: float = 2.0,
                     sleep: Callable[[float], None] = time.sleep) -> bool:
    # El watcher arranca al importar app.py, antes de que uvicorn escuche.
    for _ in range(attempts):
        try:
            if get(HEALTH_ENDPOINT, timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        sleep(interval)
    return False


def start_watcher():
    handler = DocHandler()
    observer = (PollingObserver(timeout=POLL_INTERVAL) if USE_POLLING else Observer())
    observer.schedule(handler, DOCS_DIR, recursive=False)
    observer.start()
    logger.info("[WATCHER] Vigilando %s (%s, ocr=%s)", DOCS_DIR,
                f"sondeo cada {POLL_INTERVAL:g} s" if USE_POLLING else "inotify", OCR)
    if SCAN_ON_START:
        if wait_for_backend():
            n = scan_existing(handler)
            logger.info("[WATCHER] Escaneo inicial: %s PDF revisados", n)
        else:
            logger.error("[WATCHER] El backend no respondió; se omite el escaneo inicial")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
