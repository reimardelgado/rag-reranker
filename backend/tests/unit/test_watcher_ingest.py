"""El watcher debe ver PDFs copiados desde Windows y /ingest no debe duplicarlos."""
import hashlib
import types

import pytest

from backend import watcher


# ---------------------------------------------------------------- watcher

class _Resp:
    def __init__(self, body, status=200):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def _handler(calls, body=None):
    def post(url, files, data, timeout):
        name, fh, _ = files["file"]
        calls.append({"url": url, "name": name, "data": data, "bytes": fh.read()})
        return _Resp(body or {"ok": True, "skipped": False, "pages": 1, "chunks": 2})

    return watcher.DocHandler(post=post, ocr=True, stable=lambda path: True)


@pytest.mark.parametrize("name, expected", [
    ("Baecula.pdf", True), ("INFORME.PDF", True), ("notas.txt", False),
    ("~$borrador.pdf", False), (".parcial.pdf", False),
])
def test_only_real_pdfs_are_ingested(name, expected):
    assert watcher.is_pdf(f"/app/docs/{name}") is expected


def test_polling_is_the_default_because_bind_mounts_lack_inotify():
    assert watcher.USE_POLLING is True
    assert watcher.OCR is True


def test_created_and_renamed_pdfs_are_posted_with_ocr(tmp_path):
    pdf = tmp_path / "Iliturgi.pdf"
    pdf.write_bytes(b"%PDF-1.4 contenido")
    calls = []
    handler = _handler(calls)

    handler.on_created(types.SimpleNamespace(is_directory=False, src_path=str(pdf)))
    handler.on_moved(types.SimpleNamespace(is_directory=False, src_path=str(tmp_path / "x.tmp"),
                                           dest_path=str(pdf)))
    handler.on_created(types.SimpleNamespace(is_directory=False, src_path=str(tmp_path / "a.txt")))

    assert [c["name"] for c in calls] == ["Iliturgi.pdf", "Iliturgi.pdf"]
    assert calls[0]["data"] == {"ocr": "true"}
    assert calls[0]["bytes"] == b"%PDF-1.4 contenido"
    assert calls[0]["url"].endswith("/ingest")


def test_unfinished_copy_is_not_posted(tmp_path):
    pdf = tmp_path / "grande.pdf"
    pdf.write_bytes(b"%PDF")
    calls = []
    handler = watcher.DocHandler(post=lambda *a, **k: calls.append(1), stable=lambda p: False)
    assert handler.ingest(str(pdf)) is None
    assert calls == []


def test_http_errors_are_logged_not_raised(tmp_path):
    pdf = tmp_path / "roto.pdf"
    pdf.write_bytes(b"%PDF")
    handler = watcher.DocHandler(post=lambda *a, **k: _Resp({"detail": "PDF inválido"}, 400),
                                 stable=lambda p: True)
    assert handler.ingest(str(pdf)) is None


def test_wait_until_stable_requires_unchanged_size(tmp_path):
    pdf = tmp_path / "copiando.pdf"
    sizes = iter([10, 20, 20, 20])
    pdf.write_bytes(b"x")
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds
        pdf.write_bytes(b"x" * next(sizes, 20))

    pdf.write_bytes(b"x" * 5)
    assert watcher.wait_until_stable(str(pdf), checks=3, interval=1, timeout=60,
                                     sleep=sleep, clock=lambda: now[0])
    assert now[0] >= 3


def test_wait_until_stable_gives_up_on_missing_or_endless_copy(tmp_path):
    assert not watcher.wait_until_stable(str(tmp_path / "no.pdf"), sleep=lambda s: None)

    pdf = tmp_path / "crece.pdf"
    pdf.write_bytes(b"x")
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds
        pdf.write_bytes(pdf.read_bytes() + b"x")

    assert not watcher.wait_until_stable(str(pdf), interval=1, timeout=10,
                                         sleep=sleep, clock=lambda: now[0])


def test_scan_existing_posts_every_pdf_in_folder(tmp_path):
    for name in ("b.pdf", "a.pdf", "leeme.txt"):
        (tmp_path / name).write_bytes(b"%PDF")
    (tmp_path / "sub.pdf").mkdir()
    calls = []
    assert watcher.scan_existing(_handler(calls), str(tmp_path)) == 2
    assert [c["name"] for c in calls] == ["a.pdf", "b.pdf"]


def test_wait_for_backend_retries_until_health_answers():
    answers = iter([OSError, 503, 200])

    def get(url, timeout):
        a = next(answers)
        if a is OSError:
            raise watcher.requests.ConnectionError("aún no")
        return _Resp({}, a)

    assert watcher.wait_for_backend(get=get, attempts=5, sleep=lambda s: None)
    assert not watcher.wait_for_backend(get=lambda u, timeout: _Resp({}, 503),
                                        attempts=2, sleep=lambda s: None)


# ---------------------------------------------------------------- /ingest

class _Collection:
    def __init__(self, existing=()):
        self.rows = [dict(m) for m in existing]
        self.added = []
        self.deleted = []

    def get(self, where, include, limit=None):
        (key, value), = where.items()
        hits = [(str(i), m) for i, m in enumerate(self.rows) if m.get(key) == value][:limit]
        return {"ids": [i for i, _ in hits], "metadatas": [m for _, m in hits]}

    def update(self, ids, metadatas):
        for _id, meta in zip(ids, metadatas):
            self.rows[int(_id)] = meta

    def add(self, documents, metadatas, embeddings, ids):
        self.added.append({"ids": ids, "metadatas": metadatas})

    def delete(self, where):
        self.deleted.append(where)


class _Upload:
    def __init__(self, name, data):
        self.filename = name
        self._data = data

    async def read(self):
        return self._data


def _wire(app_module, monkeypatch, collection):
    page = types.SimpleNamespace(get_text=lambda: "Texto del oppidum de Giribaile.")
    fake_doc = type("Doc", (), {"__iter__": lambda self: iter([page]),
                                "__len__": lambda self: 1, "close": lambda self: None})()
    monkeypatch.setattr(app_module, "collection", collection)
    monkeypatch.setattr(app_module.fitz, "open", lambda **kw: fake_doc)
    monkeypatch.setattr(app_module, "ENABLE_BM25_AUTOBUILD", False)
    monkeypatch.setattr(app_module, "detect_phase", lambda chunk: None)
    monkeypatch.setattr(app_module, "_chunk", lambda text: [text])
    monkeypatch.setattr(app_module, "_clean_text", lambda text: text)

    class _Vecs(list):
        def tolist(self):
            return list(self)

    monkeypatch.setattr(app_module, "get_embedder", lambda: types.SimpleNamespace(
        encode=lambda docs, normalize_embeddings: _Vecs([[0.0]] * len(docs))))


async def test_new_pdf_gets_deterministic_ids_and_content_hash(app_module, monkeypatch):
    col = _Collection()
    _wire(app_module, monkeypatch, col)
    data = b"%PDF giribaile"
    out = await app_module.ingest_pdf(_Upload("Giribaile.pdf", data), ocr=False, force=False)

    sha = hashlib.sha256(data).hexdigest()
    assert out["skipped"] is False and out["chunks"] == 1
    assert col.added[0]["ids"] == [f"{sha[:16]}-0"]
    assert col.added[0]["metadatas"][0]["doc_sha256"] == sha
    assert col.added[0]["metadatas"][0]["source"] == "Giribaile.pdf"


@pytest.mark.parametrize("existing, reason", [
    # Ingerido antes con ingest_ocr_batch.py: solo tiene source.
    ({"source": "Giribaile.pdf"}, "mismo nombre"),
    # Mismo PDF con otro nombre.
    ({"source": "copia.pdf", "doc_sha256": hashlib.sha256(b"%PDF giribaile").hexdigest()},
     "mismo contenido"),
])
async def test_already_ingested_pdf_is_skipped(app_module, monkeypatch, existing, reason):
    col = _Collection([existing])
    _wire(app_module, monkeypatch, col)
    out = await app_module.ingest_pdf(_Upload("Giribaile.pdf", b"%PDF giribaile"),
                                      ocr=False, force=False)
    assert out == {"ok": True, "skipped": True, "reason": reason, "chunks": 0}
    assert col.added == []


async def test_force_replaces_previous_chunks(app_module, monkeypatch):
    col = _Collection([{"source": "Giribaile.pdf"}])
    _wire(app_module, monkeypatch, col)
    out = await app_module.ingest_pdf(_Upload("Giribaile.pdf", b"%PDF v2"), ocr=False, force=True)
    assert col.deleted == [{"source": "Giribaile.pdf"}]
    assert out["skipped"] is False and len(col.added) == 1


async def test_old_chunks_get_the_hash_so_renamed_copies_are_caught(app_module, monkeypatch):
    data = b"%PDF giribaile"
    sha = hashlib.sha256(data).hexdigest()
    col = _Collection([{"source": "Giribaile.pdf", "page": 1}, {"source": "Giribaile.pdf", "page": 2},
                       {"source": "Otro.pdf", "page": 1, "doc_sha256": "abc"}])
    _wire(app_module, monkeypatch, col)

    await app_module.ingest_pdf(_Upload("Giribaile.pdf", data), ocr=False, force=False)
    assert [r.get("doc_sha256") for r in col.rows] == [sha, sha, "abc"]
    assert col.rows[1]["page"] == 2

    out = await app_module.ingest_pdf(_Upload("Giribaile (copia).pdf", data), ocr=False, force=False)
    assert out["reason"] == "mismo contenido" and col.added == []


async def test_backfill_never_overwrites_an_existing_hash(app_module, monkeypatch):
    col = _Collection([{"source": "Giribaile.pdf", "doc_sha256": "version-anterior"}])
    _wire(app_module, monkeypatch, col)
    await app_module.ingest_pdf(_Upload("Giribaile.pdf", b"%PDF v2"), ocr=False, force=False)
    assert col.rows[0]["doc_sha256"] == "version-anterior"
