"""BM25 is in memory, so it must be rebuilt on startup and after each ingest."""


class _Docs:
    def __init__(self, count: int) -> None:
        self._docs = [f"fragmento {i} " + "texto del oppidum ibero " * 5 for i in range(count)]

    def get(self, limit, offset, include):
        chunk = self._docs[offset : offset + limit]
        return {"documents": chunk, "metadatas": [{"source": "A.pdf", "page": 1}] * len(chunk)}


def test_default_cap_comes_from_environment_and_zero_means_unlimited(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "collection", _Docs(30))
    monkeypatch.setattr(app_module, "BM25_CAP", 10)
    assert app_module.build_bm25_from_chroma(batch=7) == 10
    monkeypatch.setattr(app_module, "BM25_CAP", 0)
    assert app_module.build_bm25_from_chroma(batch=7) == 30
    assert app_module.BM25 is not None and len(app_module.LEX_META) == 30


def test_explicit_cap_overrides_environment(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "collection", _Docs(30))
    monkeypatch.setattr(app_module, "BM25_CAP", 0)
    assert app_module.build_bm25_from_chroma(cap=5) == 5


def test_safe_rebuild_never_raises_and_keeps_previous_index(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "collection", _Docs(3))
    app_module.build_bm25_from_chroma()
    previous = app_module.BM25

    class Broken:
        def get(self, **kwargs):
            raise RuntimeError("chroma caído")

    monkeypatch.setattr(app_module, "collection", Broken())
    app_module._rebuild_bm25_safely("test")
    assert app_module.BM25 is previous
    assert len(app_module.LEX_CORPUS) == 3


def test_autobuild_is_on_by_default(app_module, monkeypatch, project_root):
    source = (project_root / "backend" / "app.py").read_text(encoding="utf-8")
    assert 'os.getenv("ENABLE_BM25_AUTOBUILD", "true")' in source
    assert '_rebuild_bm25_safely(f"ingesta de {file.filename}")' in source
