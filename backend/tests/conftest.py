import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "backend" / "app.py"


class _Collection:
    def get(self, **kwargs):
        return {"documents": [], "metadatas": []}

    def query(self, **kwargs):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _install_import_fakes(monkeypatch):
    """Keep importing the legacy composition root offline and lightweight."""
    chromadb = _module(
        "chromadb",
        HttpClient=lambda **kwargs: types.SimpleNamespace(
            get_or_create_collection=lambda **collection_kwargs: _Collection()
        ),
    )
    chromadb_config = _module("chromadb.config", Settings=lambda **kwargs: kwargs)
    chromadb.config = chromadb_config

    torch = _module(
        "torch",
        device=lambda name: name,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        no_grad=MagicMock(),
    )
    transformers = _module(
        "transformers",
        AutoTokenizer=MagicMock(),
        AutoModelForSequenceClassification=MagicMock(),
    )
    rank_bm25 = _module("rank_bm25", BM25Okapi=MagicMock())
    pymupdf = _module("pymupdf", Matrix=MagicMock(), open=MagicMock())
    pil = _module("PIL", Image=MagicMock())

    fakes = {
        "chromadb": chromadb,
        "chromadb.config": chromadb_config,
        "torch": torch,
        "transformers": transformers,
        "rank_bm25": rank_bm25,
        "pymupdf": pymupdf,
        "PIL": pil,
        "pytesseract": _module("pytesseract", image_to_string=MagicMock()),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setenv("ENABLE_WATCHER", "false")
    _install_import_fakes(monkeypatch)

    module_name = "rag_backend_contract_app"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project_root():
    return ROOT
