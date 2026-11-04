"""Retrieval algorithms shared by the HTTP endpoints.

The module is intentionally stateless. Runtime resources (Chroma, embedders,
BM25 data, reranking, and logging) remain owned by the composition root and are
passed explicitly so importing this module has no infrastructure side effects.
"""

import re
import string
from typing import Callable, Optional


def rrf_fusion(dense_blocks: list, lex_blocks: list, k: int = 60, top_k: int = 6) -> list:
    scores = {}
    for rank, block in enumerate(dense_blocks):
        scores[block] = scores.get(block, 0) + 1 / (k + rank)
    for rank, block in enumerate(lex_blocks):
        scores[block] = scores.get(block, 0) + 1 / (k + rank)
    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [b for b, _ in reranked[:top_k]]


def build_context_blocks(docs: list[str], metas: list[dict]) -> list[str]:
    blocks = []
    for d, m in zip(docs, metas):
        src = m.get("source", "desconocido")
        pg = m.get("page", "?")
        phase = m.get("phase")
        phase_tag = f" [Fase {phase}]" if phase else ""
        blocks.append(f"[{src} p.{pg}{phase_tag}]\n{d.strip()}")
    return blocks


def filter_blocks(blocks: list, intent: str) -> list:
    if intent not in ("definition", "historical_summary"):
        return blocks
    blacklist = ["bibliografía", "catálogo", "revista", "archivo español"]
    return [b for b in blocks if not any(x in b.lower() for x in blacklist)]


def expand_query(query: str) -> list:
    q = query.lower()
    expansions = [query]

    if "oppidum" in q or "oppida" in q:
        expansions += [
            "oppidum ibérico definición función urbana",
            "oppida iberos asentamiento fortificado",
        ]

    if any(
        k in q
        for k in [
            "fases",
            "proceso histórico",
            "resume",
            "etapas",
            "periodización",
            "principales fases",
        ]
    ):
        expansions += [
            "nucleación ibera nacimiento príncipes orientalizante",
            "primera fase iberos Jaén siglo VIII VII VI",
            "jefatura aristocracia ibera origen formación",
            "segunda fase iberos expansión clientelas siglo V",
            "consolidación aristocrática oppida formación estado ibero",
            "tercera fase iberos estados territoriales siglo IV III",
            "apogeo ibero ciudad estado oppida principales Jaén",
            "plena cultura ibérica Jaén campiña",
            "cuarta fase romanización hibridación iberos Roma",
            "segunda guerra púnica Baecula conquista romana Jaén",
            "municipalización Vespasiano iberos desaparición estados",
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
    if any(
        k in q
        for k in [
            "resume",
            "resumen",
            "principales fases",
            "fases del proceso",
            "proceso histórico",
            "etapas",
            "periodización",
        ]
    ):
        return "historical_summary"
    return "general"


def tokenize(text: str) -> list:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.findall(r"\w+", text, flags=re.UNICODE)


def build_bm25_from_collection(
    collection,
    bm25_factory: Callable,
    batch: int = 1000,
    cap: Optional[int] = 8000,
) -> tuple[list[str], list[dict], object]:
    corpus, metadata = [], []
    offset = 0
    while True:
        res = collection.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        if not docs:
            break
        for document, meta in zip(docs, metas):
            if document and len(document) >= 80:
                corpus.append(document)
                metadata.append(meta)
                if cap and len(corpus) >= cap:
                    break
        if cap and len(corpus) >= cap:
            break
        offset += batch
    bm25 = bm25_factory([tokenize(document) for document in corpus]) if corpus else None
    return corpus, metadata, bm25


def fetch_phase_blocks(
    collection,
    logger,
    phases: list = [1, 2, 3, 4],
    chunks_per_phase: int = 3,
) -> list:
    phase_blocks = []
    for phase in phases:
        try:
            res = collection.get(
                where={"phase": phase},
                limit=chunks_per_phase,
                include=["documents", "metadatas"],
            )
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])
            for document, meta in zip(docs, metas):
                src = meta.get("source", "?")
                pg = meta.get("page", "?")
                phase_blocks.append(
                    f"[{src} p.{pg} [Fase {phase}]]\n{document.strip()}"
                )
        except Exception as exc:
            logger.warning(f"[fetch_phase_blocks] fase {phase}: {exc}")
    return phase_blocks


def retrieve_dense(query, top_k, *, get_embedder, collection, rerank_blocks):
    emb = get_embedder()
    qemb = emb.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=qemb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    blocks = build_context_blocks(docs, metas)
    blocks = rerank_blocks(query, blocks, keep_top_k=top_k)
    return blocks, res.get("distances", [[]])[0]


def retrieve_answer_blocks(
    query,
    top_k,
    *,
    get_embedder,
    collection,
    bm25,
    lex_corpus,
    lex_meta,
    rerank_blocks,
    logger,
):
    intent = detect_intent(query)
    queries = expand_query(query)

    if intent == "historical_summary":
        dense_k, lex_k, final_k = 30, 30, 24
    elif intent == "definition":
        dense_k, lex_k, final_k = 8, 12, 5
    else:
        dense_k, lex_k, final_k = 12, 12, top_k

    guaranteed_blocks = []
    if intent == "historical_summary":
        guaranteed_blocks = fetch_phase_blocks(
            collection, logger, phases=[1, 2, 3, 4], chunks_per_phase=3
        )
        logger.info(f"[answer] Bloques garantizados por fase: {len(guaranteed_blocks)}")

    emb = get_embedder()
    dense_blocks_all = []
    seen_dense = set()
    for expanded_query in queries:
        qemb = emb.encode([expanded_query], normalize_embeddings=True).tolist()
        res = collection.query(
            query_embeddings=qemb,
            n_results=dense_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        for block in build_context_blocks(docs, metas):
            key = block[:120]
            if key not in seen_dense:
                seen_dense.add(key)
                dense_blocks_all.append(block)

    lex_blocks = []
    if bm25:
        scores = bm25.get_scores(tokenize(query))
        indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:lex_k]
        lex_blocks = [
            f"[{lex_meta[i].get('source','?')} p.{lex_meta[i].get('page','?')}]\n{lex_corpus[i]}"
            for i in indexes
        ]
    else:
        logger.warning("[answer] BM25 no disponible — solo dense retrieval")

    fused = rrf_fusion(dense_blocks_all, lex_blocks, k=60, top_k=final_k)
    fused = filter_blocks(fused, intent)

    seen_final = set()
    final_blocks = []
    for block in guaranteed_blocks:
        key = block[:120]
        if key not in seen_final:
            seen_final.add(key)
            final_blocks.append(block)
    for block in fused:
        key = block[:120]
        if key not in seen_final:
            seen_final.add(key)
            final_blocks.append(block)

    if len(final_blocks) > len(guaranteed_blocks):
        reranked_tail = rerank_blocks(
            query,
            final_blocks[len(guaranteed_blocks) :],
            keep_top_k=max(final_k - len(guaranteed_blocks), 4),
        )
        final_blocks = final_blocks[: len(guaranteed_blocks)] + reranked_tail

    return intent, final_blocks


def retrieve_lexical(query, top_k, *, bm25, lex_corpus, lex_meta):
    scores = bm25.get_scores(tokenize(query))
    indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        f"[{lex_meta[i].get('source','?')} p.{lex_meta[i].get('page','?')}] {lex_corpus[i]}"
        for i in indexes
    ]


def retrieve_hybrid(
    query,
    top_k,
    *,
    get_embedder,
    collection,
    bm25,
    lex_corpus,
    lex_meta,
    rerank_blocks,
    missing_bm25_error,
):
    emb = get_embedder()
    qemb = emb.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=qemb,
        n_results=max(top_k, 12),
        include=["documents", "metadatas", "distances"],
    )
    blocks_dense = build_context_blocks(res["documents"][0], res["metadatas"][0])
    if not bm25:
        raise missing_bm25_error()
    scores = bm25.get_scores(tokenize(query))
    indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        : max(top_k, 12)
    ]
    blocks_lex = [
        f"[{lex_meta[i].get('source','?')} p.{lex_meta[i].get('page','?')}] {lex_corpus[i]}"
        for i in indexes
    ]
    fused = rrf_fusion(blocks_dense, blocks_lex, k=60, top_k=top_k)
    return rerank_blocks(query, fused, keep_top_k=top_k)


def retrieve_debug_data(
    query,
    top_k,
    *,
    get_embedder,
    collection,
    bm25,
    lex_corpus,
    lex_meta,
    logger,
):
    queries = expand_query(query)
    emb = get_embedder()
    dense = []
    for expanded_query in queries:
        qemb = emb.encode([expanded_query], normalize_embeddings=True).tolist()
        res = collection.query(
            query_embeddings=qemb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for document, meta, score in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            dense.append(
                {
                    "query": expanded_query,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "phase": meta.get("phase"),
                    "distance": score,
                    "text": document[:300],
                }
            )
    lexical = []
    if bm25:
        scores = bm25.get_scores(tokenize(query))
        indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        for i in indexes:
            lexical.append(
                {
                    "score": scores[i],
                    "source": lex_meta[i].get("source"),
                    "page": lex_meta[i].get("page"),
                    "phase": lex_meta[i].get("phase"),
                    "text": lex_corpus[i][:300],
                }
            )
    guaranteed = fetch_phase_blocks(collection, logger)
    return {
        "expanded_queries": queries,
        "dense_results": dense,
        "lexical_results": lexical,
        "guaranteed_phase_blocks": len(guaranteed),
    }


def retrieve_chat_blocks(
    query,
    top_k,
    *,
    get_embedder,
    collection,
    bm25,
    lex_corpus,
    lex_meta,
    rerank_blocks,
    logger,
):
    intent = detect_intent(query)
    queries = expand_query(query)
    rag_blocks = []
    try:
        emb = get_embedder()
        guaranteed = []
        if intent == "historical_summary":
            guaranteed = fetch_phase_blocks(
                collection, logger, phases=[1, 2, 3, 4], chunks_per_phase=2
            )

        dense_all, seen = [], set()
        for expanded_query in queries[:5]:
            qemb = emb.encode([expanded_query], normalize_embeddings=True).tolist()
            res = collection.query(
                query_embeddings=qemb,
                n_results=12,
                include=["documents", "metadatas", "distances"],
            )
            for block in build_context_blocks(res["documents"][0], res["metadatas"][0]):
                key = block[:120]
                if key not in seen:
                    seen.add(key)
                    dense_all.append(block)

        lex_blocks = []
        if bm25:
            scores = bm25.get_scores(tokenize(query))
            indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:12]
            lex_blocks = [
                f"[{lex_meta[i].get('source','?')} p.{lex_meta[i].get('page','?')}]\n{lex_corpus[i]}"
                for i in indexes
            ]

        fused = rrf_fusion(dense_all, lex_blocks, k=60, top_k=top_k)
        fused = filter_blocks(fused, intent)

        seen_final = set()
        for block in guaranteed + fused:
            key = block[:120]
            if key not in seen_final:
                seen_final.add(key)
                rag_blocks.append(block)

        rag_blocks = rerank_blocks(query, rag_blocks, keep_top_k=max(top_k, 10))
    except Exception as exc:
        logger.warning(f"[bridge/rag] {exc}")
    return rag_blocks
