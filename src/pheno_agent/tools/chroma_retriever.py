"""
chroma_retriever.py — ChromaDB-based chunk retrieval tool.

Retrieves the most relevant note chunks for a patient from the existing
ChromaDB collection (built by ``rag_ingest.py``).

Uses BioClinicalBERT for query embedding and a multi-query strategy
covering different celiac clinical vocabularies.

This tool is OPTIONAL — the agentic system can work without ChromaDB
by reading EHR markdown files directly.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pheno_agent.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """A single retrieved chunk from ChromaDB."""
    text: str
    note_id: str = "unknown"
    note_datetime: str = ""
    note_type: str = ""
    distance: float = 0.0


# ---------------------------------------------------------------------------
# Multi-query strategy
# ---------------------------------------------------------------------------

RETRIEVAL_QUERIES = [
    "celiac disease diagnosis biopsy intraepithelial lymphocytes villous atrophy Marsh score duodenal pathology",
    "celiac sprue EGD endoscopy scalloping flat mucosa gluten enteropathy",
    "villous blunting crypt hyperplasia lamina propria lymphocytosis IEL",
    "biopsy-proven celiac confirmed celiac diagnosis duodenal biopsy",
]


# ---------------------------------------------------------------------------
# BioClinicalBERT embedding (lazy singleton)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None


def _load_embedding_model(model_name: Optional[str] = None):
    """Load BioClinicalBERT (cached after first call)."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = model_name or cfg.embed_model
    logger.info("Loading BioClinicalBERT: %s …", model_name)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()
        logger.info("BioClinicalBERT loaded on GPU.")
    else:
        logger.info("BioClinicalBERT loaded on CPU.")
    return _tokenizer, _model


def _embed_query(query: str, model_name: Optional[str] = None) -> List[float]:
    """Embed a query string using BioClinicalBERT with mean pooling."""
    import torch

    tokenizer, model = _load_embedding_model(model_name)
    device = next(model.parameters()).device

    encoded = tokenizer(
        [query], padding=True, truncation=True,
        max_length=512, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    token_embeddings = outputs.last_hidden_state
    summed = (token_embeddings * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
    mean_pooled = summed / counts

    return mean_pooled.squeeze(0).cpu().tolist()


# ---------------------------------------------------------------------------
# ChromaDB retrieval
# ---------------------------------------------------------------------------

def retrieve_relevant_chunks(
    grid: str,
    top_k: int = 20,
    db_path: Optional[Path] = None,
    collection_name: Optional[str] = None,
    embed_model: Optional[str] = None,
) -> List[ChunkResult]:
    """
    Retrieve the most relevant note chunks for a patient from ChromaDB.

    Uses multi-query retrieval (4 queries covering different clinical
    vocabularies) and deduplicates by text content.

    Parameters
    ----------
    grid : str
        Patient identifier.
    top_k : int
        Total number of unique chunks to return.
    db_path : Path, optional
        Override ChromaDB directory.
    collection_name : str, optional
        Override collection name.
    embed_model : str, optional
        Override embedding model.

    Returns
    -------
    list[ChunkResult]
        Retrieved chunks sorted by relevance.
        Returns empty list if ChromaDB is unavailable.
    """
    db_path = db_path or cfg.chroma_db_path
    collection_name = collection_name or cfg.notes_collection_name

    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.warning("ChromaDB not available (%s). Skipping chunk retrieval.", e)
        return []

    seen_texts = set()
    chunks: List[ChunkResult] = []
    per_query_k = max(top_k // len(RETRIEVAL_QUERIES), 5)

    for query_str in RETRIEVAL_QUERIES:
        query_embedding = _embed_query(query_str, embed_model)
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(per_query_k, collection.count()),
                where={"grid": grid},
                include=["documents", "distances", "metadatas"],
            )
        except Exception as e:
            logger.warning("ChromaDB query failed for grid %s: %s", grid, e)
            continue

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            if doc in seen_texts:
                continue
            seen_texts.add(doc)
            chunks.append(ChunkResult(
                text=doc,
                note_id=meta.get("note_id", "unknown"),
                note_datetime=meta.get("note_datetime", ""),
                note_type=meta.get("note_type", ""),
                distance=dist,
            ))

    logger.debug("Multi-query retrieval for %s: %d unique chunks.", grid, len(chunks))
    return chunks[:top_k]
