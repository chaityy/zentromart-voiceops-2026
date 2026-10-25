from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u0900-\u097f]+")
SECTION_RE = re.compile(r"^##\s+\[([^]]+)\]\s*(.*)$", re.M)
META_RE = re.compile(r"^([a-z_]+):\s*(.*)$", re.M)
PRECEDENCE = {"policy": 0, "agent_guide": 1, "help": 2, "faq": 3, "meta": 3, "marketing": 4}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class Chunk:
    source_id: str
    heading: str
    text: str
    path: str
    doc_type: str
    status: str
    effective_from: str

    @property
    def searchable(self) -> str:
        return f"{self.heading} {self.text}"


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = [c for c in chunks if c.status == "current"]
        self._docs = [Counter(tokens(c.searchable)) for c in self.chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / max(1, len(self._lengths))
        self._df = Counter(term for doc in self._docs for term in doc)

    @classmethod
    def from_directory(cls, directory: Path) -> "KnowledgeBase":
        chunks: list[Chunk] = []
        for path in sorted(directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            front = raw.split("---", 2)[1] if raw.startswith("---") and raw.count("---") >= 2 else ""
            meta = dict(META_RE.findall(front))
            sections = list(SECTION_RE.finditer(raw))
            for i, match in enumerate(sections):
                end = sections[i + 1].start() if i + 1 < len(sections) else len(raw)
                body = raw[match.end():end].strip()
                if body:
                    chunks.append(Chunk(
                        source_id=match.group(1).strip(), heading=match.group(2).strip(),
                        text=body, path=path.name, doc_type=meta.get("doc_type", "help"),
                        status=meta.get("status", "current"), effective_from=meta.get("effective_from", ""),
                    ))
        if not chunks:
            raise ValueError(f"No sectioned knowledge documents found in {directory}")
        return cls(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        qterms = set(tokens(query))
        if not qterms:
            return []
        ranked: list[tuple[Chunk, float]] = []
        n = len(self.chunks)
        for i, (chunk, doc) in enumerate(zip(self.chunks, self._docs)):
            dl = self._lengths[i]
            score = 0.0
            for term in qterms:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denom = tf + 1.2 * (1 - 0.75 + 0.75 * dl / max(1, self._avg_len))
                score += idf * tf * 2.2 / denom
            if score:
                ranked.append((chunk, score))
        ranked.sort(key=lambda pair: (pair[1], -PRECEDENCE.get(pair[0].doc_type, 9), pair[0].source_id), reverse=True)
        return ranked[:top_k]
