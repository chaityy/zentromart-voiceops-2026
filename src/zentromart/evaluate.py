from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .agent import LLMError, SupportAgent
from .knowledge import KnowledgeBase


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _gold_sections(row: dict[str, str]) -> set[str]:
    # Evidence strings may include non-KB sources (for example products.csv:SKU).
    return {part.strip() for part in row.get("gold_evidence", "").split(";") if part.strip() and ".csv:" not in part}


def retrieval_metrics(kb: KnowledgeBase, questions: list[dict[str, str]], k: int = 8) -> dict[str, Any]:
    scored, applicable, hits = 0, 0, 0
    per_case = []
    for row in questions:
        gold = _gold_sections(row)
        found = {chunk.source_id for chunk, _ in kb.search(row.get("question", ""), k)}
        hit = bool(gold & found) if gold else None
        if hit is not None:
            applicable += 1
            hits += int(hit)
        scored += 1
        per_case.append({"id": row.get("id", ""), "gold_sections": sorted(gold), "retrieved_sections": sorted(found), "gold_hit": hit})
    return {"cases": scored, "cases_with_kb_gold": applicable, "gold_section_recall_at_k": hits / applicable if applicable else None,
            "k": k, "per_case": per_case}


def _contains_any(answer: str, requirement: str) -> bool:
    if not requirement.strip():
        return True
    alternatives = [alt.strip().lower() for alt in requirement.split("/") if alt.strip()]
    return any(alt in answer.lower() for alt in alternatives)


def _contains_all_groups(answer: str, requirements: str) -> bool:
    return all(_contains_any(answer, group) for group in requirements.split(";") if group.strip())


def evaluate_questions(dataset_root: Path, *, mode: str = "rag", client: Any | None = None) -> dict[str, Any]:
    kb = KnowledgeBase.from_directory(dataset_root / "kb")
    questions = _read_rows(dataset_root / "practice" / "questions.csv")
    retrieval = retrieval_metrics(kb, questions)
    try:
        agent = SupportAgent(kb, dataset_root / "data", client=client)
    except LLMError as exc:
        return {"status": "blocked", "mode": mode, "reason": str(exc), "retrieval_only": retrieval}
    checks, details = [], []
    for row in questions:
        agent.history.clear()
        agent.failed_verifications.clear()
        result = agent.respond(row.get("question", ""), use_retrieval=(mode == "rag"))
        answer = result.text
        include_ok = _contains_all_groups(answer, row.get("must_include", ""))
        exclude_ok = not any(term.strip().lower() in answer.lower() for term in row.get("must_not_include", "").split(";") if term.strip())
        passed = include_ok and exclude_ok
        checks.append(passed)
        details.append({"id": row.get("id", ""), "passed": passed, "must_include_ok": include_ok,
                        "must_not_include_ok": exclude_ok, "sources": result.sources, "answer": answer})
    return {"status": "complete", "mode": mode, "questions": len(details), "rubric_checks_passed": sum(checks),
            "rubric_check_rate": sum(checks) / len(checks) if checks else None, "retrieval_only": retrieval, "details": details}


def evaluate_conversations(dataset_root: Path, *, client: Any | None = None) -> dict[str, Any]:
    kb = KnowledgeBase.from_directory(dataset_root / "kb")
    rows = _read_rows(dataset_root / "practice" / "conversations.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["conversation_id"], []).append(row)
    try:
        agent = SupportAgent(kb, dataset_root / "data", client=client)
    except LLMError as exc:
        return {"status": "blocked", "reason": str(exc), "conversations": len(grouped), "turns": len(rows)}
    details = []
    for conversation_id, turns in grouped.items():
        agent.history.clear()
        agent.failed_verifications.clear()
        for row in sorted(turns, key=lambda item: int(item["turn"])):
            result = agent.respond(row["customer_message"])
            expected_names = set(re.findall(r"([a-z_]+)\(", row.get("expected_tool_calls", "")))
            actual_names = set(result.tool_calls)
            must_ok = _contains_all_groups(result.text, row.get("must_include", ""))
            forbidden = [term.strip().lower() for term in row.get("must_not_include", "").split(";") if term.strip()]
            excludes_ok = not any(term in result.text.lower() for term in forbidden)
            tools_ok = expected_names.issubset(actual_names)
            details.append({"conversation_id": conversation_id, "turn": int(row["turn"]),
                            "must_include_ok": must_ok, "must_not_include_ok": excludes_ok,
                            "expected_tool_names": sorted(expected_names), "actual_tool_names": result.tool_calls,
                            "tools_ok": tools_ok, "passed": must_ok and excludes_ok and tools_ok,
                            "sources": result.sources, "answer": result.text})
    return {"status": "complete", "conversations": len(grouped), "turns": len(details),
            "checks_passed": sum(item["passed"] for item in details),
            "check_rate": sum(item["passed"] for item in details) / len(details) if details else None,
            "details": details}


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
