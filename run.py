from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from zentromart.agent import SupportAgent
from zentromart.knowledge import KnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps capstone agent")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat", help="start a text support conversation")
    chat.add_argument("--data", default=os.getenv("ZENTROMART_DATA_DIR", "data"))
    batch = sub.add_parser("batch", help="transcribe audio, answer, and synthesize replies")
    batch.add_argument("audio_dir", type=Path)
    batch.add_argument("--output", type=Path, default=Path("outputs"))
    batch.add_argument("--data", default=os.getenv("ZENTROMART_DATA_DIR", "data"))
    evaluate = sub.add_parser("evaluate", help="evaluate practice cases")
    evaluate.add_argument("--mode", choices=("rag", "baseline"), default="rag")
    evaluate.add_argument("--suite", choices=("questions", "conversations"), default="questions")
    evaluate.add_argument("--data", default=os.getenv("ZENTROMART_DATA_DIR", "data"))
    args = parser.parse_args()

    data_path = Path(args.data)
    if (data_path / "kb").is_dir() and (data_path / "data").is_dir():
        dataset_root = data_path
    elif data_path.name.lower() == "data" and (data_path.parent / "kb").is_dir():
        dataset_root = data_path.parent
    else:
        parser.error(f"Dataset path must be the extracted dataset folder or its data subfolder: {data_path}")

    if args.command == "evaluate":
        from zentromart.evaluate import evaluate_conversations, evaluate_questions
        if args.suite == "conversations":
            report = evaluate_conversations(dataset_root)
        else:
            report = evaluate_questions(dataset_root, mode=args.mode)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    kb = KnowledgeBase.from_directory(dataset_root / "kb")
    agent = SupportAgent(kb, dataset_root / "data")
    if args.command == "chat":
        print("Hello, I'm ZentroMart's AI support assistant. How can I help?")
        while True:
            try:
                message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if message.lower() in {"quit", "exit"}:
                break
            result = agent.respond(message)
            print("Agent:", result.text)
            print("Sources:", ", ".join(result.sources) or "none")
        return 0

    from zentromart.voice import run_batch
    run_batch(agent, args.audio_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
