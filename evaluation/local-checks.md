# Local verification record

- Dataset validator: PASS; 9 CSV files and 21 knowledge documents validated; 1,000 orders, 1,783 order items, 244 logs, 52 questions, and 23 conversations.
- Offline unit tests: PASS; 3 tests covering current-section retrieval, order identity gating, and source identifier attachment.
- Python syntax compilation: PASS.
- Retrieval-only question set: 34 of 47 cases with knowledge-base gold evidence retrieved at least one expected section in the top 8 (72.3%). This measures retrieval coverage, not answer quality.
- LLM answer scoring and conversation suite: BLOCKED because no local OPENAI_API_KEY was available.
- Gnani STT/TTS smoke test: NOT RUN because no local GNANI_API_KEY or user voice recording was available.
- CI: not configured in the GitHub repository; checks were run locally.

The supplied dataset is excluded from this repository. No API key, .env file, or personal recording is included.
