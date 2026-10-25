# ZentroMart VoiceOps 2026

A grounded customer-support agent for the VoiceOps capstone. The text agent retrieves current policy sections, keeps conversation context, and uses the supplied identity-checked order and ticket tools. The voice batch runner sends customer audio to Gnani STT and generates replies with Gnani TTS.

## Requirements

- Python 3.10+
- An OpenAI-compatible chat-completions API key for answer generation
- The separately supplied `Capstone_Project_Dataset_FTTT_2026` folder (not included here)
- A Gnani Vachana API key for audio runs

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

Add your own credentials to `.env`, then set the dataset path. Keep `.env` private and outside Git. `GNANI_API_KEY` is read by the official SDK from the environment. The dataset remains local; point to the extracted folder:

```powershell
$env:ZENTROMART_DATA_DIR = 'C:\path\to\Capstone_Project_Dataset_FTTT_2026'
```

On macOS/Linux, activate the venv and set the same environment variables in your shell.

## Run the text agent

```powershell
python run.py chat
```

The agent introduces itself as AI, uses retrieved document sections, asks for identity verification before looking up order details, and only reports a ticket after the mock tool confirms it was created.

## Required batch command

Provide a folder of short audio files (WAV/MP3/FLAC supported by your installed Gnani SDK account):

```powershell
python run.py batch path\to\customer-audio --output outputs
```

The runner saves a `.txt` transcript and reply, a `.wav` spoken response, and a `.json` record with retrieved section IDs for every file. It uses Gnani only for STT and TTS. No other speech service is used.

## Evaluation and tests

```powershell
python run.py evaluate --mode rag
python run.py evaluate --mode baseline
python run.py evaluate --suite conversations
python -m unittest discover -s tests -v
```

The evaluator reads the supplied practice questions and multi-turn conversations from the local dataset folder. It reports answer checks, retrieval evidence recall, conversation checks, and tool-call expectations. It does not read the reference-answer column to generate answers. Live LLM/Gnani runs require your own keys.

## Safety and data

- The dataset is synthetic but is intentionally excluded from this public repository. Never commit the dataset, `.env`, audio recordings, or API keys.
- Do not use real customer data or recordings. Record only your own voice for a voice smoke test.
- The mock order tool returns no order facts before a match against the registered phone suffix or delivery pincode. It permits one retry per conversation in the agent flow.
- Retrieved section identifiers are included in each answer record for auditability.

## Voice smoke test

After setting `GNANI_API_KEY`, create a short recording of your own voice saying a simple policy question, then run the batch command. Confirm the generated transcript, reply text, audio file, and JSON sources. The submitted batch runner supports multiple recordings in one directory.
