# CoolBreeze voice agent

Demo video: (add link)

Phone receptionist for an HVAC and plumbing shop. It listens, talks back, checks free slots in SQLite, and books a visit. It does **not** give appliance-repair instructions (washing machines, fridges, and so on).

This project has been tested from the **browser client** at http://localhost:8000 and with `scripts/simulate_call.py`. **Twilio is optional and has not been verified with a real phone number.**

```
caller  →  Deepgram live STT (mulaw 8 kHz)
        →  LLM (Gemini by default; any OpenAI-compatible API works)
        →  sentence-split streaming → ElevenLabs TTS (ulaw_8000)
        →  caller (browser WebSocket or Twilio Media Streams)
```

## Layout

```
voice-agent/                     run every command from this folder
  main.py                        FastAPI: GET /  POST /incoming-call  WS /media
  static/web_client.html         Browser mic/speaker client
  scripts/check_services.py      Ping Gemini, ElevenLabs, Deepgram
  scripts/simulate_call.py       Fake a call with TTS audio (no Twilio)
  scripts/latency_report.py      Summarize data/latency.csv
  tests/test_agent.py            Offline tests (no keys, no mic)
  data/                          Runtime only (gitignored except .gitkeep)
  .env                           Your keys (never commit)
  .env.example                   Key names and placeholders
  requirements.txt               Runtime Python packages
  pytest.ini                     Finds tests in tests/
```

## Quick start (Windows PowerShell)

From this folder:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
# fill DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, GEMINI_API_KEY
.venv\Scripts\python.exe scripts\check_services.py
.venv\Scripts\python.exe -m uvicorn main:app --port 8000
```

Open http://localhost:8000 in Chrome, wear **headphones**, click **Start call**, wait for the greeting, then talk.

Examples that the agent is built to handle:

- “My air conditioner stopped cooling.”
- “There’s a leak under the kitchen sink.”
- “I need a technician tomorrow afternoon.”

It collects name, phone, address, and a slot (`09:00`, `11:00`, `14:00`, `16:00`), reads them back, then books. Unrelated appliance repair is declined in one sentence.

## Keys

Required in `.env`: `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `GEMINI_API_KEY`.

Optional: `GEMINI_MODEL` (default in code is `gemini-flash-lite-latest`), `BUSINESS_NAME`, `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` (override Gemini with another OpenAI-compatible provider), `OPENAI_API_KEY` / `OPENAI_MODEL`, `LLM_REASONING_EFFORT`, Twilio fields, `PUBLIC_HOST`.

If `GEMINI_API_KEY` is set and `OPENAI_API_KEY` is not, the app uses Gemini’s OpenAI-compatible endpoint.

## How a call works

- Browser or Twilio sends mulaw 8 kHz frames to `WS /media`.
- Deepgram `SpeechStarted` barges in **only after** the agent has started sending audio for that turn (trailing mic noise will not cancel a reply that has not started yet).
- LLM tokens are split on `.?!` and sent to ElevenLabs as soon as a sentence completes. Up to four tool-call rounds: `check_availability`, `book_appointment`, `escalate_to_human`.
- If the LLM errors or returns no speech, the agent says a short fallback instead of staying silent.
- Gemini tool calls may include `extra_content` (thought signature); it is stored and sent back on the next request.
- Runtime files: `data/bookings.db`, `data/latency.csv`, `data/transcripts/<callSid>.json`.

## Tests

`pytest` is used by `tests/test_agent.py` but is not listed in `requirements.txt`. Install it into the venv if needed, then:

```powershell
.venv\Scripts\python.exe -m pip install pytest
.venv\Scripts\python.exe -m pytest -q
```

## Simulated call (no Twilio, no browser)

With the server already running:

```powershell
.venv\Scripts\python.exe scripts\simulate_call.py "Hi, my AC stopped cooling." "It is 12 Lake Road, my name is Rahul Sharma."
```

Prefix a line with `!` to talk over the agent (barge-in).

## Results (from `data/latency.csv` on this machine)

Server-side time from Deepgram end-of-turn to the first audio byte. This excludes the ~300 ms endpointing wait and playback delay on the client. The file mixes slower Gemini 3.6 turns from early testing with later `gemini-flash-lite-latest` turns.

| Metric | Value |
|---|---|
| Turns recorded | 29 |
| Time to first audio, min / median / p95 / max (ms) | 1890 / 2748 / 9225 / 22332 |
| Bookings completed correctly | fill in from your own test calls |
| Emergency calls escalated correctly | fill in from your own test calls |

```powershell
.venv\Scripts\python.exe scripts\latency_report.py
```

## Real phone (optional, not yet verified)

```powershell
ngrok http 8000
```

In the Twilio console, set the number’s Voice “A call comes in” webhook to `https://<ngrok-host>/incoming-call` (HTTP POST). Set `PUBLIC_HOST` to the ngrok hostname without `https://`. Trial accounts can only receive calls from verified numbers. Human transfer (`escalate_to_human`) also needs `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and `ESCALATION_NUMBER`.

## Known limitations

- Barge-in still cuts off on real speech (and loud noise) once the agent is talking.
- Partial assistant replies interrupted by barge-in are not written to history.
- SQLite is single-process; not for multi-server production.
- Twilio inbound calling has not been tested with a live number in this repo.
