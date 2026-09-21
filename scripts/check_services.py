"""Checks that your API keys work. From the project root:

    .venv\\Scripts\\python.exe scripts\\check_services.py

1. Gemini:      one tiny chat completion (OpenAI-compatible Gemini endpoint)
2. ElevenLabs:  synthesize a sentence to data/test.mp3 (open it and listen)
3. Deepgram:    transcribe that same mp3 -> if the words come back, TTS and STT both work
"""
import os
import sys

import httpx
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
load_dotenv(os.path.join(ROOT, ".env"))
PHRASE = "Hello, my air conditioner stopped cooling and I need a technician tomorrow."
ok = True


def report(name, passed, detail=""):
    global ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


# 1. Gemini (preferred) or OpenAI
try:
    gemini = os.getenv("GEMINI_API_KEY", "")
    if gemini:
        r = httpx.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {gemini}"},
            json={"model": os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
                  "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                  "max_tokens": 256,
                  "extra_body": {"google": {"thinking_config": {"thinking_level": "low"}}}},
            timeout=30,
        )
        name = "Gemini"
    else:
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
            json={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                  "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
                  "max_tokens": 5},
            timeout=30,
        )
        name = "OpenAI"
    if r.status_code == 200:
        detail = (r.json().get("choices") or [{}])[0].get("message", {}).get("content")
        report(name, bool(detail), f"-> {detail!r}")
    else:
        report(name, False, f"-> {r.text[:200]!r}")
except Exception as e:
    report("Gemini" if os.getenv("GEMINI_API_KEY") else "OpenAI", False, str(e))

# 2. ElevenLabs
audio = b""
try:
    voice = os.getenv("ELEVENLABS_VOICE_ID", "")
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")},
        json={"text": PHRASE, "model_id": "eleven_flash_v2_5"},
        timeout=60,
    )
    if r.status_code == 200 and r.content:
        audio = r.content
        os.makedirs("data", exist_ok=True)
        open(os.path.join("data", "test.mp3"), "wb").write(audio)
        report("ElevenLabs", True, f"-> saved data/test.mp3 ({len(audio)} bytes), go listen to it")
    else:
        report("ElevenLabs", False, r.text[:200])
except Exception as e:
    report("ElevenLabs", False, str(e))

# 3. Deepgram (transcribe the mp3 we just made)
try:
    if not audio:
        raise RuntimeError("skipped: no audio from ElevenLabs")
    r = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-2", "smart_format": "true"},
        headers={"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY', '')}", "Content-Type": "audio/mpeg"},
        content=audio,
        timeout=60,
    )
    if r.status_code == 200:
        text = r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
        report("Deepgram", bool(text), f"-> heard: {text!r}")
    else:
        report("Deepgram", False, r.text[:200])
except Exception as e:
    report("Deepgram", False, str(e))

sys.exit(0 if ok else 1)
