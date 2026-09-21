"""Fake a phone call WITHOUT Twilio. Start the server first, in another terminal:

    .venv\\Scripts\\python.exe -m uvicorn main:app --port 8000

then run (each quoted argument is one thing the caller says, in order):

    .venv\\Scripts\\python.exe scripts\\simulate_call.py "Hi, my AC stopped cooling." "It is 12 Lake Road, my name is Rahul Sharma."

Prefix a line with ! to make the caller talk OVER the agent (tests barge-in):

    .venv\\Scripts\\python.exe scripts\\simulate_call.py "Hi, my AC is broken." "!Actually, forget that, do you fix water heaters?"

How it works: the caller's voice is synthesized with ElevenLabs as mulaw 8kHz (exactly what Twilio sends),
streamed to /media in real time, and we time how long the agent takes to start talking after the caller stops.
That number includes Deepgram's endpointing wait, so it is closer to what a real caller experiences than
latency.csv (which is measured inside the server). Run this from the same folder as the server so the
transcript in ./data/transcripts/CAsim.json can be printed at the end.
"""
import asyncio
import base64
import json
import os
import statistics
import sys
import time

import httpx
import websockets
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
load_dotenv(os.path.join(ROOT, ".env"))
URL = os.getenv("SIM_URL", "ws://localhost:8000/media")
VOICE = os.getenv("SIM_VOICE_ID") or os.getenv("ELEVENLABS_VOICE_ID", "")
CHUNK = 160                    # 20 ms of 8 kHz mulaw
SILENCE = b"\xff" * CHUNK      # 0xFF is digital silence in mulaw


def synth(text: str) -> bytes:
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}",
        params={"output_format": "ulaw_8000"},
        headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")},
        json={"text": text, "model_id": "eleven_flash_v2_5"},
        timeout=60,
    )
    r.raise_for_status()
    return r.content


def media_msg(chunk: bytes) -> str:
    return json.dumps({"event": "media", "media": {"payload": base64.b64encode(chunk).decode()}})


async def main(raw_lines):
    lines = [(l[1:].strip(), True) if l.startswith("!") else (l, False) for l in raw_lines]
    print("Synthesizing caller audio...")
    audios = [synth(text) for text, _ in lines]

    st = {"bytes": 0, "first_at": None, "last_at": None, "clears": 0, "mark": None}

    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"event": "start", "start": {"streamSid": "MZsim", "callSid": "CAsim"}}))

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("event") == "media":
                    now = time.perf_counter()
                    st["bytes"] += len(base64.b64decode(m["media"]["payload"]))
                    st["last_at"] = now
                    if st["first_at"] is None and st["mark"] is not None:
                        st["first_at"] = now
                elif m.get("event") == "clear":
                    st["clears"] += 1

        rd = asyncio.create_task(reader())

        async def send_silence(seconds):
            for _ in range(int(seconds / 0.02)):
                await ws.send(media_msg(SILENCE))
                await asyncio.sleep(0.02)

        print("Waiting for the agent's greeting...")
        await send_silence(6)

        delays = []
        for i, ((text, _), audio) in enumerate(zip(lines, audios)):
            interrupt_next = i + 1 < len(lines) and lines[i + 1][1]
            st.update(bytes=0, first_at=None, last_at=None, mark=None)
            print(f"\nCALLER: {text}")
            for j in range(0, len(audio), CHUNK):
                await ws.send(media_msg(audio[j:j + CHUNK].ljust(CHUNK, b"\xff")))
                await asyncio.sleep(0.02)
            st["mark"] = time.perf_counter()  # caller just stopped talking

            # keep the line alive with silence while the agent answers
            while True:
                await ws.send(media_msg(SILENCE))
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                if st["first_at"] is None:
                    if now - st["mark"] > 15:
                        print("  AGENT: (no audio within 15 s)")
                        break
                    continue
                if interrupt_next and now - st["first_at"] > 1.0:
                    print("  (caller interrupts while the agent is talking)")
                    break
                playback = st["bytes"] / 8000  # seconds of audio the agent produced
                if now - st["last_at"] > 1.0 and now - st["first_at"] > playback:
                    break
            if st["first_at"] is not None:
                delay = st["first_at"] - st["mark"]
                delays.append(delay)
                print(f"  AGENT: started talking after {delay * 1000:.0f} ms, spoke ~{st['bytes'] / 8000:.1f} s of audio")

        await ws.send(json.dumps({"event": "stop"}))
        await asyncio.sleep(1)
        rd.cancel()

    print(f"\nclear events received (barge-in flushes): {st['clears']}")
    if delays:
        print(f"caller-perceived delay: median {statistics.median(delays) * 1000:.0f} ms over {len(delays)} turns")
    try:
        print("\n--- transcript ---")
        for msg in json.load(open("data/transcripts/CAsim.json")):
            if msg["role"] in ("user", "assistant") and msg.get("content"):
                print(f"{msg['role'].upper():9}: {msg['content']}")
    except FileNotFoundError:
        print("(no transcript found - run this from the same folder as the server)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1:]))
