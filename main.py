"""
AI voice agent for home services (HVAC / plumbing) - v1.

Pipeline:  caller -> Twilio Media Streams (WebSocket) -> Deepgram live STT
           -> LLM (streaming, tool calls) -> ElevenLabs streaming TTS -> Twilio -> caller

Run:  uvicorn main:app --port 8000     (then expose with `ngrok http 8000`)
"""
import asyncio
import base64
import csv
import json
import os
import re
import sqlite3
import time
import traceback
from datetime import datetime

import httpx
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from openai import AsyncOpenAI

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
# Any OpenAI-compatible provider works (OpenAI, Groq, Gemini, OpenRouter...).
# GEMINI_API_KEY alone is enough; LLM_* / OPENAI_* still override if set.
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or ""
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or GEMINI_KEY or ""
if os.getenv("LLM_BASE_URL"):
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
elif GEMINI_KEY and not os.getenv("OPENAI_API_KEY"):
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
else:
    LLM_BASE_URL = None
LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("GEMINI_MODEL")
    or os.getenv("OPENAI_MODEL")
    or ("gemini-flash-lite-latest" if GEMINI_KEY and not os.getenv("OPENAI_API_KEY") else "gpt-4o-mini")
)
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT") or None
USE_GEMINI = bool(GEMINI_KEY) and bool(LLM_BASE_URL) and "generativelanguage.googleapis.com" in LLM_BASE_URL
BUSINESS = os.getenv("BUSINESS_NAME") or "HVAC and Plumbing"
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
ESCALATION_NUMBER = os.getenv("ESCALATION_NUMBER", "")  # e.g. the owner's phone, E.164

# mulaw 8kHz is what Twilio speaks; Deepgram and ElevenLabs both support it natively (no transcoding)
DG_URL = (
    "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1"
    "&model=nova-2&interim_results=true&endpointing=300&utterance_end_ms=1000"
    "&vad_events=true&smart_format=true"
)

app = FastAPI()
llm = AsyncOpenAI(api_key=LLM_API_KEY or "missing", base_url=LLM_BASE_URL)
SENT_END = re.compile(r"(?<=[.!?])\s+")

# Runtime files stay out of the project root.
DATA_DIR = "data"
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
BOOKINGS_DB = os.path.join(DATA_DIR, "bookings.db")
LATENCY_CSV = os.path.join(DATA_DIR, "latency.csv")
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# ---------------------------------------------------------------- booking "backend"
db = sqlite3.connect(BOOKINGS_DB, check_same_thread=False)
db.execute(
    "CREATE TABLE IF NOT EXISTS bookings (id INTEGER PRIMARY KEY, name TEXT, phone TEXT, "
    "address TEXT, issue TEXT, slot TEXT UNIQUE, created_at TEXT)"
)
SLOTS = ["09:00", "11:00", "14:00", "16:00"]


def check_availability(date: str):
    booked = {r[0] for r in db.execute("SELECT slot FROM bookings WHERE slot LIKE ?", (date + "%",))}
    return [s for s in SLOTS if f"{date} {s}" not in booked]


def book_appointment(name, phone, address, issue, date, time_slot):
    slot = f"{date} {time_slot}"
    try:
        db.execute(
            "INSERT INTO bookings (name, phone, address, issue, slot, created_at) VALUES (?,?,?,?,?,?)",
            (name, phone, address, issue, slot, datetime.now().isoformat()),
        )
        db.commit()
        return {"ok": True, "slot": slot}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "that slot is already taken"}


async def escalate(call_sid, reason=""):
    """Redirect the live call to a human via Twilio's REST API."""
    print(f"ESCALATE ({reason})")
    if not (TWILIO_SID and TWILIO_TOKEN and ESCALATION_NUMBER and call_sid):
        return {"ok": False, "note": "no human line configured; take a callback number instead"}
    twiml = f"<Response><Say>Connecting you to a technician now.</Say><Dial>{ESCALATION_NUMBER}</Dial></Response>"
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Calls/{call_sid}.json",
            auth=(TWILIO_SID, TWILIO_TOKEN),
            data={"Twiml": twiml},
        )
    return {"ok": r.status_code < 300}


TOOLS = [
    {"type": "function", "function": {
        "name": "check_availability",
        "description": "List free appointment slots for a date.",
        "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "book_appointment",
        "description": "Book a confirmed slot after the caller has confirmed all details.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "phone": {"type": "string"}, "address": {"type": "string"},
            "issue": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
            "time_slot": {"type": "string", "description": "HH:MM, one of the free slots"}},
            "required": ["name", "phone", "address", "issue", "date", "time_slot"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Transfer the caller to a human (emergency, upset caller, or repeated misunderstanding).",
        "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]


def system_prompt():
    now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    return f"""You are the phone receptionist for {BUSINESS}, an HVAC and plumbing company. Current date/time: {now}.
This is a live phone call: reply in one or two short spoken sentences. No lists, markdown, or emojis. Ask one question at a time.
Goal: understand the problem, then collect name, phone number, service address, and a preferred day/time, then book.
Rules:
- Call check_availability before offering times. Never invent availability.
- Read back name, phone and address and get a clear yes before calling book_appointment.
- Never quote prices or promise arrival times beyond the booked slot.
- EMERGENCIES (gas smell, burst pipe or flooding, sparks or burning smell): tell them to get to safety and call local emergency services if in danger, then call escalate_to_human immediately.
- If the caller asks for a human, is upset, or you fail to understand them twice, call escalate_to_human.
- If asked about something we don't handle (for example appliance repair such as a washing machine), say so in one short sentence, suggest they contact an appliance repair service, and offer help with heating, cooling or plumbing instead.
- Always reply with something spoken. Never stay silent."""


# ---------------------------------------------------------------- one object per phone call
class Call:
    def __init__(self, twilio_ws: WebSocket):
        self.tw = twilio_ws
        self.stream_sid = None
        self.call_sid = None
        self.history = [{"role": "system", "content": system_prompt()}]
        self.parts = []            # finalized STT fragments for the current user turn
        self.task = None           # the running "agent reply" task, so we can cancel it (barge-in)
        self.turn_end_ts = None    # when the user's turn ended (for latency measurement)
        self.turn_sentences = 0    # sentences queued for speech this turn (0 => the agent would be silent)
        self.latency_logged = True
        self.http = httpx.AsyncClient(timeout=30)
        self._audio_tail = 0.0
        self._audio_t = time.perf_counter()

    # ---- output to Twilio
    async def send_json(self, obj):
        await self.tw.send_text(json.dumps(obj))

    async def send_audio(self, audio: bytes):
        if not self.latency_logged and self.turn_end_ts:
            self.log_latency()
        now = time.perf_counter()
        self._audio_tail = max(0.0, self._audio_tail - (now - self._audio_t)) + len(audio) / 8000
        self._audio_t = now
        await self.send_json({"event": "media", "streamSid": self.stream_sid,
                              "media": {"payload": base64.b64encode(audio).decode()}})

    def playback_left(self):
        """Seconds of agent audio already sent that the client may still be playing."""
        return max(0.0, self._audio_tail - (time.perf_counter() - self._audio_t))

    def log_latency(self):
        """Time from Deepgram's end-of-turn signal to the first audio byte we send to Twilio."""
        ms = (time.perf_counter() - self.turn_end_ts) * 1000
        self.latency_logged = True
        print(f"latency: {ms:.0f} ms")
        with open(LATENCY_CSV, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(), self.call_sid, round(ms)])

    # ---- TTS
    async def tts(self, text: str):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
        async with self.http.stream(
            "POST", url, params={"output_format": "ulaw_8000"},
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            json={"text": text, "model_id": "eleven_flash_v2_5"},
        ) as r:
            if r.status_code != 200:
                print("TTS error", r.status_code, (await r.aread())[:200])
                return
            async for chunk in r.aiter_bytes():
                if chunk:
                    await self.send_audio(chunk)

    async def speaker(self, q: asyncio.Queue):
        while (sentence := await q.get()) is not None:
            try:
                await self.tts(sentence)
            except Exception as e:  # e.g. network timeout: log it, keep going
                print(f"TTS ERROR: {type(e).__name__}: {e}")

    async def greet(self):
        text = f"Thanks for calling {BUSINESS}! How can I help you today?"
        self.history.append({"role": "assistant", "content": text})
        await self.tts(text)

    # ---- LLM
    async def llm_round(self, q: asyncio.Queue):
        """One streamed LLM call. Complete sentences go to the TTS queue as soon as they exist."""
        extra_args = {}
        if LLM_REASONING_EFFORT:
            extra_args["reasoning_effort"] = LLM_REASONING_EFFORT
        elif USE_GEMINI and "lite" not in LLM_MODEL.lower():
            extra_args["extra_body"] = {
                "extra_body": {"google": {"thinking_config": {"thinking_level": "low"}}}
            }
        stream = await llm.chat.completions.create(
            model=LLM_MODEL, messages=self.history, tools=TOOLS, stream=True, temperature=0.3, **extra_args)
        text, buf, calls = "", "", {}
        async for ev in stream:
            if not ev.choices:
                continue
            d = ev.choices[0].delta
            if d.content:
                text += d.content
                buf += d.content
                *done, buf = SENT_END.split(buf)
                for s in done:
                    if s.strip():
                        self.turn_sentences += 1
                        await q.put(s.strip())
            for tc in d.tool_calls or []:
                idx = getattr(tc, "index", None)
                if idx is None:  # some providers omit the index; a new id means a new call
                    idx = len(calls) if (tc.id or not calls) else len(calls) - 1
                c = calls.setdefault(idx, {"id": "", "name": "", "args": "", "extra": None})
                extra = getattr(tc, "extra_content", None)  # Gemini 3 puts its thought_signature here
                if extra:
                    c["extra"] = extra
                if tc.id:
                    c["id"] = tc.id
                if tc.function and tc.function.name:
                    c["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    c["args"] += tc.function.arguments
        if buf.strip():
            self.turn_sentences += 1
            await q.put(buf.strip())
        if text.strip():
            print("AGENT:", text.strip())
        for i, c in calls.items():
            if not c["id"]:  # some providers omit ids
                c["id"] = f"call_{i}"
        msg = {"role": "assistant", "content": text}
        if calls:
            tool_calls = []
            for c in calls.values():
                entry = {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["args"]}}
                if c["extra"]:  # must be sent back verbatim or Gemini 3 rejects the next request
                    entry["extra_content"] = c["extra"]
                tool_calls.append(entry)
            msg["tool_calls"] = tool_calls
        self.history.append(msg)
        return calls

    async def run_tool(self, name, args_json):
        try:
            args = json.loads(args_json or "{}")
            print(f"TOOL {name} {args}")
            if name == "check_availability":
                return {"free_slots": check_availability(**args)}
            if name == "book_appointment":
                return book_appointment(**args)
            if name == "escalate_to_human":
                return await escalate(self.call_sid, **args)
        except Exception as e:  # never let a bad tool call crash the call
            return {"error": str(e)}
        return {"error": "unknown tool"}

    async def respond(self, user_text: str):
        self.history.append({"role": "user", "content": user_text})
        keep = len(self.history)  # if the LLM step fails, drop the half-finished assistant/tool messages after this point
        self.turn_sentences = 0
        q: asyncio.Queue = asyncio.Queue()
        speaker = asyncio.create_task(self.speaker(q))
        try:
            try:
                for _ in range(4):  # allow a few tool-call rounds
                    calls = await self.llm_round(q)
                    if not calls:
                        break
                    for c in calls.values():
                        result = await self.run_tool(c["name"], c["args"])
                        self.history.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
            except Exception as e:
                print(f"LLM ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()
                del self.history[keep:]
            if self.turn_sentences == 0:  # error or empty answer: a caller must never get silence
                print("no reply generated; speaking fallback")
                await q.put("Sorry, I'm having a little trouble. Could you say that again?")
            await q.put(None)
            await speaker
        except asyncio.CancelledError:
            speaker.cancel()
            raise

    # ---- STT events / turn-taking
    async def interrupt(self):
        """Caller started talking: stop generating and flush audio already buffered at Twilio."""
        # Ignore VAD right after the user's turn ends, before any agent audio is sent.
        # Trailing noise would otherwise cancel the LLM and the caller hears silence.
        if self.task and not self.task.done() and not self.latency_logged and self.turn_end_ts:
            return
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self._audio_tail = 0.0
        self._audio_t = time.perf_counter()
        # always clear: Twilio may still be *playing* audio even after we finished sending it
        await self.send_json({"event": "clear", "streamSid": self.stream_sid})

    @staticmethod
    def log_task_error(task: asyncio.Task):
        if not task.cancelled() and task.exception():
            print(f"TASK ERROR: {type(task.exception()).__name__}: {task.exception()}")

    def flush_turn(self):
        if not self.parts:
            return
        text = " ".join(self.parts)
        self.parts = []
        print("USER:", text)
        self.turn_end_ts = time.perf_counter()
        self.latency_logged = False
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = asyncio.create_task(self.respond(text))
        self.task.add_done_callback(self.log_task_error)

    async def deepgram_loop(self, dg):
        try:
            async for raw in dg:
                m = json.loads(raw)
                kind = m.get("type")
                if kind == "SpeechStarted":
                    # VAD alone is too jumpy (echo, breath, keyboard). Only cut playback
                    # when the agent is idle-or-done sending, not while TTS is still queued.
                    if self.playback_left() > 0.2:
                        continue
                    await self.interrupt()
                elif kind == "Results":
                    alt = m["channel"]["alternatives"][0]
                    heard = (alt.get("transcript") or "").strip()
                    # Real words while we are speaking: barge-in even if audio is still queued.
                    if heard and self.task and not self.task.done() and self.latency_logged:
                        await self.interrupt()
                    if m.get("is_final") and heard:
                        self.parts.append(heard)
                    if m.get("speech_final"):
                        self.flush_turn()
                elif kind == "UtteranceEnd":  # fallback if speech_final never fired
                    self.flush_turn()
                elif kind in ("Error", "Warning"):
                    print(f"DEEPGRAM {kind}: {m}")
            print("DEEPGRAM connection closed")  # if you see this mid-call, nothing you say will be heard
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"DEEPGRAM ERROR: {type(e).__name__}: {e}")

    def save_transcript(self):
        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        with open(os.path.join(TRANSCRIPTS_DIR, f"{self.call_sid or 'unknown'}.json"), "w") as f:
            json.dump(self.history[1:], f, indent=2)


# ---------------------------------------------------------------- HTTP / WebSocket endpoints
@app.get("/")
async def web_client():
    """Free way to test without Twilio: open http://localhost:8000 and talk to the agent from your browser."""
    html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "web_client.html")
    return FileResponse(html)


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Twilio hits this when someone calls; we answer by opening a bidirectional media stream."""
    host = os.getenv("PUBLIC_HOST") or request.url.hostname
    twiml = (f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
             f'<Stream url="wss://{host}/media"/></Connect></Response>')
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()
    call = Call(ws)
    # Deepgram accepts the key via the websocket subprotocol; if that fails on your version, use
    # websockets.connect(DG_URL, additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"})
    dg = await websockets.connect(DG_URL, subprotocols=["token", DEEPGRAM_API_KEY])
    dg_task = asyncio.create_task(call.deepgram_loop(dg))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            event = msg.get("event")
            if event == "start":
                call.stream_sid = msg["start"]["streamSid"]
                call.call_sid = msg["start"]["callSid"]
                call.task = asyncio.create_task(call.greet())
                call.task.add_done_callback(call.log_task_error)
            elif event == "media":
                await dg.send(base64.b64decode(msg["media"]["payload"]))
            elif event == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        dg_task.cancel()
        if call.task and not call.task.done():
            call.task.cancel()
        try:
            await dg.send(json.dumps({"type": "CloseStream"}))
            await dg.close()
        except Exception:
            pass
        await call.http.aclose()
        call.save_transcript()