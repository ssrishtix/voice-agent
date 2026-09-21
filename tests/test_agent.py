"""Offline tests: no API keys, no network, no phone. Run with:  pytest -v"""
import asyncio
import base64
import importlib
import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def m(tmp_path, monkeypatch):
    """Fresh copy of main.py per test, with its own empty bookings.db in a temp dir."""
    monkeypatch.chdir(tmp_path)
    import main
    return importlib.reload(main)


class FakeWS:
    """Stands in for the Twilio websocket; records everything the agent sends."""
    def __init__(self):
        self.sent = []

    async def send_text(self, s):
        self.sent.append(json.loads(s))


# ---------------------------------------------------------------- webhook + booking tools
def test_webhook_returns_stream_twiml(m):
    r = TestClient(m.app).post("/incoming-call")
    assert r.status_code == 200
    assert "<Stream" in r.text and "/media" in r.text


def test_availability_and_double_booking(m):
    assert m.check_availability("2026-09-25") == m.SLOTS
    assert m.book_appointment("A", "1", "addr", "ac", "2026-09-25", "11:00")["ok"]
    assert not m.book_appointment("B", "2", "addr", "ac", "2026-09-25", "11:00")["ok"]
    assert "11:00" not in m.check_availability("2026-09-25")
    assert "11:00" in m.check_availability("2026-09-26")  # other dates unaffected


def test_browser_client_is_served(m):
    r = TestClient(m.app).get("/")
    assert r.status_code == 200
    assert "AudioWorklet" in r.text and "/media" in r.text


# ---------------------------------------------------------------- LLM streaming + tool calls
def _ev(content=None, tool=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool))])


def _tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()


def test_sentences_stream_out_and_tool_call_is_executed(m, monkeypatch):
    rounds = [
        [_ev("Let me check. "), _ev(tool=[_tc(0, "c1", "check_availability", '{"date":')]),
         _ev(tool=[_tc(0, None, None, '"2026-09-26"}')])],
        [_ev("We have 9 and 11 free. "), _ev("Which works?")],
    ]

    async def fake_create(**kw):
        return FakeStream(rounds.pop(0))

    monkeypatch.setattr(m.llm.chat.completions, "create", fake_create)
    spoken = []
    call = m.Call(FakeWS())

    async def fake_tts(text):
        spoken.append(text)

    call.tts = fake_tts
    asyncio.run(call.respond("book me tomorrow"))

    assert spoken == ["Let me check.", "We have 9 and 11 free.", "Which works?"]
    roles = [h["role"] for h in call.history[1:]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "09:00" in call.history[3]["content"]  # tool result was fed back to the model


def test_tool_calls_without_index_or_id(m, monkeypatch):
    """Some OpenAI-compatible providers omit tool_call index/id in streamed chunks."""
    def bare(id=None, name=None, args=None):
        return NS(id=id, function=NS(name=name, arguments=args))  # no .index attribute at all

    events = [_ev(tool=[bare("x1", "check_availability", '{"date":')]), _ev(tool=[bare(None, None, '"2026-09-26"}')])]

    async def fake_create(**kw):
        return FakeStream(events)

    monkeypatch.setattr(m.llm.chat.completions, "create", fake_create)
    call = m.Call(FakeWS())
    calls = asyncio.run(call.llm_round(asyncio.Queue()))
    assert len(calls) == 1
    only = next(iter(calls.values()))
    assert only["name"] == "check_availability" and only["args"] == '{"date":"2026-09-26"}'

    events[:] = [_ev(tool=[bare(None, "check_availability", '{"date":"2026-09-26"}')])]  # no id either
    calls = asyncio.run(call.llm_round(asyncio.Queue()))
    assert next(iter(calls.values()))["id"].startswith("call_")


def test_gemini_thought_signature_is_replayed(m, monkeypatch):
    """Gemini 3 returns tool_calls[].extra_content.google.thought_signature and needs it sent back unchanged."""
    sig = {"google": {"thought_signature": "abc123"}}
    chunk = NS(id="c1", index=0, extra_content=sig, function=NS(name="check_availability", arguments='{"date":"2026-09-26"}'))

    async def fake_create(**kw):
        return FakeStream([_ev(tool=[chunk])])

    monkeypatch.setattr(m.llm.chat.completions, "create", fake_create)
    call = m.Call(FakeWS())
    asyncio.run(call.llm_round(asyncio.Queue()))
    assert call.history[-1]["tool_calls"][0]["extra_content"] == sig


def test_no_extra_content_when_provider_sends_none(m, monkeypatch):
    async def fake_create(**kw):
        return FakeStream([_ev(tool=[_tc(0, "c1", "check_availability", '{"date":"2026-09-26"}')])])

    monkeypatch.setattr(m.llm.chat.completions, "create", fake_create)
    call = m.Call(FakeWS())
    asyncio.run(call.llm_round(asyncio.Queue()))
    assert "extra_content" not in call.history[-1]["tool_calls"][0]


def test_llm_error_still_speaks_a_fallback_and_keeps_history_valid(m, monkeypatch):
    async def boom(**kw):
        raise RuntimeError("429 quota exceeded")

    monkeypatch.setattr(m.llm.chat.completions, "create", boom)
    spoken = []
    call = m.Call(FakeWS())

    async def fake_tts(text):
        spoken.append(text)

    call.tts = fake_tts
    asyncio.run(call.respond("how do I get my washing machine fixed"))
    assert spoken and "trouble" in spoken[0]
    assert call.history[-1]["role"] == "user"  # no half-finished assistant/tool messages left behind


def test_empty_llm_answer_speaks_a_fallback(m, monkeypatch):
    async def fake_create(**kw):
        return FakeStream([_ev(None)])  # model returns nothing at all

    monkeypatch.setattr(m.llm.chat.completions, "create", fake_create)
    spoken = []
    call = m.Call(FakeWS())

    async def fake_tts(text):
        spoken.append(text)

    call.tts = fake_tts
    asyncio.run(call.respond("hello?"))
    assert len(spoken) == 1


def test_tts_failure_on_one_sentence_does_not_stop_the_reply(m, monkeypatch):
    async def fake_create(**kw):
        return FakeStream([_ev("First sentence. "), _ev("Second sentence.")])

    monkeypatch.setattr(m.llm.chat.completions, "create", fake_create)
    spoken = []
    call = m.Call(FakeWS())

    async def flaky_tts(text):
        if text.startswith("First"):
            raise TimeoutError("elevenlabs timed out")
        spoken.append(text)

    call.tts = flaky_tts
    asyncio.run(call.respond("hi"))
    assert spoken == ["Second sentence."]


def test_bad_tool_arguments_do_not_crash(m):
    call = m.Call(FakeWS())
    out = asyncio.run(call.run_tool("book_appointment", '{"name": "only a name"}'))
    assert "error" in out


# ---------------------------------------------------------------- turn-taking + barge-in
def test_barge_in_cancels_reply_and_sends_clear(m):
    async def scenario():
        ws = FakeWS()
        call = m.Call(ws)
        call.stream_sid = "MZ1"
        call.task = asyncio.create_task(asyncio.sleep(30))  # pretend the agent is mid-reply
        await asyncio.sleep(0)
        await call.interrupt()
        assert call.task.cancelled()
        assert {"event": "clear", "streamSid": "MZ1"} in ws.sent

    asyncio.run(scenario())


def test_vad_does_not_cut_agent_while_audio_is_queued(m):
    """SpeechStarted during TTS playback is usually echo; do not stop the agent."""
    async def scenario():
        ws = FakeWS()
        call = m.Call(ws)
        call.stream_sid = "MZ1"
        call.latency_logged = True
        call._audio_tail = 3.0
        call._audio_t = __import__("time").perf_counter()
        call.task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)

        async def fake_dg():
            yield json.dumps({"type": "SpeechStarted"})

        await call.deepgram_loop(fake_dg())
        assert not call.task.done()
        assert {"event": "clear", "streamSid": "MZ1"} not in ws.sent

    asyncio.run(scenario())


def test_real_words_still_barge_in_while_agent_is_speaking(m):
    async def scenario():
        ws = FakeWS()
        call = m.Call(ws)
        call.stream_sid = "MZ1"
        call.latency_logged = True
        call._audio_tail = 3.0
        call._audio_t = __import__("time").perf_counter()
        call.task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)

        async def fake_dg():
            yield json.dumps({"type": "Results", "is_final": False, "speech_final": False,
                              "channel": {"alternatives": [{"transcript": "wait"}]}})

        await call.deepgram_loop(fake_dg())
        assert call.task.cancelled()
        assert {"event": "clear", "streamSid": "MZ1"} in ws.sent

    asyncio.run(scenario())


def test_deepgram_events_drive_turns(m, monkeypatch):
    replies = []

    async def scenario():
        ws = FakeWS()
        call = m.Call(ws)
        call.stream_sid = "MZ1"

        async def fake_respond(text):
            replies.append(text)

        call.respond = fake_respond

        def result(text, final=True, speech_final=False):
            return json.dumps({"type": "Results", "is_final": final, "speech_final": speech_final,
                               "channel": {"alternatives": [{"transcript": text}]}})

        async def fake_dg():
            yield json.dumps({"type": "SpeechStarted"})
            yield result("my AC")                        # final fragment, turn not over yet
            yield result("is broken", speech_final=True)  # turn over
            yield result("", speech_final=True)           # empty -> must not trigger a reply

        await call.deepgram_loop(fake_dg())
        await asyncio.sleep(0)  # let the created reply task run
        assert {"event": "clear", "streamSid": "MZ1"} in ws.sent

    asyncio.run(scenario())
    assert replies == ["my AC is broken"]


# ---------------------------------------------------------------- /media wiring
def test_media_endpoint_wires_twilio_to_deepgram(m, monkeypatch):
    class FakeDG:
        def __init__(self):
            self.received = []

        async def send(self, data):
            self.received.append(data)

        async def close(self):
            pass

        def __aiter__(self):
            async def forever():
                await asyncio.sleep(3600)
                yield ""
            return forever()

    dg = FakeDG()

    async def fake_connect(*a, **kw):
        return dg

    monkeypatch.setattr(m.websockets, "connect", fake_connect)
    spoken = []

    async def fake_tts(self, text):
        spoken.append(text)

    monkeypatch.setattr(m.Call, "tts", fake_tts)

    audio = b"\xff" * 160
    with TestClient(m.app).websocket_connect("/media") as ws:
        ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}}))
        ws.send_text(json.dumps({"event": "media", "media": {"payload": base64.b64encode(audio).decode()}}))
        ws.send_text(json.dumps({"event": "stop"}))

    assert dg.received and dg.received[0] == audio
    assert spoken and "Thanks for calling" in spoken[0]