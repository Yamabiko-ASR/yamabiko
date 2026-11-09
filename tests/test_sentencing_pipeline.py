import asyncio
import pytest


@pytest.mark.asyncio
async def test_two_sentences_emit_two_broadcasts():
    """If ASR yields two complete sentences, translate + broadcast each."""
    from yamabiko.main import translation_stage, Segment
    from yamabiko.state import AppState

    in_q: asyncio.Queue = asyncio.Queue()
    # Seed a single chunk carrying two sentences.
    await in_q.put(Segment(audio=None, t_chunk_emit=0.0, t_asr_done=0.0, jp="こんにちは。元気ですか？"))
    # Kick the loop hard so it exits after processing the 2 sub-segments.

    broadcasts: list[dict] = []

    class FakeManager:
        async def broadcast(self, msg):
            broadcasts.append(msg)

    class FakeTranslator:
        async def translate(self, jp, genre, ctx):
            return f"[EN]{jp}"

    class FakeCtx:
        def snapshot(self):
            return []

        def push(self, _):
            pass

        def clear(self):
            pass

    class FakeStats:
        def record(self, *_):
            pass

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", newline="") as f:
        lat_log = f
        done = asyncio.Event()
        state = AppState()
        try:
            await asyncio.wait_for(
                translation_stage(
                    in_q,
                    FakeTranslator(),
                    FakeCtx(),
                    state,
                    FakeStats(),
                    lat_log,
                    max_segments=2,
                    done=done,
                    manager=FakeManager(),
                ),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            pass

    subs = [b for b in broadcasts if b.get("type") == "subtitle"]
    assert len(subs) == 2, f"expected 2, got {len(subs)}: {subs}"
    assert "こんにちは" in subs[0]["jp"]
    assert "元気" in subs[1]["jp"]
