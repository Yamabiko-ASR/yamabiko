"""Quick llama-server sanity check — avoids Windows bash UTF-8 pitfalls."""
from __future__ import annotations
import json
import sys
import time
import urllib.request

ENDPOINT = "http://localhost:8090/v1/chat/completions"
SYSTEM = (
    "You are a subtitle translator. Do not use chain-of-thought reasoning.\n"
    "Translate Japanese TV dialogue to concise, natural English.\n"
    "Preserve tone and register. Output only the translation, no explanation."
)

samples = [
    ("こんにちは、今日はいい天気ですね。", "variety"),
    ("えーっとですね、それはちょっと難しいかな。", "variety"),
    ("社長が先ほどお見えになりました。", "drama"),
    ("本日の主要ニュースをお伝えします。", "news"),
]

GENRE_NOTES = {
    "variety": "Fast informal speech. Preserve energy and reaction language. Sentence fragments are fine.",
    "drama": "Preserve politeness register (keigo vs. plain form). Full sentences preferred.",
    "news": "Formal register. Precise over natural.",
    "anime": "Preserve character speech patterns and affect. Drop honorifics. Keep sentence fragments as fragments.",
}

for jp, genre in samples:
    body = {
        "model": "qwen3.5-9b",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Genre: {genre}\n{GENRE_NOTES[genre]}\nPrevious: (none)\nTranslate: {jp}"},
        ],
        "max_tokens": 80,
        "temperature": 0.1,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read().decode("utf-8"))
    dt = (time.perf_counter() - t0) * 1000
    content = resp["choices"][0]["message"]["content"]
    has_think = "<think>" in content
    print(f"[{genre}] {dt:6.0f} ms  think={has_think}")
    print(f"  JP: {jp}")
    print(f"  EN: {content.strip()}")
    print()
    if has_think:
        print("FAIL: <think> tag present — thinking mode leak")
        sys.exit(2)

print("OK: no <think> leaks across 4 samples")
