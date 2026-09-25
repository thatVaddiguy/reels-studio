"""Choose clip boundaries from a transcript: Claude if an API key is set, else a heuristic."""
import json
import logging
import os

log = logging.getLogger("picker")

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")

RULES = """You pick short-form clips (Instagram Reels / TikTok / Shorts) from a video transcript.
Each transcript line is `[start - end] text`, in seconds.

Rules:
- Length: 20-60s each, aiming for 30-45s. Never under 15s or over 90s.
- How many: about 1 per 3-5 minutes of source, capped at 8. Quality over count. A very short source may yield just 1.
- Start on the hook: the first sentence must grab attention on its own (a bold claim, a question, a surprising number, a conflict). Skip warm-up filler.
- End clean: finish on a complete thought or punchline, never mid-sentence.
- Self-contained: skip anything that depends on earlier context.
- Boundaries: use segment start/end times from the transcript. Clips must not overlap. Order them by strength, best first.
- title: 2-6 words, lowercase, describes the clip.
- caption: one punchy line plus 3-5 relevant hashtags. No clickbait the clip doesn't pay off. Spell names correctly even if the transcript misspells them."""

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "caption": {"type": "string"},
                },
                "required": ["start", "end", "title", "caption"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def llm_enabled():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def pick(segments, duration, title, use_llm=True):
    """Returns (clips, method)."""
    if use_llm and llm_enabled():
        try:
            return _clean(_claude(segments, duration, title), duration), f"Claude ({CLAUDE_MODEL})"
        except Exception as e:  # never fail the job because of the picker
            log.warning("Claude picker failed, using heuristic: %s", e)
    return _clean(_heuristic(segments, duration, title), duration), "basic picker"


def _claude(segments, duration, title):
    import anthropic

    transcript = "\n".join(f"[{s['start']:.2f} - {s['end']:.2f}] {s['text']}" for s in segments)
    params = dict(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=RULES,
        messages=[{
            "role": "user",
            "content": f"Video title: {title}\nDuration: {duration:.0f}s\n\nTranscript:\n{transcript}",
        }],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except (anthropic.BadRequestError, TypeError):
        # Model or SDK without server-side fallbacks: plain request.
        resp = client.messages.create(**params)
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)["clips"]


def _heuristic(segments, duration, title):
    """Score windows of consecutive segments by speech density and punctuation energy."""
    if not segments or duration <= 60:
        first = segments[0]["text"] if segments else title
        return [{"start": 0.0, "end": duration, "title": title[:40], "caption": _caption(first)}]
    windows = []
    for i, s in enumerate(segments):
        end_j = None
        for j in range(i, len(segments)):
            length = segments[j]["end"] - s["start"]
            if length > 60:
                break
            if length >= 30:
                end_j = j
                if length >= 45:
                    break
        if end_j is None:
            continue
        seg = segments[i:end_j + 1]
        text = " ".join(x["text"] for x in seg)
        length = seg[-1]["end"] - s["start"]
        score = len(text.split()) / length * (1 + 0.15 * sum(text.count(c) for c in "?!"))
        windows.append((score, s["start"], seg[-1]["end"], seg[0]["text"]))
    want = max(1, min(8, round(duration / 240)))
    chosen = []
    for score, a, b, first in sorted(windows, reverse=True):
        if all(b <= c["start"] or a >= c["end"] for c in chosen):
            words = first.split()
            chosen.append({"start": a, "end": b, "title": " ".join(words[:5]).lower() or "clip",
                           "caption": _caption(first)})
            if len(chosen) == want:
                break
    return chosen


def _caption(first_line):
    line = first_line.strip()
    if len(line) > 90:
        line = line[:87].rsplit(" ", 1)[0] + "..."
    return f"{line} #reels #viral #clips"


def _clean(clips, duration):
    out = []
    for c in clips:
        a, b = max(0.0, float(c["start"])), min(duration, float(c["end"]))
        if b - a < 5 or any(not (b <= o["start"] or a >= o["end"]) for o in out):
            continue
        out.append({"start": a, "end": b, "title": c.get("title") or "clip", "caption": c.get("caption", "")})
    if not out:
        out = [{"start": 0.0, "end": min(duration, 60.0), "title": "clip", "caption": ""}]
    return out[:8]
