import os
from .config import EMOTIONS
from openai import OpenAI

client = OpenAI(api_key="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")

def summarize_song(song_id, song_segments, song_importance) -> str:
    counts = {e: 0 for e in EMOTIONS}
    for seg in song_segments:
        counts[seg["label"]] += 1
    total = len(song_segments)
    dist_str = ", ".join(
        f"{e}: {counts[e]}/{total} segments"
        for e in EMOTIONS
    )

    transitions = []
    prev = None
    for s in song_segments:
        cur = s["label"]
        if prev is None:
            prev = cur
            continue
        if cur != prev:
            transitions.append(f"{prev} → {cur} at segment {s['segment_index']}")
            prev = cur
    transitions_text = "; ".join(transitions) if transitions else "No major emotion changes detected."

    top_words_lines = []
    for e in EMOTIONS:
        freq = song_importance.get(e, {})
        if not freq:
            continue
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]
        words = ", ".join(w for w, _ in top)
        top_words_lines.append(f"{e}: {words}")
    top_words_str = "\n".join(top_words_lines)

    system_prompt = (
        "You are an assistant that explains the emotional arc of song lyrics. "
        "You MUST stay consistent with the provided emotion statistics and not invent new emotions."
    )

    user_prompt = f"""
Song: {song_id}

Emotion distribution (segment counts):
{dist_str}

Emotion transitions:
{transitions_text}

Top emotion-weighted words per emotion:
{top_words_str}

Write 2 short paragraphs:
1) Describe the overall emotional theme of the song and how it evolves.
2) Explain which emotions dominate different parts (beginning/middle/end) and mention a few indicative words.

Do NOT quote exact lyrics. Base your explanation only on the emotions and words provided.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=250,
        temperature=0.7,
    )

    return resp.choices[0].message.content.strip()
