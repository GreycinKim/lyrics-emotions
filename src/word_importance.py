import pandas as pd
import numpy as np
import re
from collections import defaultdict
from .config import EMOTIONS

NEGATORS = {"not", "never", "no", "without", "nothing", "hardly"}
INTENSIFIERS = {"very", "really", "so", "too", "extremely", "incredibly"}
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "so",
    "to", "of", "in", "on", "at", "for", "from",
    "is", "are", "am", "was", "were", "be", "been",
    "i", "you", "we", "they", "he", "she", "it",
    "me", "my", "your", "our", "their",
    "im", "youre", "dont", "cant", "aint"
}

def load_lexicon(path: str):
    df = pd.read_csv(path)
    lex = defaultdict(lambda: {e: 0.0 for e in EMOTIONS})
    for _, row in df.iterrows():
        w = str(row["word"]).lower()
        e = str(row["emotion"]).lower()
        s_raw = str(row["score"]).strip()

        # If someone accidentally put a comment after the number, keep only first token
        s_raw = s_raw.split()[0]

        try:
            s = float(s_raw)
        except ValueError:
            # skip bad rows gracefully
            continue

        if e in lex[w]:
            lex[w][e] = max(lex[w][e], s)
    return lex

def simple_tokenize(text: str):
    # keep alphabetic words and interior apostrophes (e.g., don't, i'm)
    raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())

    tokens = []
    for t in raw_tokens:
        # strip leading/trailing apostrophes: "'re" -> "re", "love'" -> "love"
        t = t.strip("'")

        # drop very short tokens (gets rid of 's, 're, 'm, etc.)
        if len(t) <= 2:
            continue

        tokens.append(t)

    return tokens

def apply_rules(tokens, lexicon_emotions):
    adjusted = [dict(scores) for scores in lexicon_emotions]

    for i, w in enumerate(tokens):
        if w in NEGATORS:
            for j in range(i+1, min(i+3, len(tokens))):
                for e in EMOTIONS:
                    adjusted[j][e] *= -0.7
        elif w in INTENSIFIERS:
            j = i + 1
            if j < len(tokens):
                for e in EMOTIONS:
                    adjusted[j][e] *= 1.8
    return adjusted

def compute_word_emotion_scores(segment_text, seg_probs, lexicon, lam=0.7):
    tokens = simple_tokenize(segment_text)

    lex_emotions = []
    for w in tokens:
        base = lexicon[w] if w in lexicon else {e: 0.0 for e in EMOTIONS}
        lex_emotions.append(dict(base))

    lex_adjusted = apply_rules(tokens, lex_emotions)

    fused = []
    for scores in lex_adjusted:
        out = {}
        for e in EMOTIONS:
            p_seg = seg_probs.get(e, 0.0)
            out[e] = max(
                0.0,
                (lam * scores[e]) + (1 - lam) * p_seg
            )
        fused.append(out)

    normed = []
    for d in fused:
        arr = np.array([max(v, 0.0) for v in d.values()])
        s = arr.sum()
        if s == 0:
            normed.append({e: 0.0 for e in EMOTIONS})
        else:
            normed.append({e: arr[i] / s for i, e in enumerate(EMOTIONS)})
    return tokens, normed

def aggregate_song_importance(song_segments, lexicon):
    """
    song_segments: list of dicts from predict_segments (must include "text" and "probs")
    returns: dict[emotion][word] -> score
    """
    song_importance = {e: defaultdict(float) for e in EMOTIONS}

    for seg in song_segments:
        seg_probs_list = seg["probs"]
        seg_probs = {e: seg_probs_list[i] for i, e in enumerate(EMOTIONS)}
        tokens, word_scores = compute_word_emotion_scores(seg["text"], seg_probs, lexicon)

        for w, scores in zip(tokens, word_scores):
            # skip generic or useless tokens
            if w in STOPWORDS:
                continue
            # only count words that appear in the lexicon at all:
            # if w not in lexicon: continue
            for e in EMOTIONS:
                song_importance[e][w] += scores[e]

    return song_importance
