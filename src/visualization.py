import os
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .config import EMOTIONS

# Colors for each emotion (used in both timeline + bubble map)
EMOTION_COLORS = {
    "joy": "yellow",
    "sadness": "blue",
    "anger": "red",
    "love": "magenta",
    "neutral": "gray",
    "fear": "purple",
    "surprise": "orange",
}

def plot_emotion_timeline(song_result, out_dir: str):
    """
    Per-song emotion timeline:
      x-axis: segment index
      y-axis: emotion (categorical)
      color: emotion color
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    labels = [seg["label"] for seg in song_result["segments"]]
    x = list(range(1, len(labels) + 1))

    # Map label to Y index
    label_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    y = [label_to_idx[l] for l in labels]
    colors = [EMOTION_COLORS.get(l, "black") for l in labels]

    plt.figure(figsize=(12, 3))
    plt.scatter(x, y, c=colors)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel("Segment index")
    plt.title(f"Emotion timeline: {song_result['artist_name']} - {song_result['song_name']}")
    plt.tight_layout()

    safe_name = f"{song_result['artist_name']}_{song_result['song_name']}".replace(" ", "_")
    out_path = os.path.join(out_dir, f"{safe_name}_timeline.png")
    plt.savefig(out_path)
    plt.close()

def emotion_wordcloud(song_importance, emotion, out_dir: str, song_id: str, max_words=40):
    """
    Build a word cloud for one emotion for a given song.
      song_importance: dict[emotion][word] -> score
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    freq_dict = song_importance.get(emotion, {})
    if not freq_dict:
        return

    # limit to top max_words
    sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:max_words]
    freq_top = {w: v for w, v in sorted_items}

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(freq_top)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{emotion.title()} words: {song_id}")
    plt.tight_layout()

    safe_id = song_id.replace(" ", "_")
    out_path = os.path.join(out_dir, f"{safe_id}_{emotion}_wordcloud.png")
    plt.savefig(out_path)
    plt.close()

def plot_genre_emotion_bubble(genre_emotion_counts, genre_total_segments, out_dir: str, top_n: int = 20):
    """
    Create a bubble map:
      x-axis: emotion
      y-axis: genre
      bubble size: percentage of segments in that genre that are that emotion
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Sort genres by total segments (most data at the top)
    sorted_genres = sorted(
        genre_total_segments.items(),
        key=lambda kv: kv[1],
        reverse=True
    )
    if top_n is not None:
        sorted_genres = sorted_genres[:top_n]

    genres = [g for g, _ in sorted_genres]
    if not genres:
        print("No genre data available for bubble map.")
        return

    x_vals = []
    y_vals = []
    sizes = []
    colors = []

    # Map emotions to x positions
    emotion_to_x = {e: i for i, e in enumerate(EMOTIONS)}

    for gi, g in enumerate(genres):
        total = genre_total_segments[g]
        if total == 0:
            continue

        counts = genre_emotion_counts[g]
        for e in EMOTIONS:
            c = counts.get(e, 0)
            if c == 0:
                continue

            pct = c / total  # 0..1
            x_vals.append(emotion_to_x[e])
            y_vals.append(gi)
            # scale bubble area; tweak factor to taste
            sizes.append(pct * 2000.0)
            colors.append(EMOTION_COLORS.get(e, "black"))

    if not x_vals:
        print("No non-zero genre-emotion combinations to plot.")
        return

    plt.figure(figsize=(12, max(5, len(genres) * 0.4)))
    plt.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7)

    plt.xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45)
    plt.yticks(range(len(genres)), genres)
    plt.xlabel("Emotion")
    plt.ylabel("Genre")
    plt.title("Genre–Emotion Bubble Map\n(bubble size = % of segments in that genre with that emotion)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "genre_emotion_bubble.png")
    plt.savefig(out_path)
    plt.close()
