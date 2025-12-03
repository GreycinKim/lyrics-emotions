import os
from pathlib import Path
from .config import (
    CSV_PATH,
    MODEL_DIR,
    LEXICON_PATH,
    OUTPUT_TIMELINES,
    OUTPUT_WORDCLOUDS,
    OUTPUT_SUMMARIES,
    EMOTIONS,
)
from .segments_from_csv import load_songs_and_segments_csv
from .run_inference import load_model, predict_segments
from .word_importance import load_lexicon, aggregate_song_importance
from .visualization import plot_emotion_timeline, emotion_wordcloud
from .narrative_llm import summarize_song
from collections import defaultdict

def run_pipeline():
    Path(OUTPUT_TIMELINES).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_WORDCLOUDS).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_SUMMARIES).mkdir(parents=True, exist_ok=True)

    # 1. Load songs + segments from CSV
    songs = load_songs_and_segments_csv(CSV_PATH, segment_mode="line")
    print(f"Loaded {len(songs)} songs from {CSV_PATH}")

    # GLOBAL: genre-emotion stats
    genre_emotion_counts = defaultdict(lambda: {e: 0 for e in EMOTIONS})
    genre_total_segments = defaultdict(int)

    # 2. Load classifier (directly from HuggingFace hub or local dir)
    tokenizer, model, device = load_model(MODEL_DIR)

    # 3. Load lexicon
    lexicon = load_lexicon(LEXICON_PATH)

    for song in songs:
        song_id = f"{song['artist_name']} - {song['song_name']}"
        print(f"Processing: {song_id}")

        segments = song["segments"]
        if not segments:
            print("  (no lyrics, skipping)")
            continue

        # Module 1: segment-level emotion
        seg_results = predict_segments(segments, tokenizer, model, device)
        song_result = {
            "artist_name": song["artist_name"],
            "song_name": song["song_name"],
            "genres": song["genres"],
            "language": song["language"],
            "artist_popularity": song["artist_popularity"],
            "new_artist_popularity": song["new_artist_popularity"],
            "segments": seg_results
        }

        # split multi-genre string like "Pop; Axé; Romântico"
        raw_genres = song["genres"] or ""
        song_genres = [g.strip() for g in raw_genres.split(";") if g.strip()]
        if not song_genres:
            song_genres = ["(unknown)"]

        for seg in seg_results:
            label = seg["label"]
            for g in song_genres:
                genre_emotion_counts[g][label] += 1
                genre_total_segments[g] += 1

        # After processing all songs, plot genre-emotion bubble map
        from .visualization import plot_genre_emotion_bubble

        print("Building genre-emotion bubble map...")
        plot_genre_emotion_bubble(
            genre_emotion_counts,
            genre_total_segments,
            OUTPUT_TIMELINES,
            top_n=25  # max genres to show
        )
        print("Genre-emotion bubble saved.")

        # Plot timeline
        plot_emotion_timeline(song_result, OUTPUT_TIMELINES)

        # Module 2: word-level importance + word clouds
        song_importance = aggregate_song_importance(seg_results, lexicon)
        for e in EMOTIONS:
            emotion_wordcloud(song_importance, e, OUTPUT_WORDCLOUDS, song_id)

        # Module 3: narrative summary
        summary = summarize_song(song_id, seg_results, song_importance)
        safe_id = song_id.replace(" ", "_")
        summary_path = os.path.join(OUTPUT_SUMMARIES, f"{safe_id}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print("  Summary written to:", summary_path)

if __name__ == "__main__":
    run_pipeline()
