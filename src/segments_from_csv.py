from typing import List, Dict
import pandas as pd
from .load_songs_from_csv import load_song_df
from .encoding_utils import fix_mojibake

def segment_lyrics(lyrics: str, mode: str = "line") -> List[str]:
    """
    Split lyrics into segments.
    mode="line"   -> each non-empty line
    mode="stanza" -> blocks separated by blank lines
    """
    if not isinstance(lyrics, str):
        return []

    lyrics = fix_mojibake(lyrics)

    text = lyrics.replace("\r\n", "\n").replace("\r", "\n")

    if mode == "stanza":
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        return blocks
    else:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return lines

def load_songs_and_segments_csv(path: str, segment_mode: str = "line") -> List[Dict]:
    """
    Returns list of songs with segmented lyrics:
    {
      "artist_name": ...,
      "song_name": ...,
      "genres": ...,
      "language": ...,
      "artist_popularity": ...,
      "new_artist_popularity": ...,
      "segments": ["line 1", "line 2", ...]
    }
    """
    df: pd.DataFrame = load_song_df(path)

    songs = []
    for _, row in df.iterrows():
        segments = segment_lyrics(row["lyrics"], mode=segment_mode)
        songs.append({
            "artist_name": row["artist_name"],
            "song_name": row["song_name"],
            "genres": fix_mojibake(row.get("genres", "")),
            "language": row.get("language", None),
            "artist_popularity": row.get("artist_popularity", None),
            "new_artist_popularity": row.get("new_artist_popularity", None),
            "segments": segments
        })
    return songs

if __name__ == "__main__":
    songs = load_songs_and_segments_csv("data/songs.csv", segment_mode="line")
    print(f"Loaded {len(songs)} songs\n")
    for s in songs:
        print("SONG:", s["artist_name"], "-", s["song_name"])
        print("Genres:", s["genres"])
        print("First 8 segments:")
        for seg in s["segments"][:8]:
            print("  •", seg)
        print("-" * 40)
