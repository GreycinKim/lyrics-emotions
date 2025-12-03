import pandas as pd

def load_song_df(path: str) -> pd.DataFrame:
    """
    Load CSV containing:
    artist_name, song_name, genres, language, lyrics,
    artist_popularity, new_artist_popularity
    """
    df = pd.read_csv(path)  # standard CSV
    return df

if __name__ == "__main__":
    df = load_song_df("data/songs.csv")
    print("Columns:", df.columns.tolist())
    print("First artist:", df.iloc[0]["artist_name"])
    print("First song:", df.iloc[0]["song_name"])
    print("First 200 chars of lyrics:")
    print(df.iloc[0]["lyrics"][:200], "...")
