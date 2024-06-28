import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

def get_spotify_client():
    client_id = os.getenv('SPOTIPY_CLIENT_ID')
    client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError("Spotify client ID or client secret not found in environment variables.")
    
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
    return sp

def get_track_ids(sp, data, limit=10):
    track_ids = []
    for idx, row in data.iterrows():
        if idx >= limit:
            break
        query = f"{row['Song']} {row['Artist']}"
        print(f"Searching for: {query}")
        while True:
            try:
                result = sp.search(query, type='track', limit=1)
                break
            except spotipy.SpotifyException as e:
                print(f"Rate limit reached. Sleeping for 60 seconds. Error: {e}")
                time.sleep(60)
        if result['tracks']['items']:
            track_ids.append(result['tracks']['items'][0]['id'])
        else:
            track_ids.append(None)
        time.sleep(1)  # Add a delay to avoid hitting rate limits
    return track_ids

def get_audio_features(sp, track_ids):
    features_list = []
    for track_id in track_ids:
        if track_id:
            while True:
                try:
                    features = sp.audio_features(track_id)
                    break
                except spotipy.SpotifyException as e:
                    print(f"Rate limit reached. Sleeping for 60 seconds. Error: {e}")
                    time.sleep(60)
            if features:
                features_list.append(features[0])
            else:
                features_list.append({})
        else:
            features_list.append({})
    return pd.DataFrame(features_list)

# Example usage:
if __name__ == "__main__":
    sp = get_spotify_client()
    data = pd.read_excel('Liked Songs Playlist Contenders Manual Data.xlsx', header=1)
    print("Fetching track IDs...")
    track_ids = get_track_ids(sp, data)
    print("Fetching audio features...")
    audio_features = get_audio_features(sp, track_ids)
    print(audio_features.head())
