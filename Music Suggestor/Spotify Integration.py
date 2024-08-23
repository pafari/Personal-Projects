# import os
# import pandas as pd
# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

# # Load environment variables
# client_id = os.getenv('SPOTIPY_CLIENT_ID')
# client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

# # Authenticate with Spotify
# sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# def get_audio_features(track_ids):
#     features_list = []
#     for track_id in track_ids:
#         features = sp.audio_features(track_id)
#         if features:
#             features_list.append(features[0])
#     return pd.DataFrame(features_list)

# def get_track_ids(sp, data):
#     track_ids = []
#     for _, row in data.iterrows():
#         query = f"{row['Song']} {row['Artist']}"
#         result = sp.search(query, type='track', limit=1)
#         if result['tracks']['items']:
#             track_ids.append(result['tracks']['items'][0]['id'])
#         else:
#             track_ids.append(None)
#     return track_ids

# # Load the manually created dataset
# data = pd.read_excel('Liked Songs Playlist Contenders Manual Data.xlsx', header=1)

# # Fetch the track IDs from Spotify
# data['Track_ID'] = get_track_ids(sp, data)

# # Remove rows where Track_ID is None
# data = data.dropna(subset=['Track_ID'])

# # Fetch the audio features from Spotify
# audio_features = get_audio_features(data['Track_ID'])

# # Combine the original data with audio features
# full_data = pd.concat([data.reset_index(drop=True), audio_features.reset_index(drop=True)], axis=1)

# # Save the full data to an Excel file
# full_data.to_excel('Kpop Songs with Features.xlsx', index=False)
# print("Spotify features fetched and saved to 'Kpop Songs with Features.xlsx'")