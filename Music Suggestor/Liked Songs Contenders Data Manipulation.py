import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split

'''Step 1: Data Preparation and EDA: I am excited to be working on something
that is actually useful for my everyday life. Since I spent too much time looking at my google document trying to know which songs to add to my playlist
it is really rewarding to see that will be definitely made significantly easier starting today'''

# Loading the original data that I manually made
data = pd.read_excel('Liked Songs Playlist Contenders Manual Data.xlsx', header=1)

# Convert categorical columns to binary values
data['Added (Gray)'] = data['Added (Gray)'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Saved for Later (Yellow)'] = data['Saved for Later(Yellow)'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Added to Alternate (Green)'] = data['Added to Alternate (Green)'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Rejected'] = data['Rejected'].apply(lambda x: 1 if x == 'Yes' else 0)
data['In Review (has not been added)'] = data['In Review (has not been added)'].apply(lambda x: 1 if x == 'Yes' else 0)

# Basic Data Analysis to get a visual read on the data

#pd.display_dataframe_to_user(name="Cleaned Data for Kpop Songs Project", dataframe=data)
print(data.head())
#print(data.describe())

# Analyze the distribution of songs in each category
added_count = data['Added (Gray)'].value_counts()
saved_for_later_count = data['Saved for Later(Yellow)'].value_counts()
rejected_count = data['Rejected'].value_counts()
in_review_count = data['In Review (has not been added)'].value_counts()

# Create a bar plot to visualize the distribution
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
if not added_count.empty:
    added_count.plot(kind='bar', color='gray', title='Added (Gray)')
    plt.xlabel('Status')
    plt.ylabel('Count')

plt.subplot(2, 2, 2)
if not saved_for_later_count.empty:
    saved_for_later_count.plot(kind='bar', color='yellow', title='Saved for Later (Yellow)')
    plt.xlabel('Status')
    plt.ylabel('Count')

plt.subplot(2, 2, 3)
if not rejected_count.empty:
    rejected_count.plot(kind='bar', color='red', title='Rejected')
    plt.xlabel('Status')
    plt.ylabel('Count')

plt.subplot(2, 2, 4)
if not in_review_count.empty:
    in_review_count.plot(kind='bar', color='blue', title='In Review (has not been added)')
    plt.xlabel('Status')
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Count songs by artist
artist_count = data['Artist'].value_counts().head(10)  # Top 10 artists

plt.figure(figsize=(10, 6))
artist_count.plot(kind='bar')
plt.title('Top 10 Artists by Number of Songs')
plt.xlabel('Artist')
plt.ylabel('Number of Songs')
plt.show()




''' Step 2: Predidctive Model'''

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Authenticate using Spotify's client credentials
client_credentials_manager = SpotifyClientCredentials(client_id='cefb834435394e8c826c8f84ed76c3bd', client_secret='6ba790be451547d4a45fe635abe006ae')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to get song attributes
def get_song_attributes(artist, track):
    results = sp.search(q=f'artist:{artist} track:{track}', type='track')
    if results['tracks']['items']:
        track_id = results['tracks']['items'][0]['id']
        attributes = sp.audio_features(track_id)[0]
        return attributes
    return None

# Example usage
attributes = get_song_attributes('BTS', 'Dynamite')
print(attributes)





# Convert categorical features to numerical values using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Artist', 'Song'])

# Define the target variable (whether the song is added or not)
data_encoded['Added'] = data_encoded['Added (Gray)'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop the original categorical columns
data_encoded = data_encoded.drop(columns=['Added (Gray)', 'Saved for Later(Yellow)', 'Date Added to Playlist',
                                          'Added to Alternate (Green)', 'Rejected', 'In Review (has not been added)'])

# Split the data into training and testing sets
X = data_encoded.drop(columns=['Added'])
y = data_encoded['Added']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print(report)

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Score: {best_score:.2f}')
