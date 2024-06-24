import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

'''Step 1: Data Preparation and EDA: I am excited to be working on something
that is actually useful for my everyday life. Since I spent too much time looking at my google document trying to know which songs to add to my playlist
it is really rewarding to see that will be definitely made significantly easier starting today'''

# Loading the original data that I manually made
data = pd.read_excel('Liked Songs Playlist Contenders Manual Data.xlsx', header=1)

# Convert categorical columns to binary values
data['Added (Gray)'] = data['Added (Gray)'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Saved for Later(Yellow)'] = data['Saved for Later(Yellow)'].apply(lambda x: 1 if x == 'Yes' else 0)
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

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
if not added_count.empty:
    added_count.plot(kind='bar', color='gray', title='Added (Gray)')
    plt.xlabel('Status')
    plt.ylabel('Count')

plt.subplot(2, 2, 2)
if not saved_for_later_count.empty:
    saved_for_later_count.plot(kind='bar', color='yellow', title='Saved for Later(Yellow)')
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

# Add a feature for the count of songs by each artist
artist_counts = data['Artist'].value_counts()
data['Artist_Count'] = data['Artist'].map(artist_counts)

# Count songs by artist
# Distribution by Artist
artist_count = data['Artist'].value_counts().head(10)  # Top 10 artists

plt.figure(figsize=(10, 6))
artist_count.plot(kind='bar')
plt.title('Top 10 Artists by Number of Songs')
plt.xlabel('Artist')
plt.ylabel('Number of Songs')
plt.show()

# Number of Songs in Each Category
category_count = {
    'Added': data['Added (Gray)'].value_counts().get(1, 0),
    'Saved for Later': data['Saved for Later(Yellow)'].sum(),
    'Rejected': data['Rejected'].sum(),
    'In Review': data['In Review (has not been added)'].sum()
}
plt.figure(figsize=(10, 6))
plt.bar(category_count.keys(), category_count.values(), color=['gray', 'yellow', 'red', 'blue'])
plt.title('Number of Songs in Each Category')
plt.xlabel('Category')
plt.ylabel('Number of Songs')
plt.show()



''' Step 2: Feature Encoding:
So essentially what we are doing here is first One_Hot Encoding which is
converting the categories' values into numerical ones. And  then we want to add an artist count feature into the dataset
Finally we should define a target variable. That simply means if a song was added to the playlist or not. '''

# Convert categorical features to numerical values using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Artist', 'Song'])

# Add the Artist Count feature to the encoded data
data_encoded['Artist_Count'] = data['Artist_Count']

# Define the target variable (whether the song is added or not)
data_encoded['Added'] = data_encoded['Added (Gray)'].apply(lambda x: 1 if x == 1 else 0)


# Drop the original categorical columns
data_encoded = data_encoded.drop(columns=['Added (Gray)', 'Saved for Later(Yellow)', 'Date Added to Playlist',
                                          'Added to Alternate (Green)', 'Rejected', 'In Review (has not been added)'])

'''Step 3: Predictive Model Trainijng and Tuning the Hyperparamters:
So the overall goal is to train a Random Forest Model(which will be conveniently called an RFM from now on)
There are 3 crucial steps:
Splitting the data into a training and testing set (parellels to the search project)
Initializing and setting up the Random Forest Classifier and 
Tuning the hyperparameters which is the biggest struggle of alL: Utilizing grid search to find the best set of parameters. '''
from sklearn.model_selection import train_test_split, GridSearchCV

# Split the data into training and testing sets
X = data_encoded.drop(columns=['Added'])
y = data_encoded['Added']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

#Defining the parameter grid
param_grid ={
    'n_estimators' : [100,200, 300],
    'max_depth' : [None, 10, 20, 30],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
}

#Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

#Fit the model
grid_search.fit(X_train, y_train)

# Get the optimal hyperparameters and cross-validation accuracy
optimal_hyperparameters = grid_search.best_params_
cross_validation_accuracy = grid_search.best_score_

print(f'Optimal Hyperparameters: {optimal_hyperparameters}')
print(f'Cross-Validation Accuracy: {cross_validation_accuracy:.2f}')


# Evaluate the model on the test set
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Test Set Accuracy: {accuracy:.2f}')
print(report)

def recommend_songs(model, data, n_recommendations=5):
    # Filter songs that are 'In Review'
    in_review_songs = data[data['In Review (has not been added)'] == 1]
    if in_review_songs.empty:
        print("No songs are currently in review.")
        return []

    # Encode the 'In Review' songs
    in_review_encoded = pd.get_dummies(in_review_songs, columns=['Artist', 'Song'])
    in_review_encoded = in_review_encoded.reindex(columns=X.columns, fill_value=0)  # Align with the training data columns
    in_review_encoded['Artist_Count'] = in_review_encoded['Artist_Count']

    # Predict probabilities for 'In Review' songs
    probabilities = model.predict_proba(in_review_encoded)[:, 1]

    # Select top N recommendations
    recommendations = in_review_songs.iloc[probabilities.argsort()[-n_recommendations:]]
    return recommendations[['Artist', 'Song', 'Artist_Count']]

# Get top 5 song recommendations
recommended_songs = recommend_songs(best_model, data, n_recommendations=5)
print("Recommended Songs:")
print(recommended_songs)

# Visualize the recommended songs
plt.figure(figsize=(10, 6))
sns.barplot(data=recommended_songs, x='Artist_Count', y='Song', hue='Artist', dodge=False)
plt.title('Top Song Recommendations')
plt.xlabel('Artist Count')
plt.ylabel('Song')
plt.savefig('recommended_songs.png')

