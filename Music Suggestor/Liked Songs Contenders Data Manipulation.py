import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import  train_test_split, RandomizedSearchCV
from scipy.stats import randint
# Tree Visualization
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

'''Step 1: Data Preparation and EDA: I am excited to be working on something
that is actually useful for my everyday life. Since I spent too much time looking at my google document trying to know which songs to add to my playlist
it is really rewarding to see that will be definitely made significantly easier starting today'''

# Loading the original data that I manually made
data = pd.read_excel('Corrected_Liked_Songs_Playlist_Contenders_Data.xlsx')

# Function to casefold strings for case-insensitive comparison
def casefold_series(series):
    return series.apply(lambda x: x.casefold() if isinstance(x, str) else x)

# Convert categorical columns to binary values
data['Added (Gray)'] = data['Added (Gray)'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Saved for Later(Yellow)'] = data['Saved for Later(Yellow)'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Added to Alternate (Green)'] = data['Added to Alternate (Green)'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Rejected'] = data['Rejected'].apply(lambda x: 1 if x == 'Yes' else 0)
data['In Review (has not been added)'] = data['In Review (has not been added)'].apply(lambda x: 1 if x == 'Yes' else 0)

# Add a feature for the count of songs by each artist
artist_counts = casefold_series(data['Artist']).value_counts()
data['Artist_Count'] = data['Artist'].map(artist_counts)


''' Step 2: Feature Encoding:
So essentially what we are doing here is first One_Hot Encoding which is
converting the categories' values into numerical ones. And  then we want to add an artist count feature into the dataset
Finally we should define a target variable. That simply means if a song was added to the playlist or not. '''

# Convert categorical features to numerical values using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Artist', 'Song'])

# Add the Artist Count feature to the encoded data
data_encoded['Artist_Count'] = data['Artist_Count']* 10 # Just scaling it so it is more important but this may serve to only be 
#Temporary fix for the actual code not working properly


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
#from sklearn.model_selection import train_test_split, GridSearchCV

# Split the data into training and testing sets
X = data_encoded.drop(columns=['Added'], axis = 1)
y = data_encoded['Added']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

model = RandomForestClassifier()

model_search = RandomizedSearchCV(model,
                                  param_distributions= param_dist,
                                  n_iter=100,
                                  cv =5,
                                  verbose =2, n_jobs=-1)


#Fit the model
model_search.fit(X_train, y_train)

# Create a variable for the best model
best_model = model_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:', model_search.best_params_)

# Evaluating to see if the forest is making accurate predictions
y_pred = best_model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Check feature importance
importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print(importance_df)

# Recommend new songs to add to the playlist
def recommend_songs(model, X_test, original_data, y_test, n=10):
    # Predict on the data
    predictions = model.predict(X_test)
    
    test_data = original_data.loc[X_test.index]
    
    # Identify the indices of songs to recommend
    recommended_indices = test_data[(predictions == 1) & (y_test == 0)].index
    
    # Return the top N recommended songs using the original data to get the artist and song names
    return original_data.loc[recommended_indices, ['Artist_Count', 'Artist', 'Song']].head(n)

# Applying the recommendation function
recommendations = recommend_songs(best_model, X_test, data, y_test)
print("Recommended Songs to Add:")
print(recommendations)

# Export the first three decision trees from the forest
for i in range(3):
    tree = best_model.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.view()


