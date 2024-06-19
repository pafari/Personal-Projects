import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

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
