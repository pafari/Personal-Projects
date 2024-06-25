# Personal-Projects
Just Started my project so making this initial commit

Design Notes(so far as of 6-24)

So far the predictive model is focused on the amount of times the artist has appeared as one of the weights and factors into deciding what songs to suggest.

The song suggestor function focuses on the artist_count and provides the song along with artist name. The random forest classifier was decided upon to not be overly conservative in my approach to this project. I wanted to generate several decision trees based on the data inputted. The hyperparameter search grid was just a sample length I chose based on the data. 

So basically lets break down what the hyperparameters are so u dont foreget like a dummy Pap: 


n_estimators = the number of trees(Increasing typically improves the performance but will increase the computational cost)

max_depth = the maximum depth of each tree (Controls the complexness of the trees and a deeper tree could capture more patterns but I could run the risk of the tree overfitting)

(all these next two do is effect and make higher values)

min samples_split = the minimum no of samples required to split an internal node. 

min_samples_leaf = the minimum no of samples to be at the leaf node.

The Cross validation part was actually pretty spontaneous was extremely confused why this was so common but similarly to when I did Search Engine and Decision Tree we need a technique to evaluate the model's performance. 
This involves splitting the data into multiple folds and training the model on some of the folds, and validating finally with the remaining folds. So this would help in assessing how the model is gonna generalize to new data. 

A More In Deph Implementation Breakdown: 

Step 1: We need to load the data and perform an EDA. So essentially I started by loading the excel file which contains information about different songs and a their status (whether it was added to my playlist already, saved for later, rejected, in review etc)

I needed to classify the yes or no's for the different statuses as binary values, to make it easier for the model to understand and process. 

Then just plotting the data to get a visual representation of what is going on with the data. 

Additional thing was addung an extra feature that counts the number of songs already added to my playlist or available in the overall dataset per artist. So it helps the model understand how popular or favorable the artist is to me and will influence whether the song gets added to the playlist. 

Step 2: Encoding

Utilizing One Hot encoding, you convert the categorical data like Artist names and Song titles and mae them numerical values. So each one should have a unique value that becomes a seperate column with binary values. 

Then I defined the target variable which is the overall goal of the project so the column should say if the song was added to the playlist or not. So this is the target variable the model should try and predict. 
In this process the original columns get dropped after they get encoding. 

Step 3: So Lets Train the model and Tune the Hyperparamters

Split the Daa into the training and testing sets. Then create a Random Forest Classifier which creates multiple deciison trees and combines their outputs. 
Set up the hyperparamater grid, grid search,and then trained the model based on the best hyperparamters found from Grid Search. 

Step 4: Model Evaluation and then it produces a recommendation

Evaluating the model on the test set to see how well it performs Accuracy and a detailed classification report. 

The trained model predicts the probabilities for songs that are in the "In Review" section of the data and suggest a list of songs to add to the playlist. 

Made a little bar plot at the end to visualize the recommeneded songs showing their artist counts. 





