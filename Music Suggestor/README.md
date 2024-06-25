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

