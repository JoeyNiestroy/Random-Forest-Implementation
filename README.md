# Random-Forest-Implementation
Python random forest for regressions only, built using split criteria SSE and only uses data from Pandas Dataframe. Also includes comparison to ScikitLearn.  
The following project includes 4 files: Main.py, Tree.py, Random_Forest_model.py, test_data.csv
Tree.py is my build of a basic decson tree using SSE as splitting criteria and uses a greedy algorithm for variable selection for each node.
RandomForest.py is a random forest built on the descion tree, includes hyperparamters: numoftrees, max_depth, threashold, min_nodes(See code notes for further description).
Main.py is a simple side by side comparison of R^2 scores on test data for both my implementation and SckitLearn's.
test_data.csv is data for linear regressions acquired from https://www.kaggle.com/datasets/hellbuoy/car-price-prediction?resource=download . 
Conclusion: Average R^2 on clean test data for my implemention with preset hyperparameters is roughly .74-.86. Sckitlearn with similar hyperparameters is around .73-.76 (Note: SKlean does not offer SSE as split criteria). However the average time to build the model in Scikitlearn is 4 seconds, while my model takes around 8 seconds. I belive the reason for this differnce is the current lack of parrell tree creation in my implentation.  

Future changes: In the future I plan to add hyperparameters for splitting crteria as well as using the model for classification. 
