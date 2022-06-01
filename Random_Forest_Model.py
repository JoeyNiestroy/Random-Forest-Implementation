from Tree import DecisionTree
import math
import random
import pandas as pd

class RandomForest():
    #Data should be a Pandas Dataframe
    #explain_var should a list of column names that are the explainitory variables 
    #out_put should entered as a string and is the column name of the predicted variable
    #numoftrees is an int for the number of trees
    #max depth is the maxium depth used in prediction
    #threashold_split is an int that is the threashold decrease of parent's SSE. Ex: .01 means a required 1% decrease from parents SSE for split to occur at that node
    def __init__(self, data, explain_vars, out_put, numberoftrees = 10, max_depth = 8, threashold_split = 0.01, min_nodes_for_split = 4):
        self.__data = data
        self.__min_nodes_for_split = min_nodes_for_split
        self.__num_trees = numberoftrees
        self.__x_var = explain_vars
        self.__y_var = out_put
        self.__depth = max_depth
        self.__threashold = threashold_split
        self.__forest = self.__generate_forest()

    #Function to generate forest based on hyperparameters and it called at when class is created
    def __generate_forest(self):
        forest_array = []
        for _ in range(0, self.__num_trees):
            bootstrapped = self.__data.sample(replace = True, frac = .33)
            bootstrapped = bootstrapped.reset_index(drop = True)
            x_var = self.__sqrt_random_selection()
            tree = DecisionTree(bootstrapped,x_var,self.__y_var, max_height= self.__depth, threashold= self.__threashold, min_nodes= self.__min_nodes_for_split)
            forest_array.append(tree)
        return forest_array
#Non private function to predict value for a passed pandas row object
    def predict_value_for_row(self, row):
        sum = 0
        for tree in self.__forest:
            prediction = tree.predict_value(row, tree.root)
            sum = sum + prediction
        return sum/self.__num_trees

#Private function that randomly selects variables for each tree, uses sqrt(number of variables). 
    def __sqrt_random_selection(self):
        num = round(math.sqrt(len(self.__x_var)))
        return random.sample(self.__x_var, num)
#Returns Int of r_sqaured score of the orginal data entered in forest
    def r_squared_forest(self):
        return 1 - (self.__SSR()/self.__SST())
#Private function that calculates SSR   
    def __SSR(self):
        sum = 0
        for index in self.__data.index:
            sum =  sum + (((self.__data[self.__y_var][index]) - (self.predict_value_for_row(self.__data.iloc[index])))**2)
        return sum
#Private function that calculates SST 
    def __SST(self):
        sum = 0
        mean = (self.__data[self.__y_var]).mean()
        for index in self.__data.index:
            sum = sum + (((self.__data[self.__y_var][index])-mean)**2)
        return sum
#Function that calculates R^2 for test data that is passed in as pandas dataframe. (Requires columns to be same as dataframe built on model)
    def r_squared_for_test_data(self, data):
        data = data.reset_index(drop = True)
        return 1 - (self.__SSR_test(data)/self.__SST_test(data))
#Private function that calculates SSR for test data    
    def __SSR_test(self, data_SSR):
        sum = 0
        for index in data_SSR.index:
            sum =  sum + (((data_SSR[self.__y_var][index]) - (self.predict_value_for_row(data_SSR.iloc[index])))**2)
        return sum
#Private function that calculates SST for test data   
    def __SST_test(self, data):
        sum = 0
        mean = (data[self.__y_var]).mean()
        for index in data.index:
            sum = sum + (((data[self.__y_var][index])-mean)**2)
        return sum
if __name__ == '__main__':
    df = pd.read_csv("test_data.csv")
    train = df[0:183]
    test = df[183:]
    x = ["highwaympg","horsepower","citympg", "stroke", "carlength","carwidth", "carheight","curbweight", "enginesize", "peakrpm"] 
    y = "price"
    model = RandomForest(train,x,y, numberoftrees= 135, max_depth= 12, threashold_split= .01)
    print(model.r_squared_for_test_data(test))
