
import numpy as np
import pandas as pd 




class DecisionTree():
    class Node():
        #Creates Node for Tree
        def __init__(self, data):
            self.value = None
            self.data = data
            self.var = None
            self.lc = None
            self.rc = None
            self.index = 0
            
            
    """Data should be a Pandas Dataframe
    x_data should a list of column names that are the explainitory variables 
    y_data should entered as a string and is the column name of the predicted variable
    max height is the maxium depth used in prediction
    threashold is an int that is the threashold decrease of parent's SSE. Ex: .01 means a required 1% decrease from parents SSE for split to occur at that node"""
           
    def __init__(self, data, x_data, y_data, max_height = None, threashold = 0, min_nodes = 4): 
        self.min_nodes = min_nodes
        self.tree_data = data
        self.x_data = x_data
        self.y_data = y_data
        self.threashold = threashold
        self.max_h = max_height
        self.tree_array = np.array([None]*self.__determine_array())
        self.node = DecisionTree.Node(data)
        self.tree_array[0] = self.node
        self.root = self.__constructor_function(self.node)
    
    """"Determines Max size possible for decsion tree array"""
    def __determine_array(self):
        return (2**self.max_h)-1
        


    """function that build decsion tree when class is created
    Conditional manages number of data points to split nodes and threashold
    Recursivly builds tree"""
    def __constructor_function(self, t):
        if len(t.data) > self.min_nodes:
            t.value, t.var = self.determine_theta(t)
            if t.value is None or t.var is None:
                return t
            left_data, right_data = self.__split_data(t)
            left_data = left_data.reset_index(drop = True)
            right_data = right_data.reset_index(drop = True)
            left_child =  DecisionTree.Node(left_data)
            right_child =  DecisionTree.Node(right_data)
            left_child.index = (t.index*2)+1
            right_child.index = (t.index*2)+2
            if left_child.index < len(self.tree_array):
                self.tree_array[left_child.index] = left_child
                self.__constructor_function(left_child)
            if right_child.index < len(self.tree_array):
                self.tree_array[right_child.index] = right_child
                self.__constructor_function(right_child)
        
    
    
    
    """Way to long function to determine node value TODO make seprate functions for determine_theta
    returns value to split node and the var it uses to split at""" 
    def determine_theta(self, t):
        low_theta = None
        low_SSE = None
        final_index = None
        data_set = t.data
        #Loop for greedy algorthim to determine best var to split at 
        for index_value in range(len(self.x_data)):
            index_x = index_value
            #Loop to find best value in current var to split at
            for index in data_set.index:
                starter = data_set[self.x_data[index_x]][index]
                left = []
                right = []
                #print(data_set.index)
                for index in data_set.index:
                    #print(index)
                    if data_set[self.x_data[index_x]][index] <= starter:
                        left.append(index)
                    else:
                        right.append(index)
                if len(left)<0 or len(right)<0:
                    pass
                else:
                    left_data = data_set.iloc[left]

                    right_data = data_set.iloc[right]

                    SSE_right = self.__SSE(left_data)
                    SSE_left = self.__SSE(right_data)
                    final_SSE = (SSE_left+SSE_right)
                    if final_SSE >= (self.__SSE(data_set)*(1-self.threashold)):
                        break
                    if low_theta is  None:
                        low_theta = starter
                        low_SSE = final_SSE
                        final_index = index_x
                    else:
                        if final_SSE < low_SSE:
                            low_theta = starter
                            low_SSE = final_SSE
                            final_index = index_x
                        else:
                            pass
            
        return low_theta, final_index

    """Called from constructor function and returns data to be entered to left and right child nodes  """  
    def __split_data(self, t):
        left = []
        right = []
        for index in t.data.index:
            #print(index)
            if t.data[self.x_data[t.var]][index] <= t.value:
                left.append(index)
            else:
                right.append(index)
        left = t.data.iloc[left]
        right = t.data.iloc[right]
        return left, right
    

    """Public function that predicts value for entered row, requires root to be entered as function. built recursivly"""    
    def predict_value(self, row, index = 0):
        t = self.tree_array[index]
        if t.var is None or t.index*2+1 > len(self.tree_array) or t.index*2+2 > len(self.tree_array):
            est = (int((t.data[self.y_data]).mean()))
            return est
        if row[self.x_data[t.var]] <= t.value:
                return self.predict_value(row, index = ((t.index*2)+1))
        else:
                return self.predict_value(row, index = ((t.index*2)+2))
    
    
    """Function to return r_squared for the decsion tree orginal data   """ 
    def r_squared(self):
        return 1 - (self.__SSR()/self.__SST())
    """Calculates SSR"""    
    def __SSR(self):
        sum = 0
        for index in self.tree_data.index:
            sum =  sum + (((self.tree_data[self.y_data][index]) - (self.predict_value(self.tree_data.iloc[index], self.root)))**2)
        return sum
    """Calculates SST"""
    def __SST(self):
        sum = 0
        mean = (self.tree_data[self.y_data]).mean()
        for index in self.tree_data.index:
            sum = sum + (((self.tree_data[self.y_data][index])-mean)**2)
        return sum
    """Calculates SSE used in determine theta function"""   
    def __SSE(self, data):
        sum = 0
        mean = (data[self.y_data]).mean()
        for index in data.index:
            sum = sum + (((data[self.y_data][index])-mean)**2)
        return sum

   
           

  


if __name__ == '__main__':
    df = pd.read_csv("test_data.csv")
    x = ["highwaympg","horsepower","citympg", "stroke", "carlength","carwidth", "carheight","curbweight", "enginesize", "peakrpm"] 
    y = "price"
    tester_tree = DecisionTree(df, x, y, max_height= 4, min_nodes= 6)
    print(df.iloc[80])
    print(tester_tree.predict_value(df.iloc[80]))
    #print(tester_tree.tree_array)



    


   




    


   


