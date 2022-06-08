
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
        self.np_data = self.tree_data.to_numpy()
        self.x_data = x_data
        self.y_data = y_data
        self.threashold = threashold
        self.max_h = max_height
        self.dic = self.__create_dic()
        self.tree_array = np.array([None]*self.__determine_array())
        self.node = DecisionTree.Node(self.np_data)
        self.tree_array[0] = self.node
        self.root = self.__constructor_function(self.node)

    """Function to map column names to their index in Array"""
    def __create_dic(self):
        col = self.x_data
        dic = {}
        for column in col:
            dic[column] = self.tree_data.columns.get_loc(column)
        dic[self.y_data] = self.tree_data.columns.get_loc(self.y_data)
        return dic
    
    """"Determines Max size possible for decsion tree array"""
    def __determine_array(self):
        return (2**self.max_h)-1
        


    """function that build decsion tree when class is created
    Conditionals manage number of data points to split nodes, threshold, and height 
    Recursivly builds tree"""
    def __constructor_function(self, t):
        if len(t.data) > self.min_nodes:
            t.value, t.var = self.determine_theta(t)
            if t.value is None or t.var is None:
                return t
            left_data, right_data = self.__split_data(t)
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
        """Loop for greedy algorthim to determine best var to split at"""
        for index_value in self.x_data:
            index_x = self.dic[index_value]
            """Loop to find best value in current var to split at"""
            for row in data_set:
                starter = row[index_x]
                left = data_set[np.where(data_set[:,index_x]<= starter)]
                right = data_set[np.where(data_set[:,index_x]> starter)]
                SSE_right = self.__SSE(right)
                SSE_left = self.__SSE(left)
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
        left = t.data[np.where(t.data[:,t.var]<= t.value)]
        right = t.data[np.where(t.data[:,t.var]> t.value)]
        return left, right
    

    """Public function that predicts value for entered row, built recursivly. Row must be numpy 1d array"""    
    def predict_value(self, row, index = 0):
        t = self.tree_array[index]
        if t.var is None or t.index*2+1 > len(self.tree_array) or t.index*2+2 > len(self.tree_array):
            return np.mean(t.data[:,self.dic[self.y_data]])
        if row[t.var] <= t.value:
                return self.predict_value(row, index = ((t.index*2)+1))
        else:
                return self.predict_value(row, index = ((t.index*2)+2))
    
    
    """Function to return r_squared for the decsion tree orginal data   """ 
    def r_squared(self):
        return 1 - (self.__SSR()/self.__SST())
    """Calculates SSR"""    
    def __SSR(self):
        sum = 0
        for row in self.np_data:
            sum =  sum + (((row[self.dic[self.y_data]]) - (self.predict_value(row)))**2)
        return sum
    """Calculates SST"""
    def __SST(self):
        mean = np.mean(self.np_data[:,self.dic[self.y_data]])
        y_array = self.np_data[:,self.dic[self.y_data]]
        square = ((y_array-mean)**2).sum()
        return square
    """Calculates SSE used in determine theta function"""   
    def __SSE(self, data):
        """Conditional to avoid RuntimeWarning"""
        if len(data) > 0:
            mean = np.mean(data[:,self.dic[self.y_data]])
        else:
            mean = 0
        y_array = data[:,self.dic[self.y_data]]
        square = ((y_array-mean)**2).sum()
        return square

   
           

  


if __name__ == '__main__':
    df = pd.read_csv("test_data.csv")
    x = ["highwaympg","horsepower","citympg", "stroke", "carlength","carwidth", "carheight","curbweight", "enginesize", "peakrpm"] 
    y = "price"
    tester_tree = DecisionTree(df, x, y, max_height= 4, min_nodes= 6)
    print(df.iloc[80])
    z = df.iloc[80].to_numpy()
    print(tester_tree.predict_value(z))



    


   




    


   


