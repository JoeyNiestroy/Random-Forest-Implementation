from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from Random_Forest_Model import RandomForest
df = pd.read_csv("test_data.csv")
df_train = df[0:183]
df_test = df[183:]
x_list = ["highwaympg","horsepower","citympg", "stroke", "carlength","carwidth", "carheight","curbweight", "enginesize", "peakrpm"]
y_string = "price"
x = df_train[x_list]
y = df_train["price"]
x2 = df_test[x_list]
y2 = df_test["price"]
SK_tree_reg = RandomForestRegressor(n_estimators= 130, max_depth= 12, min_samples_split=4, max_features= "sqrt", min_impurity_decrease=.01)
SK_tree_reg.fit(x,y)
random_forest_model = RandomForest(df_train, x_list, y_string, numberoftrees= 135, max_depth= 12, threashold_split=.01)
print("R^2 Score for SK forest: "+str(SK_tree_reg.score(x2,y2)))
print(("R^2 Score for my forest: ")+str(random_forest_model.r_squared_for_test_data(df_test)))

