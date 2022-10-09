from model import LogisticRegressionModel
import pandas as pd
import csv


#   Data Import function to read from .csv files returns a Data Frame obj 
def dataImport(loc):
    with open(loc) as data:
        return pd.DataFrame(csv.reader(data), columns=['X1', 'X2', 'y'])

data = dataImport('./data/data.csv')
X = data[['X1', 'X2']]
y = data['y']

n_features = X.shape[1] # 1 == Row numbers


test = LogisticRegressionModel(n_features) #shap[1] = n of columns
test.fit(X, y)