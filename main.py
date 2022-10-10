#   Imoports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import csv

#importing implemented model
from model import LogisticRegressionModel

#   Data Import function to read from .csv files returns a Data Frame obj 
def dataImport(loc):
    with open(loc) as data:
        return pd.DataFrame(csv.reader(data), columns=['X1', 'X2', 'y'], dtype=float)

#   Split Data in X and y
def splitData(data):
    return data.loc[:,["X1", "X2"]], data.loc[:,"y"]

#   Use the SKLearn Logistic Regression Function imported from library 
def imported_logistic_regression(data):
    # Splitting DataFrame into X, y fro training model
    X, y = splitData(data)

    

    # Training the model
    fitModel = LogisticRegression(random_state=0).fit(X, y)

    prediction = fitModel.predict(X)

    print(f"Predition accuracy by sklearn model: {int(fitModel.score(X, y)*100)}%")

    pass


def implemented_logistic_regression(data):
    # Splitting DataFrame into X, y fro training model
    X, y = splitData(data)

    X = X.values
    y = y.values

    X = X.T
    y = y.reshape(1, X.shape[1])


    model = LogisticRegressionModel()
    model.fit(X, y)

    pred_Ys = model.predict(X)

    model.accuracy(pred_Ys, y[0])


    pass

#######################################################################################
#       MAIN FUNCTION       ###########################################################
#######################################################################################
if __name__ == "__main__":
    """
	Run ML training on the data in data/data.csv using:
	1. The LogisticRegression Model implemented by you, and
	2. The LogisticRegression Model in sklearn 
	and print out the accuracy for both models
	"""
    data = dataImport('./data/data.csv')

    # Logistinc regression from sk.Learn
    imported_logistic_regression(data)

    implemented_logistic_regression(data)