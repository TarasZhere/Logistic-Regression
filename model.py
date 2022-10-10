import numpy as np
import pandas as pd

# You can use fmin_tnc function to do optimization if you have hard time to implement gradient descent
# or it will be even better if you can implement your own function to do gradient descent,
'''
Uses the fmin_tnc function that is used to find the minimum for any function
It takes arguments as
	1) func : function to minimize
	2) x0 : initial values for the parameters
	3) fprime: gradient for the function defined by 'func'
	4) args: arguments passed to the function
'''
from scipy.optimize import fmin_tnc


class LogisticRegressionModel:
    
    def __init__(self):
        self.w = None
        self.b = None
 
    def cost_function(self, theta, y):
        # Computes the cost function for all the training samples
        # TODO: implement this function	
        return -(1/self.m)*np.sum(y*np.log(theta) + (1-y)*np.log(1-theta))

    def gradient(self, theta, X, y):
        # Computes the gradient of the cost function at the point theta
        # TODO: implement this function
        return {
                "dw":(1/self.m)*np.dot(theta-y, X.T), 
                "db":(1/self.m)*np.sum(theta-y)
            }


    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def fit(self, X, y, learning_rate=0.0015, iterations = 100000):
        """trains the model from the training data
        
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features] // [rows, columns]
            Training samples
        y: array-like, shape = [n_samples, n_target_values]
            Target classes
        init_theta: initial weights
		alpah: learning rate
		num_iters: number of iterations 
        Returns
        -------
        final optimized set of parameters theta
        """
        # TODO: implement this function

        self.n, self.m = X.shape
        self.w = np.zeros((self.n, 1), dtype=float)
        self.b = 0

        costs = []
        for i in range(iterations):

            z = np.dot(self.w.T, X) + self.b

            theta = self.sigmoid(z)

            cost = self.cost_function(theta, y)

            costs.append(cost)

            gradient = self.gradient(theta, X, y)

            self.w -= learning_rate*gradient['dw'].T
            self.b -= learning_rate*gradient['db']

        pass

    def predict(self, x, probab_threshold=0.5):
        """ Predicts the class labels
        Parameters: 
        ----------
        x: array-like, shape = [n_samples, n_features]
            Test samples
		theta: final set of trained parameters
		probab_threshold: threshold/cutoff to categorize the samples into different classes
        Returns
        -------
        predicted class labels
        """
        #TODO implement this function

        predictions = self.sigmoid(np.dot(self.w.T, x) + self.b)

        return [1 if i >= probab_threshold else 0 for i in predictions.T]




    def accuracy(self, predicted_classes, actual_classes):
        """Computes the accuracy of the classifier
        Parameters
        ----------
        predicted_classes: class labels from prediction
        actual_classes : class labels from the training data set
        
		
        Returns
        -------
        accuracy: accuracy of the model
        """
		#TODO implement this function

        m = len(predicted_classes)
        acc = 0

        for i in range(m):
            if predicted_classes[i] == actual_classes[i]:
                acc +=1

        print(f"Predition accuracy by implemented model: {int(round(acc/m*100, 2))}%")

        pass