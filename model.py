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
    
    def __init__(self, n_features ):
        self.weights = np.zeros((1, n_features))
        self.b = 0
        pass
 
    def cost_function(self, X, y):
        # Computes the cost function for all the training samples
        # TODO: implement this function	

        m = X.shape[0] #rows
        Y_T = y.T # Transposition
        sigma = self.sigmoid(np.dot(self.weights,X.T)+self.b)

        return (-1/m)*(np.sum((Y_T*np.log(sigma)) + ((1-Y_T)*(np.log(1-sigma)))))
        pass




    def gradient(self, theta, X, y):
        # Computes the gradient of the cost function at the point theta
        # TODO: implement this function

        m = X.shape[0]

        #Gradient calculation
        dw = (1/m)*(np.dot(X.T, (theta-y.T).T))
        db = (1/m)*(np.sum(theta-y.T))
        
        gradient = {"dw": dw, "db": db}

        return gradient
        pass

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit(self, X, y, learning_rate=0.0001, iterations = 100):
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
        costs = []
        for i in range(iterations):
            cost = self.cost_function(X, y)
            gradient = self.gradient(X, y)

        pass

    def predict(self, x, theta, probab_threshold=0.5):
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



        pass

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



        pass