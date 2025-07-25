import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,learning_rate=0.1,num_epochs=1000,loss_function='gradient_descent'):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.W = None
        self.b = None
        self.X = None
        self.y = None
        self.errors = []
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        self.num_rows ,self.num_features = X.shape
        
        self.W = np.zeros(self.num_features)
        self.b = 0
        self.y = np.where(y == 0, -1, 1) 
        
        for _ in range(self.num_epochs):
            if self.loss_function == 'perceptron_trick':
                self._perceptron_trick()
            elif self.loss_function == 'gradient_descent':
                self._gradient_descent()
            else:
                return None 

    def _gradient_descent(self):
        errors = 0
        for (x,y) in zip(self.X,self.y):
            
            z = np.dot(x,self.W) + self.b
            
            # Misclassified point
            if y*z <= 0:
                # Update weights and bias
                self.W += self.learning_rate * y * x
                self.b += self.learning_rate * y  
                errors += 1
        self.errors.append(errors)                 

    def _perceptron_trick(self):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
            
        X_shuffled = self.X[indices] 
        y_shuffled = self.y[indices] 
            
        for index,(x,y) in enumerate(zip(X_shuffled,y_shuffled)):
            # Check if the predicted output matches
            z = np.dot(x,self.W) + self.b
            y_pred = self._activation_function(z)
                
            # Update the weights and bias accordingly
            self.W = self.W + self.learning_rate * (y_pred - y) * x   
            self.b = self.b + self.learning_rate * (y_pred - y)             
    
    def predict(self,x):
        return self._activation_function(
            np.dot(x,self.W.T) + self.b  
        )
    
    def _activation_function(self,z):
        return np.where(z >= 0, 1, -1)
