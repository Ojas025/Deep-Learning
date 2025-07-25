import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
from Perceptron import Perceptron 
from sklearn.metrics import r2_score,accuracy_score
from mlxtend.plotting import plot_decision_regions 
import matplotlib.pyplot as plt


def main():
    X,y = make_classification(n_samples=200,n_features=2,n_redundant=0,n_repeated=0,random_state=12345,n_clusters_per_class=1)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

    model = Perceptron(loss_function="gradient_descent",learning_rate=0.5,num_epochs=1000)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    score = accuracy_score(y_test,y_pred)
    print("Score:",score)
    
    plot(X_train,y_train,model)
    plot_error(model)
    

def plot(X_train,y_train,model):
    plt.figure()
    plot_decision_regions(X_train,y_train,clf=model,legend=2)
    plt.show()

    
def plot_error(model):
    plt.figure()
    plt.plot(range(len(model.errors)),model.errors,label="Training Error") 
    plt.xlabel("Epochs")
    plt.ylabel("Misclassified Points")
    plt.title("Perceptron Error vs Epochs")
    plt.grid(True)
    plt.legend()
    plt.show() 
      
main()