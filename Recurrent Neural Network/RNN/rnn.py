import numpy as np
import tqdm
from input_data import X,y
from dataset import *
     
    
class RNN:
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Network
        self.w1 = self.init_weights(input_size, hidden_size)
        self.w2 = self.init_weights(hidden_size, hidden_size)
        self.w3 = self.init_weights(hidden_size, output_size)
        
        self.b2 = np.zeros((1, hidden_size))
        self.b3 = np.zeros((1, output_size))
        
        self.input_size = input_size
    
    def one_hot_encode(self, indices):
        inputs = []
        for idx in indices:
            vector = np.zeros((1, self.input_size))
            vector[0][idx] = 1
            inputs.append(vector)
        
        return inputs    
    
    def init_weights(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))
    
    def tanh(self, input, derivative=False):
        if derivative:
            return 1 - (input ** 2)
        
        return np.tanh(input)
    
    def softmax(self, input):
        return np.exp(input) / np.sum(np.exp(input))
    
    def forward(self, inputs):
        # Maintain a list of hidden states
        self.hidden_states = [np.zeros_like(self.b2)]
        
        for input in inputs:
            layer1_output = np.dot(input, self.w1)
            layer2_output = self.tanh(layer1_output + np.dot(self.hidden_states[-1], self.w2) + self.b2)

            self.hidden_states.append(layer2_output)
            
        return np.dot(self.hidden_states[-1], self.w3) + self.b3            

    
    def backward(self, error, inputs):
        d_b3 = error
        d_w3 = np.dot(self.hidden_states[-1].T, error)
        
        d_b2 = np.zeros_like(self.b2)
        d_w2 = np.zeros_like(self.w2)
        d_w1 = np.zeros_like(self.w1)
        
        d_hidden_state = np.dot(error, self.w3.T)
        
        for q in reversed(range(len(inputs))):
            d_hidden_state *= self.tanh(self.hidden_states[q + 1], derivative=True)
            d_b2 += d_hidden_state
            d_w1 = np.dot(inputs[q].T, d_hidden_state)
            d_w2 = np.dot(self.hidden_states[q].T, d_hidden_state)
            d_hidden_state = np.dot(d_hidden_state, self.w2)
            
        for d_ in (d_b3, d_w3, d_b2, d_w2, d_w1):
            self.b3 += self.learning_rate * d_b3
            self.w3 += self.learning_rate * d_w3
            self.b2 += self.learning_rate * d_b2
            self.w2 += self.learning_rate * d_w2
            self.w1 += self.learning_rate * d_w1                    
    
    def train(self, inputs, labels):
        for _ in tqdm.tqdm(range(self.num_epochs)):
            for input, label in zip(inputs, labels):
                input = self.one_hot_encode(input)
                prediction = self.forward(input)
                error = -self.softmax(prediction)
                error[0][label] += 1
                
                self.backward(error, input)

    def test(self, inputs, labels):
        accuracy = 0
        for input, label in zip(inputs, labels):
            print(input)
            input = self.one_hot_encode(input)
            prediction = self.forward(input)

            print(['Negative', 'Positive'][np.argmax(prediction)], end='\n\n')
            if (np.argmax(prediction) == label):
                accuracy += 1
                
        print(f"Accuracy: {round(accuracy * 100 / len(inputs), 2)}")        
    

def main():
    dataset = Dataset(X,y,split_ratio=0.8, batch_size=1)
    X_train,X_test, y_train, y_test = dataset.prepare_data()
    vocab_size = dataset.vocab_size()

    
    rnn = RNN(input_size = vocab_size, hidden_size = 64, output_size = 2, learning_rate = 0.01, num_epochs= 1000)
    
    rnn.train(X_train, y_train)
    
    rnn.test(X_test, y_test)

main()