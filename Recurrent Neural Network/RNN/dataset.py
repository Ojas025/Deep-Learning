import numpy as np

class Dataset:
    def __init__(self, X, y, split_ratio=0.8, batch_size=1):
        self.split_ratio = split_ratio
        self.X = X
        self.y = y
        
        self.vocab = None
        self.word_to_index = {}
        self.index_to_word = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def tokenize_sentence(self, sentence):
        # Split sentence into tokens
        sentence = sentence.lower()
        return sentence.split()

    
    def build_vocab(self, data):
        self.vocab = sorted(list(set([text for i in data for text in self.tokenize_sentence(i)])))
    
    def encode_sentence(self, sentence):
        tokens = self.tokenize_sentence(sentence)
        indices = []
        for word in tokens:
            indices.append(self.word_to_index[word])
            
        return indices    
    
    def encode_dataset(self, sentences):
        encoded_dataset = []
        for sentence in sentences:
            encoded_sentence = self.encode_sentence(sentence)
            encoded_dataset.append(encoded_sentence)
        
        return encoded_dataset    
    
    def decode_indices(self, indices):
        sentence = []
        for index in indices:
            sentence.append(self.index_to_word[index])

        return sentence
            
    def split(self):
        n_samples = len(self.X)
        indices = np.random.permutation(len(self.X))
        split_index = int(n_samples * self.split_ratio)
        
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
    
        self.X_train = [self.X[i] for i in train_indices]
        self.y_train = [self.y[i] for i in train_indices]
        
        self.X_test = [self.X[i] for i in test_indices]
        self.y_test = [self.y[i] for i in test_indices]
    
    def prepare_data(self):
        self.split()
        
        self.build_vocab(self.X_train)
        
        self.word_to_index = {w:i for i, w in enumerate(self.vocab)}
        self.index_to_word = {i:w for i, w in enumerate(self.vocab)}
        
        X_train_encoded = self.encode_dataset(self.X_train)
        
        X_test_encoded = self.encode_dataset(self.X_test)
        
        return X_train_encoded, X_test_encoded, self.y_train, self.y_test
        
    
    def get_batches(self):
        pass
    
    def vocab_size(self):
        return len(self.vocab)