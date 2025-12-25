import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
    
    def forward(self, x):
        output, self.hidden_states = self.LSTM(x.view(x.shape[0],x.shape[1],self.input_size))
        
        return output, self.hidden_states
    
    def init_hidden_states(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )
    
class Decoder(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, encoder_hidden_states):
        output, self.hidden_states = self.LSTM(x.unsqueeze(0), encoder_hidden_states)
        output = self.fc(output.squeeze(0))
        
        return output, self.hidden_states
        
    
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = Encoder(input_size,hidden_size)
        self.decoder = Decoder(input_size,hidden_size)
    
    def train(self, input, target, n_epochs, target_len, batch_size, training_prediction='recursive', teacher_forcing_ratio=0.5,learning_rate=0.01,dynamic_tf=False):
        losses = np.full(n_epochs, np.nan)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        n_batches = int(input.shape[1] / batch_size)
        
        for epoch in range(n_epochs):
            batch_loss = 0.0
            for batch_index in range(n_batches):
                input_batch = input[:, batch_index: batch_index + batch_size, :]
                target_batch = target[:, batch_index: batch_index + batch_size, :]

                outputs = torch.zeros(target_len, batch_size, input.shape[2])
                
                encoder_hidden = self.encoder.init_hidden_states(batch_size)
                
                optimizer.zero_grad()
                
                encoder_output, encoder_hidden = self.encoder(input_batch)
                
                decoder_input = input_batch[-1,:,:]
                decoder_hidden = encoder_hidden
                
                if training_prediction == 'recursive':
                    for t in range(target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[t] = decoder_output
                        decoder_input = decoder_output
                
                if training_prediction == 'teacher_forcing':
                    if random.random() < teacher_forcing_ratio:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = target_batch[t,:,:]
                    else:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output      
                
                loss = criterion(outputs, target_batch)
                batch_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            batch_loss /= n_batches
            losses[epoch] = batch_loss

        return losses                                                                                             
    
    def predict(self, input, target_len):
        input = input.unsqueeze(1)
        encoder_output, encoder_hidden = self.encoder(input)
        
        outputs = torch.zeros(target_len, input.shape[2])
        
        decoder_input = input[-1,:,:]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output
        
        return outputs            