import torch
import torch.nn as nn

# Feed-forward NN with 2 hidden layers
class Neuralnet(nn.Module): #it will be a feed forward neural net with layer wich will has numbers from different patterns and with two hidden layers that will take the bag of words and will give possibilities for every class
    def __init__(self, input_size, hidden_size, num_classes):
        super(Neuralnet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    #three layers    
    def forward(self, x): 
        out = self.l1(x)
        out = self.relu(out)
        
        out = self.l2(out)
        out = self.relu(out)
        
        out = self.l3(out)
        # No activation and no softmax
        return out