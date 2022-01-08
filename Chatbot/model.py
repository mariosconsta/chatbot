import torch
import torch.nn as nn

# Feed-forward NN with 2 hidden layers
class Neuralnet(nn.Module): #tha einai ena fedd forword neural net me ena layer poy tha exei toys arithmoys apo diaforetika patterns kai me dyo hidden layers poy tha pairnei to bag of words kai tha dinei pithanotites gia to kathe class
    def __init__(self, input_size, hidden_size, num_classes):
        super(Neuralnet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x): #exoyme tria layers opws eipame kai parapanw
        out = self.l1(x)
        out = self.relu(out)
        
        out = self.l2(out)
        out = self.relu(out)
        
        out = self.l3(out)
        # No activation and no softmax
        return out