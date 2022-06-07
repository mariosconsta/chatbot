#IMPORTS
import json

from torch import optim 
from nltk_utils import tokenize, stem, bow
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import Neuralnet

#load JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []

#Scan each intent and its words. Add them to their array while appling tokenization and stemming
# The for loop iterates through each intent in the intents json file.
# The tag variable is assigned the value of the tag from the json file. 
# The tags array is appended with the tag variable. 
# The for loop iterates through each pattern in the intent json file. 
# The w variable is assigned the tokenized version of the pattern. 
# The all_words array is extended with the w variable. 
# The xy array is appended with the tag and w variables.
for intent in intents['intents']:
    tag = intent['tag'] #tags from the json file
    tags.append(tag) #adding tags to the array
    for pattern in intent['patterns']:
        w = tokenize(pattern) #tokenization of each pattern
        all_words.extend(w) #insert each word into the all_words array
        xy.append((w,tag)) #create corralation between each tag and word for each intent
        
ignore_words = ['?', '!', '.', ','] #these words will be ignored

all_words = [stem(w) for w in all_words if w not in ignore_words] #Applying stemming too all words that are not ignored
all_words = sorted(set(all_words)) #remove duplicate words

tags = sorted(set(tags)) #remove duplicate tags

X_train = [] # array for bow (bag of words)
y_train = [] #array for tags

#Insert data into our X_train and y_train
# For each sentence in the training data, we create a bag of words. 
# 
# Then we label each bag with the appropriate label. 
# 
# Then we convert the labels to a numpy array and then we are ready to train our model.
for (pattern_sentence, tag) in xy:
    bag = bow(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag) 
    y_train.append(label) #CrossEntropy Loss
    
#Set X_train and y_train as np arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# The ChatDataset class inherits from the Dataset class. 
# It takes the number of samples, the data, and the labels as arguments. 
# It also initializes the number of samples and the data. 
# The __getitem__ method returns the data and labels at a given index. 
# The __len__ method returns the number of samples.
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.001
num_epochs = 1000

#Save our dataset    
dataset = ChatDataset()

#Use our dataset in a data loader with all the hyperparameters we created
train_loader = DataLoader(dataset=dataset , batch_size=batch_size, shuffle = True, num_workers=0) #create a data loader with these  characteristics

# This code is telling the computer to use the GPU if it is available. If not, it will use the CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Set our model with its hyperparameters
model = Neuralnet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Training Loop
# 1. For each epoch, we iterate through all of the batches in the training set.
# 2. For each batch, we forward pass the input data through the model, calculate the loss, and then
# update the model weights.
# 3. We also compute the loss periodically as we iterate through the epoch.
for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if(epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')
        
#Print final epoch loss
print(f'final loss, loss = {loss.item():.4f}')

#Save data into data.pth file for later use
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags 
}

FILE = "data.pth"
torch.save(data, FILE) #serialise and save the data in a file

print(f'traning complete. file saved to {FILE}')