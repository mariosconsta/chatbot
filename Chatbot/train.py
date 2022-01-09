#IMPORTS
import json

from torch import optim 
from nltk_utils import tokenize, stem, bow
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import Neuralnet
###################################################################

#load JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = [] #we create empty words, we want to collect all the differnet patterns
tags = [] #we create an empty array for all the different patterns of json
xy = [] #we create an empty array witch contain all the different patterns and tags

for intent in intents['intents']:
    tag = intent['tag'] #tags from the json file
    tags.append(tag) #adding tags to the array
    for pattern in intent['patterns']:
        w = tokenize(pattern) #tokenization of each pattern
        all_words.extend(w) #insert each word into the all_words array
        xy.append((w,tag)) #create corralation between each tag and word
        
ignore_words = ['?', '!', '.', ','] #these words will be ignored

all_words = [stem(w) for w in all_words if w not in ignore_words] #Applying stemming too all words that are not ignored
all_words = sorted(set(all_words)) #remove duplicate words

tags = sorted(set(tags)) #remove duplicate tags

X_train = [] # array for bow (bag of words)
y_train = [] #array for tags

for (pattern_sentence, tag) in xy:
    bag = bow(pattern_sentence, all_words) #we create a bag of words, it will collect the tokenized sentences
    X_train.append(bag) # kai to prosaptoyme sto training data mas
    
    label = tags.index(tag) # numbers for tags 
    y_train.append(label) #CrossEntropy Loss
    
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset): #we create a new dataset
    def __init__(self):
        self.n_samples = len(X_train) #sumples numbe equals to the size of the array x_train
        self.x_data = X_train # datas in the array
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] # epistrefei ta dedomena twn pinakwn
    
    def __len__(self):
        return self.n_samples #epistrefei ton arithmo twnd deigamtwn
    

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags) #number of different classes
input_size = len(all_words) #number of the lenght of each bag of word. Exei to idio megethos oso oles oi lekseis
learning_rate = 0.001
num_epochs = 1000
    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset , batch_size=batch_size, shuffle = True, num_workers=0) #create a data loader with these  characteristics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #elegxoyme an exoyme gpu support gia na to xrhsimopoihsoyme
model = Neuralnet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs): #training loop
    for(words, labels) in train_loader: #training loader
        words = words.to(device) #puss it to the device
        labels = labels.to(dtype=torch.long).to(device)
        
        #forward
        outputs = model(words)
        loss = criterion(outputs, labels) #calculate the loss
        
        #backward and optimizer step
        optimizer.zero_grad() #empty the gradience
        loss.backward() #calculate the back propagation
        optimizer.step()
        
    if(epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}') #every 100 steps provide epoch
        
print(f'final loss, loss = {loss.item():.4f}') #provide final loss
# save the data
data = {
    "model_state": model.state_dict(), #save the model state
    "input_size": input_size, #save the input size
    "output_size": output_size, #save the output size
    "hidden_size": hidden_size, #save the hiden size
    "all_words": all_words, #store all the words that we colected
    "tags": tags #store all the tags
}

FILE = "data.pth"
torch.save(data, FILE) #serialise and save the data in a file

print(f'traning complete. file saved to {FILE}')