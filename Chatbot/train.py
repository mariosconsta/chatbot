import json #bazoyme to json arxeio poy exoyme ftiaksei

from torch import optim 
from nltk_utils import tokenize, stem, bow
import numpy as np #h bibliothikh numpy prosthetei yposthriksh gia polidiastatoys pinakes 

import torch #einai mia beltistopoihmenh bibliothikh poy periexei domes dedomenwn kai orizei mathhmatikes prakseis
import torch.nn as nn #mas bohthaei na dhmioyrghsoyme kai na ekpaideysoyme to neural natwork
from torch.utils.data import Dataset, DataLoader #bohthaei sthn polydiastath fortwsh dedomenwn, sto aytomatopoihmeno batching, ftiaxnei thn seira fortwshs dedomenwn

from model import Neuralnet

with open('intents.json', 'r') as f: #fortwnoyme to json arxeio se read mode('r')
    intents = json.load(f)

all_words = [] #dhmioyrgoyme kenes listes, epshs theloyme na sylleksoyme ola ta diaforetika motiba
tags = [] #dhmioyrgoyme mia kenh lista gia ta diaforetika tags toy json
xy = [] #dhmioyrgoyme mia kenh lista poy tha periexei ola ta patterns kai ta tags mas

for intent in intents['intents']:
    tag = intent['tag'] #ayta einai ta tags toy arxeioy json
    tags.append(tag) #tha to prosapsoyme ston pinaka tags
    for pattern in intent['patterns']:
        w = tokenize(pattern) #kanoyme tokenization sta patterns dhladh se ayta poy plhktrologei o xrhsths
        all_words.extend(w) # ta bazoyme ston pinaka (all words). bazoyme to extend giati einai hdh array kai den theloyme na exoyme array apo arrays
        xy.append((w,tag)) #tha kserei to pattern kai to tag
        
ignore_words = ['?', '!', '.', ','] # den tha lambanei ypopshn toy shmeia stikshs

all_words = [stem(w) for w in all_words if w not in ignore_words] #kanoyme stemming se oles tis lekseis gia to "w" sto "all words" ean den einai shmeio stikshs fysika
all_words = sorted(set(all_words)) #afairoyme ta diplotypa stoixeia

tags = sorted(set(tags)) # tha exei monadikes etiketes

X_train = [] # lista gia to bow (bag of words)
y_train = [] #lista gia ta tags

for (pattern_sentence, tag) in xy:
    bag = bow(pattern_sentence, all_words) #ftiaxnoyme ena bag of words, tha parei tis tokenized protaseis
    X_train.append(bag) # kai to prosaptoyme sto training data mas
    
    label = tags.index(tag) # tha exoyme arithmoys gia ta tags mas 
    y_train.append(label) #CrossEntropy Loss
    
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset): #dhmioyrgoyme ena neo dataset
    def __init__(self):
        self.n_samples = len(X_train) # o arithmos twn deigmatwn einai isos me to megethos toy pinaka x_train
        self.x_data = X_train # bazoyme ta dedomena ston pinaka
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
train_loader = DataLoader(dataset=dataset , batch_size=batch_size, shuffle = True, num_workers=0) #dhmioyrgoyme enan data loader me ayta ta xarakthristika 

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
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}') #kathe 100 bhmata theloyme na dinei to epoch
        
print(f'final loss, loss = {loss.item():.4f}') #dinei to final loss
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