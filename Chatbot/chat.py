import random #theloyme random epilogh apo tis pithanes apanthseis
import json
import torch
from model import Neuralnet
from nltk_utils import bow, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #elegxoyme an exoyme gpu support

with open('intents.json', 'r') as f: #theloyme na anoiksoyme to file
    intents = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)
#pairnoyme tis idies plhrofories me to train
input_size = data["input_size"] 
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = Neuralnet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) #load the state dict of the creation.
model.eval()
# dhmioyrgoyme to chat
context = {}
bot_name = "Peanut"

def response(msg, userID='123'):
    sentence = tokenize(msg) #tokenize the sentence
    X = bow(sentence, all_words) #create the bag of words. me thn tokenized protash kai oles tis lekseis poy exoyme apo to saved arxeio
    X = X.reshape(1, X.shape[0]) #reshape
    X = torch.from_numpy(X) #bag of words returns in numpy array
    
    output = model(X)
    _, predicted = torch.max(output, dim=1) # tha mas dwsei thn problepsh
    tag = tags[predicted.item()] # we want the intexts tag
    
    probs = torch.softmax(output, dim=1)
    probs = probs[0][predicted.item()]
    
    if probs.item() > 0.65: #theloyme h pithanothta na einai panw apo 65% wste oi apanthseis na einai sxetikes
        for intent in intents["intents"]: #psaxnei ta intents me to idio tag
            if tag == intent["tag"]:
                if 'context_set' in intent:
                    context[userID] = intent['context_set']
                
                if not 'context_filter' in intent or \
                 (userID in context and 'context_filter' in intent and intent['context_filter'] == context[userID]):
                     return random.choice(intent['responses']) #tyxaia epilogh apo ta responses
                     
    return "I do not understand..." #ean den einai panw apo 65% tote to bot tha dwsei ayth thn apanthsh