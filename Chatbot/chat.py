import random
import json
import torch
from model import Neuralnet
from nltk_utils import bow, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = Neuralnet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

context = {}
bot_name = "Peanut"

def response(msg, userID='123'):
    sentence = tokenize(msg)
    X = bow(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    probs = probs[0][predicted.item()]
    
    if probs.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                if 'context_set' in intent:
                    context[userID] = intent['context_set']
                
                if not 'context_filter' in intent or \
                 (userID in context and 'context_filter' in intent and intent['context_filter'] == context[userID]):
                     return random.choice(intent['responses'])
                     
    return "I do not understand..."