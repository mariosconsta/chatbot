import json
from nltk_utils import tokenize, stem, bow
import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data improt Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
        
ignore_words = ['?', '!', '.', ',']

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bow(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label) #CrossEntropy Loss
    
X_train = np.array(X_train)
y_train = np.array(y_train)