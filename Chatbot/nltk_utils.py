import numpy as np   
import nltk #NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language
from nltk.tokenize.api import TokenizerI #To tokenization splits the sentence to words
nltk.download('punkt') #it is a package with pre trained tokenizer

from nltk.stem.porter import PorterStemmer #Interfaces used to remove morphological affixes from words
stemmer = PorterStemmer() #root of the word. no ending

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower()) #stemming the word and convert every letter to lowercase


def bow(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence] #tokenized sentence with the bag of words of the pattern
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words): # text kai thn trexoysa leksh
        if w in tokenized_sentence:# 0 when there is no conection between the users input the pattern. 1 when there is
            bag[idx] = 1.0
            
    return bag