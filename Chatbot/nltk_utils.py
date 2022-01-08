import numpy as np   
import nltk #einai mia bibliothikh ths python h opoia bohthaei sthn dhmioyrgia programmatwn poy exoyn na kanoyn me thn anthrwpinh glwssa
from nltk.tokenize.api import TokenizerI #To tokenization xwrizei thn protash se lekseis
nltk.download('punkt') #einai ena paketo me proekpaideymeno tokenizer

from nltk.stem.porter import PorterStemmer #O algorithmos porterstemmer afairei tis koines morfologikes katalhkseis apo tis agglikes lekseis
stemmer = PorterStemmer() #To stemming paragei thn riza ths lekshs ( kobei tis katalikseis)

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower()) #theloyme na kanei stemming sthn leksh kai na kanei ola ta grammata mikra


def bow(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence] # exoyme thn tokenized protash mas mazi me to bag of words toy ekastote pattern
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words): # tha mas dwsei to keimeno kai thn trexoysa leksh
        if w in tokenized_sentence:# oysiastika bazoyme 0 otand den yparxei syndesh stis lekseis poy exei balei o xrhsths kai sto pattern kai 1 opoy yparxei syndesh
            bag[idx] = 1.0
            
    return bag