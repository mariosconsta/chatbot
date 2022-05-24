import numpy as np   
import nltk
from nltk.tokenize.api import TokenizerI #To tokenization splits the sentence to words
nltk.download('punkt') #it is a package with pre trained tokenizer

from nltk.stem.porter import PorterStemmer #Interfaces used to remove morphological affixes from words
stemmer = PorterStemmer() #root of the word. no ending

def tokenize(sentence):
    '''
    Given a sentence, tokenize it into a list of words.
    
    :param sentence: the sentence to be tokenized
    :return: A list of tokens.
    '''
    return nltk.word_tokenize(sentence)


def stem(word):
    '''
    This function takes a word as input and returns its stem.
    
    :param word: the word to be stemmed
    :return: The stemmed word
    '''
    return stemmer.stem(word.lower()) #stemming the word and convert every letter to lowercase


def bow(tokenized_sentence, all_words):
    '''
    The function takes in a tokenized sentence and the bag of words (all_words). 
    It returns a bag of words array with 1s at the indices of the words in the bag that are present in
    the sentence.
    
    :param tokenized_sentence: the bag of words of the pattern
    :param all_words: a list of all the words in the vocabulary
    :return: The bag of words of the pattern.
    '''
    tokenized_sentence = [stem(w) for w in tokenized_sentence] #tokenized sentence with the bag of words of the pattern
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:# 0 when there is no conection between the users input the pattern. 1 when there is
            bag[idx] = 1.0
            
    return bag