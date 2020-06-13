### Neural Machine Translation (NMT) 
# usage: python seq2seq_query.py
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load Model
print("Loading Model...")
model = load_model("model/seq2seq_deu-eng.h5")

## Import Dataset

# read raw text file
filename = "dataset/nmt/deu.txt" # English-to-Deutsche
f = open(filename, mode='rt', encoding='utf-8')
data = f.read()
f.close()

# split a text into sentences
def to_lines(text): #
	sents = text.strip().split('\n')
	sents = [i.split('\t') for i in sents]
	return sents

deu_eng = to_lines(data)
deu_eng = array(deu_eng)    #total = 150,000 sentences
deu_eng = deu_eng[:50000,:] #train =  50,000 sentences 

# Remove punctuation
deu_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,0]]
deu_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,1]]

# convert text to lowercase
for i in range(len(deu_eng)):
	deu_eng[i,0] = deu_eng[i,0].lower()
	deu_eng[i,1] = deu_eng[i,1].lower()
	
## Preprocessing Data (Tokenize)

# function to build a tokenizer
def tokenization(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# prepare english tokenizer
eng_tokenizer = tokenization(deu_eng[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 8
print('English Vocabulary Size: %d' % eng_vocab_size)

# prepare Deutch tokenizer
deu_tokenizer = tokenization(deu_eng[:, 1])
deu_vocab_size = len(deu_tokenizer.word_index) + 1

deu_length = 8
print('Deutch Vocabulary Size: %d' % deu_vocab_size)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	seq = tokenizer.texts_to_sequences(lines)               # integer encode sequences    
	seq = pad_sequences(seq, maxlen=length, padding='post') # pad sequences with 0 values
	return seq



def get_word(n, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == n:
			return word
	return None
	
# Input test data
deu_text =" "
while(deu_text!='exit'):
    deu_text = input('Deutsche>')
    eng_text="????????"
    test=[]
    test.append([eng_text, deu_text])
    test=array(test)

    test_X = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
    test_Y = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

    # predict 
    preds = model.predict_classes(test_X)

    # convert predictions into text	  
    preds_text = []
    for i in preds:
	    temp = []
	    for j in range(len(i)):
		    t = get_word(i[j], eng_tokenizer)
		    if j > 0:
			    if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
				    temp.append('')
			    else:
				    temp.append(t)
		    else:
			    if(t == None):
				    temp.append('')
			    else:
				    temp.append(t) 
	    preds_text.append(' '.join(temp))
    print("English>", preds_text[0])
    print()
