### Neural Machine Translation (NMT) 
# usage: python seq2seq_query.py "Ich probiere es."
import string
import re
from numpy import array, argmax, random, take
import pandas as pd
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## for GPU
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

## Import Dataset

# read raw text file
def read_text(filename): 
	file = open(filename, mode='rt', encoding='utf-8')
	text = file.read()
	file.close()
	return text

# split a text into sentences
def to_lines(text): #
	sents = text.strip().split('\n')
	sents = [i.split('\t') for i in sents]
	return sents

data = read_text("data/nmt/deu-eng.txt")
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
		 
# prepare test data
deu_text =sys.argv[1]
eng_text="????????"
test=[]
test.append([eng_text, deu_text])
test=array(test)

testX = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

# Load Model
print("Loading Model...")
model = keras.models.load_model("model/seq2seq_deu-eng.h5")

# Test Model
preds = model.predict_classes(testX)

def get_word(n, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == n:
			return word
	return None

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


pred_df = pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})
print(pred_df)
print()
print("input  deu: ", deu_text)
print("output eng: ", preds_text[0])
