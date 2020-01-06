### Text Generation
import sys
import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import LambdaCallback

# load ascii text and covert to lowercase
filename = "data/text/shakespeare.txt"
text = open(filename, encoding='utf-8').read()
text = text.lower()
print('corpus length:', len(text))

# create mapping of unique chars to integers
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of max_seq_len characters
max_seq_len = 100
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - max_seq_len, step):
    sentences.append(text[i: i + max_seq_len])
    next_chars.append(text[i + max_seq_len])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), max_seq_len, len(chars)), dtype=np.bool)
Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    Y[i, char_indices[next_chars[i]]] = 1

print(X.shape)
print(Y.shape)
print(max_seq_len, len(chars))

# Build Model
model = Sequential()

model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]) ))
model.add(Dense(len(chars), activation='softmax'))

model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Train model
batch_size = 128
num_epochs = 10 #60
model.fit(X, Y, batch_size=batch_size, epochs=num_epochs)

# Save Model
model.save('model/textgen_lstm.h5')

# Generate Text
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text():
    start_index = random.randint(0, len(text) - max_seq_len - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + max_seq_len]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, max_seq_len, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

generate_text()
