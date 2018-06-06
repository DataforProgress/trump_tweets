from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from keras_utils import save_model
import numpy as np
import random
import sys
import csv
import re

MAXLEN = 40
STEP = 3
CHAR_MAP = {'â': 'a', 'è': 'e', 'é': 'e', 'í': 'i', 'ï': 'i', 'ñ': 'n', 'ó': 'o', 'ø': 'o', 'ú': 'u', 'ğ': 'g', 'ı': '1', 'ĺ': 'l', 'ō': 'o', 'ễ': 'e', '–': '-', '—': '-', '―': '-', '‘': "'", '’': "'", '“': "'", '”': "'", '•': '*', '…': '...', '′': "'", '£': '$', '€': '$', 'ｒ': 'r', 'ｔ': 't'}
CHARS = [' ', '!', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '@', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~']


def data_processing(tweets_csv):
    """
    Takes csv file of tweets where tweet text is 6th column, produces vectorized x,y for character LSTM
    :param tweets_csv: file of tweets
    :return:
        x: input sequence characters to learn to predict from
        y: output char to predict from input
        tweets: list of processed tweets
        chars: chars in tweets, in practice same as CHARS
        char_indices: chars to indices map
        indices_char: indices to chars map
    """
    with open(tweets_csv, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        tweets=[]
        for line in reader:
            if len(line[5]) < 61:
                continue
            tweet = line[5].lower()
            # removing urls is always bad but this is my favorite very comprehensive regex
            tweet = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()'''
                            '''<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".'''
                            ''',<>?«»“”‘’]))''', "", tweet)
            processed_tweet = ''
            for char in tweet:
                if char in CHAR_MAP.keys():
                    char = CHAR_MAP[char]
                processed_tweet += char if char in CHARS else ''
            tweets.append(processed_tweet)
    print('corpus length:', len(tweets))

    chars = sorted(list(set("".join(tweets))))
    print(chars)
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters

    sentences = []
    next_chars = []
    for tweet in tweets:
        for i in range(0, len(tweet) - MAXLEN, STEP):
            sentences.append(tweet[i: i + MAXLEN])
            next_chars.append(tweet[i + MAXLEN])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y, tweets, chars, char_indices, indices_char


# build the model: a single LSTM
def craete_char_lstm(chars=CHARS):
    """
    Create simple character LSTM model
    :param chars: set of characters the model will be able to predict on and generate
    :return: LSTM model
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=(MAXLEN, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    return model


def sample(preds, temperature=1.0):
    """
    Given the vector of predicted next chars (a PDF over all possible chars) and the temperature, return a single char
    drawn from the distribution
    :param preds: vector of predicted next chars (a PDF over all possible chars
    :param temperature: increases or decreases probability of returning more proabable chars
    :return: next char sample
    """
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    """
    At the end of each epoch print several sample generated sentences from model
    :param epoch:
    :param logs:
    :return:
    """
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    tweet = np.random.choice(tweets)
    start_index = random.randint(0, len(tweet) - MAXLEN - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = tweet[start_index: start_index + MAXLEN]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(140):
            x_pred = np.zeros((1, MAXLEN, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

x, y, tweets, chars, char_indices, indices_char = data_processing('realdonaldtrump.csv')
model = craete_char_lstm(chars)
model = multi_gpu_model(model)
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
h = model.fit(x, y, batch_size=128, epochs=60, verbose=2, callbacks=[print_callback])
save_model('trump_lstm', model, h)
