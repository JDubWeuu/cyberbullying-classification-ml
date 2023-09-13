import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import Word
import string


#preprocessing data
def pre_process(dataframe, stopwords):
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join(x if not x.isdigit() else '' for x in x.split()))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join(x for x in x.split() if x not in stopwords))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ' '.join(x if not x.startswith('@') else '@USERNAME' for x in x.split()))
    df['tweet_text'] = df['tweet_text'].apply(lambda x: ''.join(char for char in x if char == '#' or char not in string.punctuation))

    return df
stop_words = stopwords.words('english')
df = pd.read_csv('cyberbullying_tweets.csv')
df = pre_process(df, stop_words)

print(df)

tokenizer = Tokenizer(num_words=10000, split=' ')
tokenizer.fit_on_texts(df['tweet_text'].values)

sequences = tokenizer.texts_to_sequences(df['tweet_text'].values)
padded_sequences = pad_sequences(sequences)

encoder = LabelEncoder()
df['cyberbullying_type'] = encoder.fit_transform(df['cyberbullying_type'])
labels = df['cyberbullying_type']

X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    Embedding(10000, 128, input_length=padded_sequences.shape[1]),
    SpatialDropout1D(0.4),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='LeakyReLU'),
    Dense(6, activation="softmax")
    ])

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

print(model.summary())

model.fit(X_train, Y_train, validation_split=0.1, epochs = 10, batch_size=32, verbose=1)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Loss: {test_loss}\nAccuracy: {test_acc}')