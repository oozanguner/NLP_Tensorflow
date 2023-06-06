from modules.functions import *
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import io
import json
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore") 

# 1) USING RNNs, LSTM with BIDIRECTIONAL LAYER

# Load the dataset
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

tokenizer = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 256

# Get the train and test splits
train_data, test_data = dataset['train'], dataset['test'], 

# Shuffle the training data
train_dataset = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the datasets to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)


# Build the model

## Bi-directional recurrent neural networks (Bi-RNNs) are artificial neural networks that process input data
## in both the forward and backward directions. They are often used in natural language processing tasks, such as language translation,
## text classification, and named entity recognition. In addition, they can capture contextual dependencies in the input data 
## by considering PAST and FUTURE contexts. Bi-RNNs consist of two separate RNNs that process the input data in opposite directions, 
## and the outputs of these RNNs are combined to produce the final output.
## For example, to predict the next word in a sentence, it is often useful to have the context around the word, 
## not only just the words that come before it.
## One of disadvantage of Bi-RNNs is that they require more memory and computation.

embedding_dim = 64

model1 = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

## You can build multiple layer LSTM models by simply appending another LSTM layer in your Sequential model and 
## enabling the return_sequences flag to True. This is because an LSTM layer expects a sequence input so if the previous layer is 
## also an LSTM, then it should output a sequence as well.

model2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.LSTM(32)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

model1.summary()

history = model1.fit(train_dataset, epochs=3, validation_data=test_dataset)

plot_metrics(history, "accuracy")


# USING CONVOLUTIONAL NETWORK

# Load the dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 256
embedding_dim = 64

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

# Get the tokenizer
tokenizer = info.features['text'].encoder

# Get the train-test split
train_data, test_data = dataset["train"], dataset["test"] 

# Displaying first 5 sentences and labels in train_data
for sentence, label in train_data.take(5):
    print(sentence.numpy())
    print(label.numpy())

# Shuffle the training data
# Why should we shuffling? Because we want to have a random order of the training data to reduce the bias of the model 
# and to escape local minima for gradient descent.
# Changing the value of affects how uniform the shuffling is: if is greater than the number of elements in the dataset,
# you get a uniform shuffle; if it is then you get no shuffling at all. In our example we have 25000 number of elements in the dataset 
# and we set BUFFER_SIZE = 10000. So we get a uniform shuffle of the dataset.
train_dataset = train_data.shuffle(BUFFER_SIZE)

# Batch and pad the datasets to the maximum length of the sequences 
train_dataset  = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=embedding_dim),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

model.summary()

history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

plot_metrics(history, "accuracy")
plot_metrics(history, "loss")


# LAB: SARCASM DATASET

# Load the JSON file
datastore = []
for line in open("/Users/ozanguner/VS_Projects/NLP_Tensorflow/datasets/Sarcasm_Headlines_Dataset.json", 'r'):
    datastore.append(json.loads(line))

sentences = []
labels = []
for row in datastore:
    sentences.append(row["headline"])
    labels.append(row["is_sarcastic"])

arr_sentences = np.array(sentences)
arr_labels =np.array(labels)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(arr_sentences, arr_labels, test_size=0.2, random_state=42)

# Tokenize the sentences
vocab_size = 10000
oov_token = "<OOV>"
maxlen = 120
padding_type = "post"
trunc_type = "post"
emb_dim = 64
lstm_dim = 32

tokenizer = Tokenizer(vocab_size, oov_token=oov_token)

tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index

seq_train = tokenizer.texts_to_sequences(x_train)
padded_train = pad_sequences(seq_train, maxlen=maxlen, padding=padding_type, truncating=trunc_type)

seq_test = tokenizer.texts_to_sequences(x_test)
padded_test = pad_sequences(seq_test, maxlen=maxlen, padding=padding_type, truncating=trunc_type)

# 1) BIDIRECTIONAL LSTM MODEL

bi_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])

bi_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

bi_model.summary()

bi_history = bi_model.fit(padded_train, y_train, epochs=5, validation_data=(padded_test, y_test))

plot_metrics(bi_history, "accuracy")
plot_metrics(bi_history, "loss")

# 2) CONVOLUTIONAL NEURAL NETWORK MODEL (Conv1D)
# Conv1D is a 1D convolutional layer that operates on the last axis of the input data.
# Conv1D is a type of convolution layer that processes the input data by applying a sequence of filters

conv_filters = 128
kernel_size = 5

conv_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, emb_dim, input_length=maxlen),
    tf.keras.layers.Conv1D(conv_filters, kernel_size, activation="relu"),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

conv_model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

conv_model.summary()

conv_history = conv_model.fit(padded_train, y_train, epochs=5, validation_data=(padded_test, y_test))

plot_metrics(conv_history, "accuracy")
plot_metrics(conv_history, "loss")

# IMPORTANT: 
 # This is how you need to set the Embedding layer when using pre-trained embeddings
    #    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=maxlen, weights=[embeddings_matrix], trainable=False)