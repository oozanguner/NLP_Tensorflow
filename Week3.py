from functions import *
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import io
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 

# USING RNNs, LSTM with BIDIRECTIONAL LAYER

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
