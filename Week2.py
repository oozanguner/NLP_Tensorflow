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

# Embedding is the expression of a language or individual words in the given data as real-valued vectors in a less dimensional space.
# The main idea in Embedding is to represent each word in your vocabulary with vectors. These vectors have trainable weights so as your neural network learns,
#  words that are most likely to appear in a positive tweet will converge towards similar weights.

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb["train"], imdb["test"]

# Convert sentences and labels to numpy arrays
training_sentences = []
training_labels = []
for s, l in train_data:
    training_sentences.append(s.numpy().decode("utf8"))
    training_labels.append(l.numpy())


test_sentences = []
test_labels = []
for s, l in test_data:
    test_sentences.append(s.numpy().decode("utf8"))
    test_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
test_labels_final = np.array(test_labels)

# Now you can do the text preprocessing steps you've learned last week. You will tokenize the sentences and pad them to a uniform length.

vocab_size = 10000
oov = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

seq = tokenizer.texts_to_sequences(training_sentences)
max_word = max([len(t) for t in seq])
padded = pad_sequences(seq, maxlen=max_word, padding="post", truncating="post")

test_seq = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_seq, maxlen=max_word, padding="post", truncating="post")

# Building a model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=max_word),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=6, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
]
)

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ["accuracy"])
model.summary()

# Train the model
model.fit(padded, training_labels_final, batch_size=16, epochs=10, validation_data=(test_padded, test_labels_final))
model.evaluate(test_padded, test_labels_final)

# VISUALIZING THE EMBEDDINGS
embedding_layers = model.layers[0]
embedding_weights = embedding_layers.get_weights()[0]

embedding_weights.shape

# Get the index-word dictionary
reverse_index_word = tokenizer.index_word

out_v = io.open("vecs.tsv", "w", encoding="utf-8")
out_m = io.open("meta.tsv", "w", encoding="utf-8")

for word_count in range(1,vocab_size):              # we start from 1 because 0 represents OOV 
  if word_count not in reverse_index_word:
    continue
  else:
    word = reverse_index_word[word_count]
    vec = embedding_weights[word_count]
    io.open("vecs.tsv", "a", encoding="utf-8").write("\t".join([str(x) for x in vec]) + "\n")
    io.open("meta.tsv", "a", encoding="utf-8").write(word + "\n")

out_v.close()
out_m.close()

# Download the vecs and the meta files to https://projector.tensorflow.org/ then you can visualize the embeddings.



# BUILDING A CLASSIFIER FOR SARCASM DATASET
data = []
for line in open('datasets/Sarcasm_Headlines_Dataset.json', 'r'):
    data.append(json.loads(line))

headlines = []
labels = []
for item in data:
    headlines.append(item['headline'])
    labels.append(item['is_sarcastic'])

headlines_array = np.array(headlines)
labels_array = np.array(labels)

# Parameters
vocab_size = 1000
oov = "<OOV>"
max_word = 16
padding_type = "post"
truncate_type = "post"
emb_output_dim = 32

# Train-Test Split
train_percent  = 0.75
train_length = int(len(headlines_array) * train_percent)

train_headlines = headlines_array[:train_length]
train_labels = labels_array[:train_length]

# Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov)
tokenizer.fit_on_texts(train_headlines)
word_index = tokenizer.word_index
train_seq = tokenizer.texts_to_sequences(train_headlines)
train_padded = pad_sequences(train_seq, maxlen=max_word, padding=padding_type, truncating=truncate_type)

# Test Data
test_headlines = headlines_array[train_length:]
test_labels = labels_array[train_length:]
test_seq = tokenizer.texts_to_sequences(test_headlines)
test_padded = pad_sequences(test_seq, maxlen=max_word, padding=padding_type, truncating=truncate_type)

# Model
model = tf.keras.Sequential(
   [
   tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emb_output_dim, input_length=max_word),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(units=16, activation="relu"),
   tf.keras.layers.Dense(units=4, activation="relu"),
   tf.keras.layers.Dense(units=1, activation="sigmoid")
]
)

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ["accuracy"])
model.summary()

# Train and Evaulate
history = model.fit(train_padded, train_labels, batch_size=16, epochs=10, validation_data=(test_padded, test_labels))

plot_metrics(history, "loss")
plot_metrics(history, "accuracy")


# SUBWORD TOKENIZATION WITH THE IMDB DATASET
# This is an alternative to word-based tokenization which you have been using in the previous labs. 
# You will see how it works and its implications on preparing your data and training your model.

# 1)Word-Based Tokenization
# Take 2 training examples and print the text feature
imdb_plaintext, info_plaintext = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_plaintext, test_plaintext = imdb_plaintext["train"], imdb_plaintext["test"]

vocab_size = 8000
max_words = 32
padding_type = "post"
truncate_type = "post"
oov = "<OOV>"  

train_sentences = []
train_labels = []
for s, l in train_plaintext:
    train_sentences.append(s.numpy().decode("utf8"))
    train_labels.append(l.numpy())

train_sentences_final = np.array(train_sentences)
train_labels_final = np.array(train_labels)

test_sentences = []
test_labels = []
for s, l in test_plaintext:
   test_sentences.append(s.numpy().decode("utf8"))
   test_labels.append(l.numpy())

test_sentences_final = np.array(test_sentences)
test_labels_final = np.array(test_labels)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov)
tokenizer.fit_on_texts(train_sentences_final)
word_index = tokenizer.word_index
train_seq = tokenizer.texts_to_sequences(train_sentences_final)
train_padded = pad_sequences(train_seq, maxlen=max_words, padding=padding_type, truncating=truncate_type)

test_seq = tokenizer.texts_to_sequences(test_sentences_final)
test_padded = pad_sequences(test_seq, maxlen=max_words, padding=padding_type, truncating=truncate_type)

# 2)Subword Tokenization
# Take 2 training examples and print the subword feature
imdb_subwords, info_subwords = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

train_subwords, test_subwords = imdb_subwords["train"], imdb_subwords["test"]

for s,l in train_subwords.take(2):
    print(s.numpy())
    print(l.numpy())


# Description of features
info_subwords.features

# Encode the subwords
tokenizer_subwords = info_subwords.features["text"].encoder
tokenizer_subwords.subwords
# Example
sample_sentence = np.array(["I'm still thinking of you. I think you're a great person. I love you."])
# Encode using the plain-text  tokenizer (Word-based Tokenization)
tokenized_string = tokenizer.texts_to_sequences(sample_sentence)
original_string = tokenizer.sequences_to_texts(tokenized_string)[0]
# Encode using the subword tokenizer (Subword Tokenization)
tokenized_string_subwords = tokenizer_subwords.encode(sample_sentence[0])
original_string_subwords = tokenizer_subwords.decode(tokenizer_subwords.encode(sample_sentence[0]))

# Original String
for example in train_subwords.take(1):
  print(tokenizer_subwords.decode(example[0]))

# Encoded String
for example in train_subwords.take(1):
  print(tokenizer_subwords.encode(example[0]))

# Model
BUFFER_SIZE = 10000
BATCH_SIZE = 64
# Shuffle the training data
train_dataset = train_subwords.shuffle(BUFFER_SIZE)

# Batch and pad the datasets to the maximum length of the sequences
train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_subwords.padded_batch(BATCH_SIZE)

embedding_dim = 64

model = tf.keras.Sequential([
   tf.keras.layers.Embedding(tokenizer_subwords.vocab_size, embedding_dim),
   tf.keras.layers.GlobalAveragePooling1D(),
   tf.keras.layers.Dense(units=6, activation="relu"),
   tf.keras.layers.Dense(units=1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics = ["accuracy"])
model.summary()

history = model.fit(train_dataset, batch_size=8, epochs=10, validation_data=test_dataset)

plot_metrics(history, "loss")
plot_metrics(history, "accuracy")