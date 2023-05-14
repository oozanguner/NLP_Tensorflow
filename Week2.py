import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import io
import json
import matplotlib.pyplot as plt

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

def plot_metrics(history, metric="loss"):
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history["val_" + metric], label="val_" + metric)
    plt.legend()
    plt.show()

plot_metrics(history, "loss")
plot_metrics(history, "accuracy")
