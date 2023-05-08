import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Embedding is the expression of a language or individual words in the given data as real-valued vectors in a less dimensional space.
# The main idea in Embedding is to represent each word in your vocabulary with vectors. These vectors have trainable weights so as your neural network learns,
#  words that are most likely to appear in a positive tweet will converge towards similar weights.

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb["train"], imdb["test"]

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
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=max_word),
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