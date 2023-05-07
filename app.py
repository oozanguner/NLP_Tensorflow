import tensorflow as tf
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10)

sentences = ["Merhaba ben ozan",
             "ozan ben, nasılsın"]

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

tokenizer.texts_to_sequences(sentences)

