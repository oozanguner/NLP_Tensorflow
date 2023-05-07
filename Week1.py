import tensorflow as tf
import pandas as pd
import json
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# NLP in Tensorflow (Coursera)
# Example 1:
tokenizer = Tokenizer(num_words=10)

sentences = ["I love my cat",
             "You love your cat!"]

# Tokenize the sentences
tokenizer.fit_on_texts(sentences)

# Get the word index dict
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

# Example 2:
tokenizer = Tokenizer(num_words=100, oov_token="<oov>")      # If the tokenizer see a word that didn't shown at the begining of the process, you can determine it using oov_token hyperparameter.(out-of-vocabulary)
# If oov_token isn't used, then texts_to_sequences return nothing about unseen word in the test data.

sentences = ["I really love my dog",
             "Do you love your cat?"]

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_sentence = ["I don't love rabbits",
                 "How are you today?"]

test_sequences = tokenizer.texts_to_sequences(test_sentence)

# Padding 
# you will usually need to pad the sequences into a uniform length because that is what your model expects. You can use the pad_sequences for that
# Example 3:
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
sentences = ["We live in Barcelona",
            "I really love Spain",
            "Are you Spanish?"
            "Do you think that Madrid is more livable than Barcelona?"]

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
# pad the sequences to get a uniform length
padded  = pad_sequences(sequences, maxlen=15, padding="post", truncating="post")



# SARCASM DATASET
data = []
for line in open('datasets/Sarcasm_Headlines_Dataset.json', 'r'):
    data.append(json.loads(line))

link = []
text = []
label = []
for line in data:
    link.append(line["article_link"])
    text.append(line["headline"])
    label.append(line["is_sarcastic"])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index 

seq = tokenizer.texts_to_sequences(text)

padded = pad_sequences(seq, padding="post", truncating="post")