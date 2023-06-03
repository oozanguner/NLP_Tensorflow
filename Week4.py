from functions import *
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional
import numpy as np
import requests

# 1 ) Ungraded Lab: Generating Text with Neural Networks

# Define the lyrics of the song
data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

# Splitting to lines 
corpus = data.lower().split("\n")

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index

# Define the total words. You add 1 for the index `0` which is just the padding token.
total_words = len(word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for c in range(1,len(token_list)):
        n_gram_sequence = token_list[:c+1]
        input_sequences.append(n_gram_sequence)

# Getting max length of input sequences
max_line_length = max([len(line) for line in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_line_length, padding="pre")


# Splitting inputs and labels
xs = input_sequences[:, :-1]
labels = input_sequences[:,-1]

# One hot encoding to labels
ys = to_categorical(labels, num_classes=total_words)

# Model
model = Sequential([
    Embedding(input_dim = total_words, output_dim=64, input_length=max_line_length-1),
    Bidirectional(LSTM(64, activation="relu")),
    Dropout(0.2),
    Dense(total_words, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

model.summary()

history = model.fit(xs, ys, epochs=500)

plt.plot(history.history["loss"])
plt.show();



# Prediction Exercise
seed_text = "Laurence went to Dublin"

# Define total words to predict
num_words_predict = 100

for i in tokenizer.texts_to_sequences([seed_text])[0]:
    print(tokenizer.index_word[i])

for p in range(num_words_predict):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], padding="pre", maxlen=max_line_length-1)

    probabilities = model.predict(token_list)

    # Selecting one of the first x highest probs prediction randomly to avoid frequent repetition.
    # This is not the most time efficient solution because it is always sorting the entire array even if you only need the top x.
    # We can also develop our own method of picking the next word.
    high_x_probs = 5
    highest_probs = np.argsort(probabilities)[0][-high_x_probs:]
    predicted = np.random.choice(highest_probs)

    # Ignore if index is 0 because that is just the padding.
    if predicted != 0:

        output_word = tokenizer.index_word[predicted]

        seed_text += f' {output_word}'


# 2 ) Ungraded Lab: Poetry
master = "https://raw.githubusercontent.com/https-deeplearning-ai/tensorflow-1-public/main/C3/W4/misc/Laurences_generated_poetry.txt"
req = requests.get(master)
req = req.text

corpus = req.lower().split("\n")

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index

# Define the total words. You add 1 for the index `0` which is just the padding token.
total_words = len(word_index) + 1

# Preprocessing
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1,len(token_list)):
        n_gram_sequences = token_list[:i+1]
        input_sequences.append(n_gram_sequences)


max_seq_length = max([len(s) for s in input_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding="pre")

xs = input_sequences[:,:-1]
labels = input_sequences[:, -1]
ys = to_categorical(labels, num_classes=total_words)

# Model
model = Sequential([
    Embedding(input_dim=total_words, output_dim=100, input_length=max_seq_length-1),
    Bidirectional(LSTM(150, activation="relu")),
    Dropout(0.2),
    Dense(total_words, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])

model.summary()

history = model.fit(xs,ys,epochs=100)

plt.plot(history.history["accuracy"])
plt.show();

# Generating Text
seed_text = "help me obi-wan kinobi youre my only hope"

next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding="pre")

    probabilities = model.predict(token_list)

    highest_probs = 3
    preds = np.argsort(probabilities)[0][-highest_probs:]
    predicted = np.random.choice(preds)

	# Ignore if index is 0 because that is just the padding.
    if predicted != 0:
		
		# Look up the word associated with the index. 
        output_word = tokenizer.index_word[predicted]

		# Combine with the seed text
        seed_text += f' {output_word}'