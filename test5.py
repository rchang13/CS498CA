import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


#Read all the data and labels from the input files.
f = open("Topic_prediction_data_train.txt", "r+")
raw_lines = f.readlines()
f.close()

f = open("Topic_prediction_labels_train.txt", "r+")
raw_labels = f.readlines()
f.close()

#Among all the input lines, the first 95% will be used as training, and the last 5% will be used as testing
split_pt = int(len(raw_lines) * 0.95)

training_data = np.empty(split_pt, dtype='object')
training_labels = np.zeros(split_pt)

#For all the input data, ignore everything else and only take the textual description and ignore everything else. 
for i in range(split_pt):
	words = (raw_lines[i].split("\t"))[0]
	training_data[i] = words
	
	training_labels[i] = int(raw_labels[i])

	
testing_data = np.empty(len(raw_lines)-split_pt, dtype='object')
testing_labels = np.zeros(len(raw_labels)-split_pt)

for i in range(split_pt, len(raw_labels)):
	words = (raw_lines[i].split("\t"))[0]
	testing_data[i-split_pt] = words
	
	testing_labels[i-split_pt] = int(raw_labels[i])



max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(training_data) # only fit on train

x_train = tokenize.texts_to_matrix(training_data)
x_test = tokenize.texts_to_matrix(testing_data)

encoder = LabelEncoder()
encoder.fit(training_labels)
y_train = encoder.transform(training_labels)
y_test = encoder.transform(testing_labels)

num_classes = 10
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 2

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)