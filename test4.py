import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

f = open("Topic_prediction_data_train.txt", "r+")
raw_lines = f.readlines()
f.close()

f = open("Topic_prediction_labels_train.txt", "r+")
raw_labels = f.readlines()
f.close()

split_pt = int(len(raw_lines) * 0.8)

training_data = np.empty(split_pt, dtype='object')
training_labels = np.zeros(split_pt)

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
	

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(training_data)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
clf = MultinomialNB().fit(train_tfidf, training_labels)
res_nb = clf.predict(count_vect.transform(testing_data))
print(np.mean(res_nb == np.array(testing_labels)))

svc = LinearSVC().fit(train_tfidf, training_labels)
res_svc = svc.predict(count_vect.transform(testing_data))
print(np.mean(res_svc == np.array(testing_labels)))

lr =  LogisticRegression(random_state=0).fit(train_tfidf, training_labels)
res_lr = lr.predict(count_vect.transform(testing_data))
print(np.mean(res_lr == np.array(testing_labels)))