import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

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

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(training_data)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
tf = tf_vectorizer.fit_transform(training_data)
tf_feature_names = tf_vectorizer.get_feature_names()


nmf = NMF(n_components=10)
nmf.fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_components=10)
lda.fit(tf)

tfidf_test = tfidf_vectorizer.transform(testing_data)
tf_test = tf_vectorizer.transform(testing_data)

tfidf_res_unormalized = np.matrix(nmf.transform(tfidf_test))
tf_res_unormalized = np.matrix(lda.transform(tf_test))

#print(tfidf_res_unormalized)
#print(tf_res_unormalized)

doc_topic_dist = tf_res_unormalized/tf_res_unormalized.sum(axis=1)
ret = doc_topic_dist.argmax(axis=1)

correct = 0
for i in range(len(ret)):
	if(np.isclose(ret[i], testing_labels[i])):
		correct += 1
	
print(correct)
print(correct/len(ret))
