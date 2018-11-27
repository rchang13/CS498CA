import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

data_file_path = sys.argv[1]
label_file_path = sys.argv[2]

#Read all the data and labels from the input files.
f = open(data_file_path, "r+")
raw_lines = f.readlines()
f.close()

f = open(label_file_path, "r+")
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

#We convert the training data (as well as the testing data in the code below) into a matrix of token count, and then transform the matrix to a term-frequency representation.
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(training_data)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)

#The Naive Bayes model
clf = MultinomialNB().fit(train_tfidf, training_labels)
res_nb = clf.predict(count_vect.transform(testing_data))
print("Naive Bayes accuracy:")
print(np.mean(res_nb == np.array(testing_labels)))

#The linear SVC model
svc = LinearSVC().fit(train_tfidf, training_labels)
res_svc = svc.predict(count_vect.transform(testing_data))
print("LinearSVC accuracy:")
print(np.mean(res_svc == np.array(testing_labels)))

#The linear SDG model
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, random_state=42, max_iter=100).fit(train_tfidf, training_labels)
res_sgd = sgd.predict(count_vect.transform(testing_data))
print("SGD accuracy:")
print(np.mean(res_sgd == np.array(testing_labels)))

#The logistic regression model
lr =  LogisticRegression(random_state=0).fit(train_tfidf, training_labels)
res_lr = lr.predict(count_vect.transform(testing_data))
print("Logistic Regression accuracy:")
print(np.mean(res_lr == np.array(testing_labels)))

#The K nearest neighbor model. It cannot be run due to memory errors.
#knn = KNeighborsClassifier(n_neighbors = 2).fit(train_tfidf, training_labels)  
#knn_predictions = knn.predict(count_vect.transform(testing_data))
#print(np.mean(knn_predictions == np.array(testing_labels)))

#The decision tree model
dtree_model = DecisionTreeClassifier(max_depth = 6).fit(train_tfidf, training_labels)
dtree_predictions = dtree_model.predict(count_vect.transform(testing_data))
print("Dtree accuracy:")
print(np.mean(dtree_predictions == np.array(testing_labels)))

print("As you can see, LinearSVC has the best performance. We select LinearSVC over all other methods.")

f = open("output.txt", "w+")
for num in res_svc:
	f.write(str(int(num)) + "\n")

