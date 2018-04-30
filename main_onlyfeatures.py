import torch
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm


from dataset_reader import *

np.random.seed(0)
# Uncomment the two lines below if this is the first time you are running
labels_directory = "data/labels"
# preprocess_data(labels_directory)

train_ratio = 0.8
labels_arr, features_arr = process_mats(labels_directory)
mask = split_data(train_ratio, labels_arr)


# print labels_arr, features_arr
X_train = features_arr[mask["train"]]
y_train = labels_arr[mask["train"]]
X_test = features_arr[mask["test"]]
y_test = labels_arr[mask["test"]]

print X_train.shape, X_test.shape

# classifier = KNeighborsClassifier(n_neighbors=8) 
classifier = svm.LinearSVC(C=0.1)
# classifier = svm.SVC(kernel='rbf', gamma=0.5, C=0.1)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) 
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))  
