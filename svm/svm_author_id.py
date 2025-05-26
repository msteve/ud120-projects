#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# print("features_train shape:", features_train)
# print("features_test shape:", features_test)
# print("labels_train shape:", labels_train)
# print("labels_test shape:", labels_test)



##me training
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

svm = svm.SVC(kernel='linear', C=1.0, random_state=42)
# Train the SVM classifier
svm.fit(features_train, labels_train)
# Make predictions on the test set
predictions = svm.predict(features_test)
# Calculate accuracy
accuracy = accuracy_score(labels_test, predictions)
print("Accuracy:", accuracy)
# Print classification report
print("Classification Report:")
print(classification_report(labels_test, predictions))
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(labels_test, predictions))
# Print F1 score
print("F1 Score:", f1_score(labels_test, predictions))
# Print precision score
print("Precision Score:", precision_score(labels_test, predictions))
# Print recall score
print("Recall Score:", recall_score(labels_test, predictions))
# Print the number of features      


#########################################################
### your code goes here ###


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
