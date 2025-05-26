#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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




#########################################################
### your code goes here ###


#########################################################
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
# Train the classifier
clf.fit(features_train, labels_train)
# Make predictions on the test set
predictions = clf.predict(features_test)
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
print("Number of features:", features_train.shape[1])
# Print the number of features used by the Decision Tree
print("Number of features used by the Decision Tree:", clf.n_features_)



