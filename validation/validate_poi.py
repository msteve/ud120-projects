#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

sort_keys = '../tools/python2_lesson13_keys.pkl'

data = featureFormat(data_dict, features_list,sort_keys=sort_keys)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
### You can use the "features" and "labels" variables to train a classifier
### and then make predictions on the test set.  You can also use
### the "data_dict" variable to look at the data for individual points.
print("Number of data points:", len(data_dict))
print("Number of features:", len(features_list) - 1)  # Exclude the label 'poi'
print("Features:", features_list[1:])  # Exclude the label 'poi'
print("First 5 data points:", features[:5])
print("First 5 labels:", labels[:5])
# Example of how to access a specific data point
print("Data point for 'METTS MARK':", data_dict['METTS MARK'])
# Example of how to access a specific label
print("Label for 'METTS MARK':", data_dict['METTS MARK']['poi'])
# Example of how to access a specific feature for 'METTS MARK'
print("Salary for 'METTS MARK':", data_dict['METTS MARK']['salary'])
# Example of how to access a specific feature for 'METTS MARK'
print("Total number of POIs:", sum(1 for v in data_dict.values() if v['poi']))

#Create a decision tree classifier (just use the default parameters), train it on all the data (you will fix this in the next part!), and print out the accuracy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()
# Train the classifier on the training data
clf.fit(X_train, y_train)
# Calculate the accuracy on the test set  
accuracy = clf.score(X_test, y_test)
print("Accuracy of the decision tree classifier:", accuracy)

