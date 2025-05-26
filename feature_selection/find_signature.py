#!/usr/bin/python3

import joblib
import numpy as np
np.random.seed(42)



### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = joblib.load( open(words_file, "rb"))
authors = joblib.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

# print("features_train shape:", features_train[:10])
#print("features_test shape:", features_test)
#print("labels_train shape:", labels_train)
#print("labels_test shape:", labels_test)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()

#print("features_train 2:", features_train[:10])

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]



### your code goes here

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42, min_samples_split=40)
# Train the classifier
clf.fit(features_train, labels_train)
# Make predictions on the test set
predictions = clf.predict(features_test)
print("Predictions:", predictions)
# Calculate accuracy
accuracy= clf.score(features_test, labels_test)
print("Accuracy:", accuracy)
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


importances = clf.feature_importances_
important_features = [(i, imp) for i, imp in enumerate(importances) if imp > 0.2]
print("Very important features (index, importance):", important_features)

# Map to actual words
for i, importance in important_features:
    print(f"{importance:.3f} => {vectorizer.get_feature_names_out()[i]}")


threshold = 0.2

# Get the feature names from the vectorizer
feature_names = vectorizer.get_feature_names_out()

for i, importance in enumerate(importances):
    if importance > threshold:
        print(f"Feature #{i} ({feature_names[i]}) has importance {importance:.3f}")




# Get index of the most important feature
# importances = clf.feature_importances_
# print("Feature importances:", importances)


# most_important_index = np.argmax(importances)
# most_important_score = importances[most_important_index]
# most_important_word = feature_names[most_important_index]

# # Print result
# print(f"Most important feature index: {most_important_index}")
# print(f"Importance score: {most_important_score:.3f}")
# print(f"Associated word: {most_important_word}")