#!/usr/bin/python3

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    

import os
import sys
import joblib
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
dictionary = joblib.load( open("../final_project/final_project_dataset_modified.pkl", "rb") )

#print ("dictionary == ",dictionary)

### list the features you want to look at--first item in the 
### list will be the "target" feature  long_term_incentive
features_list = ["bonus", "salary"]
#features_list = ["bonus", "long_term_incentive"]  #We have a better score when using long-term incentive to predict someone's bonus, 
#which translates to a better fit.
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = '../tools/python2_lesson06_keys.pkl')
#print("Data for", data)
target, features = targetFeatureSplit( data )
#print("target for", target)
#print("features for", features)

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn.linear_model import LinearRegression
reg=LinearRegression()

#print("feature_train",feature_train)
#print("target_train",target_train)
reg.fit(feature_train,target_train)

print(features_list[1],"Score Train == ",reg.score(feature_train,target_train))
print(features_list[1],"Coef == ",reg.coef_)
print(features_list[1],"Intercept == ",reg.intercept_)


print(features_list[1],"Score Test == ",reg.score(feature_test,target_test))







### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
plt.xlabel(features_list[1])

reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
print(features_list[1],"New Data Coef == ",reg.coef_)
print(features_list[1],"New Data Intercept == ",reg.intercept_)

plt.ylabel(features_list[0])
plt.legend()
plt.show()
