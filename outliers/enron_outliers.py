#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
#remove the TOTAL outlier
data_dict.pop('TOTAL',0)

data = featureFormat(data_dict, features)

print (max(data[0:len(data), 0]))
print (max(data[0:len(data), 1]))

#A quick way to remove a key-value pair from a dictionary is the following line: dictionary.pop( key, 0 ) Write a line like this (you’ll have to modify the dictionary and key names, of course) and remove the outlier before calling featureFormat(). Now rerun the code, so your scatterplot doesn’t have this outlier anymore. Are all the outliers gone?
#data_dict.pop('TOTAL',0)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()




### your code below
#print('Data:',data)
# Find the maximum salary
max_salary = 0
for point in data:
    salary = point[0]
    if salary > max_salary:
        max_salary = salary
        
print('Max salary:',max_salary)

# Find the maximum bonus
max_bonus = 0
for point in data:
    bonus = point[1]
    if bonus > max_bonus:
        max_bonus = bonus

print('Max bonus:',max_bonus)

# Find the person with the maximum salary
for key, value in data_dict.items():
    if value['salary'] == max_salary:
        print('Person with max salary:',key)

# Find the person with the maximum bonus
for key, value in data_dict.items():
    if value['bonus'] == max_bonus:
        print('Person with max bonus:',key)




