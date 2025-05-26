#!/usr/bin/python3

""" 
    Skeleton code for k-means clustering mini-project.
"""

import os
import joblib
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)
#print("Data dict keys:",data_dict)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

values = [
    person["exercised_stock_options"]
    for person in data_dict.values()
    if person["exercised_stock_options"] != 'NaN'
]

print("Maximum:", max(values))
print("Minimum:", min(values))


rescaled_values = scaler.fit_transform(numpy.array(values).reshape(-1, 1))
print("Rescaled values:", rescaled_values)

print("rescaled_values Maximum:", max(rescaled_values))
print("rescaled_values Minimum:", min(rescaled_values))
stock1m= 1000000
rescaled_stock1m = scaler.transform([[stock1m]])
print("Rescaled value for 1000000:", rescaled_stock1m)

values = [
    person["salary"]
    for person in data_dict.values()
    if person["salary"] != 'NaN'
]

print("Salary Maximum:", max(values))
print("Salary Minimum:", min(values))

rescaled_values = scaler.fit_transform(numpy.array(values).reshape(-1, 1))
print("Rescaled values:", rescaled_values)

print("rescaled_values Maximum:", max(rescaled_values))
print("rescaled_values Minimum:", min(rescaled_values))
stock200= 200000
rescaled_sal200= scaler.transform([[stock200]])
print("Rescaled value for 200000:", rescaled_sal200)


exit()

### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3= "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2,feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
#for f1, f2 in finance_features:
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.show()


### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
pred = kmeans.fit_predict(finance_features)
#print(pred)


#Just Me..
# Plot each point with color based on POI status

# for i, (f1, f2) in enumerate(finance_features):
#     if poi[i]:
#         plt.scatter(f1, f2, color='r', marker='*', s=100)  # POI in red with a star
#     else:
#         plt.scatter(f1, f2, color='b')  # non-POI in blue

# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Clusters with POIs Highlighted")
# plt.show()

#end of Just Me



### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named pred found, no clusters to plot")
