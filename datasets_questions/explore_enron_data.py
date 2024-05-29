#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))
print("Pple Total = ",len( enron_data.keys()) )
print("Features = ",len(enron_data[list(enron_data.keys())[0]].keys()))
poi=[p for  p in enron_data if enron_data[p]['poi']==1]
print("Total POI = ",len(poi))


with open("../final_project/poi_names.txt", "r") as file:
    enron_name=file.read().strip()

lines=enron_name.split('\n')

# for line in lines:
#     print(line)
print("Names = ",len(lines)-2)

print("total value of the stock belonging to James Prentice",enron_data["Prentice James".upper()]["total_stock_value"])

print("email messages we have from Wesley Colwell to persons of interest",enron_data["Colwell Wesley".upper()]["from_this_person_to_poi"])


print("value of stock options exercised by Jeffrey K Skilling",enron_data["SKILLING JEFFREY K".upper()]["exercised_stock_options"])


#total_payments
print("who took home the most money Skilling ",enron_data["SKILLING JEFFREY K".upper()]["total_payments"])
print("who took home the most money LAY ",enron_data["LAY KENNETH L".upper()]["total_payments"])
print("who took home the most money ASTOW ",enron_data["FASTOW ANDREW S".upper()]["total_payments"])

#print(enron_data["SKILLING JEFFREY K".upper()])
salary=[enron_data[p]['salary'] for  p in enron_data if  not enron_data[p]['salary']=='NaN']
print("salary",len(salary))

emails=[enron_data[p]['email_address'] for  p in enron_data if  not enron_data[p]['email_address']=='NaN']
print("Emails ",len(emails))

#from tools.feature_format import featureFormat
#from ..tools import feature_format

# from ..tools import feature_format

# np_result = feature_format.featureFormat(enron_data,enron_data["FASTOW ANDREW S".upper()].keys())
# print(np_result,len(np_result))

total_payments=[enron_data[p]['total_payments'] for  p in enron_data if  enron_data[p]['total_payments']=='NaN' ]

print("total_payments_NaN",total_payments,len(total_payments),"percent ",len(total_payments)/len(enron_data)*100)

total_payments_poi=[enron_data[p]['total_payments'] for  p in enron_data if  enron_data[p]['total_payments']=='NaN' and enron_data[p]['poi']==1 ]

print(total_payments_poi,len(total_payments_poi)," percent ",len(total_payments_poi)/len(enron_data)*100)

#total_payments_poi=[enron_data[p]['total_payments'] for  p in enron_data if  enron_data[p]['total_payments']=='NaN' and enron_data[p]['poi']==1 ]

total_payments_poi_plus10=10
print(total_payments_poi_plus10,"Data set",len(enron_data)+10," percent ",(total_payments_poi_plus10+21)/(len(enron_data)+10)*100)


# import numpy
# import matplotlib
# matplotlib.use('agg')
#import matplotlib.pyplot as plt
# plt.clf()
# plt.scatter(ages_train, net_worths_train, color="b", label="train data")
# plt.scatter(ages_test, net_worths_test, color="r", label="test data")
# plt.plot(ages_test, reg.predict(ages_test), color="black")
# plt.legend(loc=2)
# plt.xlabel("ages")
# plt.ylabel("net worths")

