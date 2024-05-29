#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

     # Calculate the residual errors
    errors = abs(predictions - net_worths)

    # Combine ages, net_worths, and errors into a list of tuples
    data = list(zip(ages, net_worths, errors))
   # print(data)

    # Sort the data by errors in ascending order
    data_sorted = sorted(data, key=lambda x: x[2])
    #print(data_sorted)

     # Determine the number of points to keep (90% of the data)
    keep_count = int(len(data) * 0.9)

    # Keep only the 90% of points with the smallest errors
    cleaned_data = data_sorted[:keep_count]
    
    return cleaned_data

