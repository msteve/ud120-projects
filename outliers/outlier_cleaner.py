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

    # Calculate the error   
    errors = (net_worths - predictions)**2
    #print('Errors squared ',errors)
    # Create a list of tuples with age, net_worth and error
    cleaned_data = list(zip(ages, net_worths, errors))
    # Sort the list by error
    cleaned_data.sort(key=lambda x: x[2])
    # Calculate the number of elements to keep
    n = int(len(cleaned_data) * 0.9)
    # Keep the first n elements
    cleaned_data = cleaned_data[:n]

    
    return cleaned_data

