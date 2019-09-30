import math
from random import shuffle
from k_fold_cross_val import k_fold_data_split

def data_byclass_separate(input_data):
    """Separates the training dataset points by class value. 
    Returns a map of class values to lists of data points.
    """
    data_byclass = {} 
    for row in input_data[1:]:
        if row[-1] in data_byclass:
            data_byclass[row[-1]].append(row)
        else:
            data_byclass[row[-1]] = [row]
    return data_byclass

def mean(values):
    """Returns a mean of a set of data values.
    """
    return sum(values)/float(len(values))

def stdv(values, mean_value):
    """Returns a standard deviation of a set of data values.
    """
    N = len(values)
    sum_sqr = 0
    for x in values:
        sum_sqr += (x - mean_value)**2
    return math.sqrt(sum_sqr/float((N-1)))

def class_freq(data_separated):
    freq = {}
    count = 0
    for class_key in data_separated:
        freq[class_key] = len(data_separated[class_key])
        count += len(data_separated[class_key])
    for key in freq:
	    freq[key] /= count
    return freq

def statistic_byclass(data_separated):
    """Calculates the mean and standard deviation for each data set feature and class value.
    Returns a map of class values to feature mean and standard deviation.
    """
    stat_byclass = {}
    for class_key in data_separated:
        columns_stat = []
        for col_values in list(zip(*data_separated[class_key])):
            mean_value = mean(col_values)
            stdv_value = stdv(col_values, mean_value)
            columns_stat.append((mean_value, stdv_value))
        columns_stat.pop(-1)  

        stat_byclass[class_key] = columns_stat
    return stat_byclass

def gauss_probability(x, mean_value, stdv_value):
    """Calculates Gaussian Probability Density Function for a given feature.
    """
    exponent = math.exp(-(x-mean_value)**2/(2*stdv_value**2))
    return exponent * (1/(math.sqrt(2*math.pi)*stdv_value)) 

def gauss_class_probability(data, stat_byclass, class_frequency):
    """Returns Gaussian Probability Density Function for a given feature and each class value.
    """
    probabilities = {} 
    for class_key in stat_byclass:
        P = class_frequency[class_key]
        for i in range(len(data)-1):
            P *= gauss_probability(data[i], stat_byclass[class_key][i][0], stat_byclass[class_key][i][1])
        probabilities[class_key] = P
    return probabilities

def nb_class_probability(data, data_separated):
    """Returns Posterior Probability for each class value, calculated using Bayes’ Theorem.
    """
    probabilities = {}
    class_frequency = class_freq(data_separated) 
    for class_key in data_separated:
        P = class_frequency[class_key]
        col_values = list(zip(*data_separated[class_key]))
        for i in range(len(col_values)-1):
            if data[i] not in col_values[i]:
                P *= 1/len(col_values[i])
            else:
                P *= col_values[i].count(data[i])/len(col_values[i])
        probabilities[class_key] = P
    return probabilities

def prediction(probabilities):
    """Returns a prediction of the class for a given data point. 
    """
    sum_P = sum(probabilities.values())
    for class_key in probabilities:
        probabilities[class_key] /= sum_P
    return sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[0][0]   

def nb_gauss_predict(test_data, train_set):
    """Returns a prediction of the class for a given data point. 
    """
    train_separated = data_byclass_separate(train_set)
    train_statistic = statistic_byclass(train_separated)
    class_frequency = class_freq(train_separated)
    class_probabilities = gauss_class_probability(test_data, train_statistic, class_frequency)
    class_predicted = prediction(class_probabilities)
    return class_predicted 

def nb_predict(test_data, train_set):
    train_separated = data_byclass_separate(train_set)
    class_probabilities = nb_class_probability(test_data, train_separated)
    class_predicted = prediction(class_probabilities)
    return class_predicted

def nb_accuracy_score(test_set, predictions):
    """Returns accuracy of predicting model in %.
    """
    accurate_pred = 0 
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            accurate_pred += 1
    return (round(accurate_pred/len(test_set) * 100.0, 4))

def nb_k_fold_cross_val(input_data, k, predict_function):
    """Splits the input data set into a k number of sections/folds where each fold is used as a testing set 
    and the rest of input data are used as training set.
    Returns a mean accuracy of predicting model used.

    Prameters:
    
    k (int): Number of folds \n
    """
    k_fold_sets = k_fold_data_split(input_data, k)
    all_accuracy = []
    for k_set in k_fold_sets:
        test_set = k_set[0]
        train_set = k_set[1]
        all_predictions = []
        for x in range(len(test_set)):
            prediction = predict_function(test_set[x], train_set)
            all_predictions.append(prediction)
        accuracy_score = nb_accuracy_score(test_set, all_predictions)
        all_accuracy.append(accuracy_score)
    
    mean_accuracy = sum(all_accuracy) / len(all_accuracy)
    return mean_accuracy
