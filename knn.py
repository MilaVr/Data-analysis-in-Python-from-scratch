import math

def eucli_dist(data1, data2): 
    """Returns straight-line distance between two points in n-dimensional space.
    
    Parameters:

    data1 (list): Feature values for data1 \n
    data2 (list): Feature values for data2 
    """
    distance = 0
    for x1, x2 in zip(data1, data2):
        distance += (x1 - x2)**2
    return math.sqrt(distance)

def k_neighbors(test_data, train_set, k):
    """Returns k nearest training data points in the feature space.

    Parameters:

    test_data (list): Feature values for test data point \n
    train_set (list): Feature values for train data set
    """
    distances = []
    for x in range(len(train_set)):
        distance = eucli_dist(test_data, train_set[x])
        distances.append((train_set[x], distance))
    distances.sort(key = lambda tr_dis: tr_dis[1])
    k_neighbors = [distances[x][0] for x in range(k)]
    return k_neighbors

def predict(k_neighbors):
    """Returns a prediction of the class for a given data point. 
    An object is classified by a plurality vote of its neighbors, 
    with the object being assigned to the class most common among its k nearest neighbors using k_neighbors.
    """
    prediction_classes = {}
    for neighbor in k_neighbors:
        if neighbor[-1] in prediction_classes:
            prediction_classes[neighbor[-1]] += 1
        else:
            prediction_classes[neighbor[-1]] = 1
    return sorted(prediction_classes.items(), key = lambda key_value: key_value[1], reverse = True)[0][0]

def accuracy_score(test_set, predictions):
    """Returns accuracy of predicting model in %.
    """
    accurate_pred = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predictions[x]:
            accurate_pred += 1
    return (round(accurate_pred/len(test_set) * 100.0, 4))
    

    
   