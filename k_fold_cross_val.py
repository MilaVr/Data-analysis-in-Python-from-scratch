from knn import eucli_dist, k_neighbors, predict, accuracy_score
from copy import deepcopy

def k_fold_data_split(input_data, k):
    """Splits the input data set into k number of sections/folds, where each fold is used as testing set. 
    Returns a list of sets (type: tuple) each containing test and training data set. 

    Prameters:

    k (int): Number of folds   
    """
    input_data = input_data[1:]
    step = len(input_data)//k
    test_sets = []
    k_fold_data_set = []
    for i in range(0, len(input_data), step):
        if len(test_sets) < k:
            test_sets.append(input_data[i:i+step])
        else:
            test_sets[-1] += input_data[i:i+step]
    
    for test_set in test_sets:
        train_set = deepcopy(input_data)
        for test_data in test_set:
            train_set.remove(test_data)
        k_fold_data_set.append((test_set, train_set))
    
    return k_fold_data_set
        
def k_fold_cross_val(input_data, K, k):
    """Splits the input data set into a k number of sections/folds where each fold is used as a testing set 
    and the rest of input data are used as training set.
    Returns a mean accuracy of predicting model used.

    Prameters:
    
    k (int): Number of folds \n
    K (int): Number of nearest neighbors
    """
    k_fold_sets = k_fold_data_split(input_data, k)
    all_accuracy = []
    for k_set in k_fold_sets:
        test_set = k_set[0]
        train_set = k_set[1]
        all_predictions = []
        for x in range(len(test_set)):
            neighbors = k_neighbors(test_set[x][:-1], train_set, K)
            prediction = predict(neighbors)
            all_predictions.append(prediction)
        accuracy = accuracy_score(test_set, all_predictions)
        all_accuracy.append(accuracy)
    
    mean_accuracy = sum(all_accuracy) / len(all_accuracy)

    return mean_accuracy


        

    




    
        
        

