import csv
import random
from random import shuffle
import math
from copy import deepcopy
from data_prevew import data_shape, data_head, data_tail, data_info, data_load
from data_prep import data_type_convert, label_encoding
from knn import eucli_dist, k_neighbors, predict, accuracy_score
from k_fold_cross_val import k_fold_data_split, k_fold_cross_val
from nb import data_byclass_separate, statistic_byclass, mean, stdv, gauss_probability, gauss_class_probability, prediction, nb_gauss_predict, nb_accuracy_score, nb_k_fold_cross_val

iris_data_set = data_load(r"C:\Users\milav\Desktop\iris.csv")
data_type_convert(iris_data_set)

data_head(iris_data_set)
data_info(iris_data_set)

old_values = ["Setosa", "Versicolor", "Virginica"]
new_values = [0,1,2]
label_encoding(iris_data_set, "variety",old_values, new_values)

field_names = [iris_data_set[0]]
data_set = iris_data_set[1:]
shuffle(data_set)

iris_data_set = field_names + data_set

# NAIVE BAYES MODEL TRAINING
print("NAIVE BAYES MODEL TRAINIG ACCURACY")
test_data_set = []
train_data_set = []
n = int((len(iris_data_set) - 1) * 0.3)
idx_test = random.sample(range(1, len(iris_data_set)-1), n)
for idx in range(1, len(iris_data_set)-1):
    if idx in idx_test:
        test_data_set.append(iris_data_set[idx])
    else:
        train_data_set.append(iris_data_set[idx])

all_predictions = []
for x in range(len(test_data_set)):
    prediction = nb_gauss_predict(test_data_set[x], train_data_set)
    all_predictions.append(prediction)
accuracy_score_nb = nb_accuracy_score(test_data_set, all_predictions)
print(accuracy_score_nb)
print()

# k-fold-cross-validation
print("K-FOLD-CROSS-VALIDATION - NAIVE BAYES")
accuracy = round(nb_k_fold_cross_val(iris_data_set, 10, nb_gauss_predict), 2)
print(f"Mean accuracy: {accuracy} % ")
print() 

# KNN MODEL TRAINIG
print("K-FOLD-CROSS-VALIDATION - KNN")
k_accuracy = []
for K in range(1, 31):
    accuracy = round(k_fold_cross_val(iris_data_set, K, 10), 2)
    k_accuracy.append(accuracy)
    print(f"K = {K:<8} Mean accuracy: {accuracy} % ")
print(round(sum(k_accuracy)/len(k_accuracy), 2))
print()





