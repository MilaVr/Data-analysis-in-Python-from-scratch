import csv
import random
from random import shuffle
import math
from copy import deepcopy
from data_prevew import data_load, data_shape, data_head, data_tail, data_info
from data_prep import data_type_convert, column_remove, label_encoding, one_hot_encoding, median, column_median, column_median_1, column_median_2
from data_analysis import categorical_analysis, numerical_analysis
from knn import eucli_dist, k_neighbors, predict, accuracy_score
from k_fold_cross_val import k_fold_data_split, k_fold_cross_val
from nb import data_byclass_separate, class_freq, nb_class_probability, prediction, nb_predict, nb_accuracy_score, nb_k_fold_cross_val

def age_col_fill(input_data, all_median):
    for row in input_data[1:]:
        idx_age = input_data[0].index("Age")
        idx_sex = input_data[0].index("Sex")
        idx_pclass = input_data[0].index("Pclass")

        sex = row[idx_sex]
        pclass = row[idx_pclass]
        
        if row[idx_age] == "":
            row[idx_age] = all_median[sex][pclass - 1]

def age_bands(input_file):
    idx = input_file[0].index("Age")
    for row in input_file[1:]:
        if row[idx] <= 16:
            row[idx] = 0
        elif 16 < row[idx] <= 32:
            row[idx] = 1
        elif 32 < row[idx] <= 48:
            row[idx] = 2
        elif 48 < row[idx] <= 64:
            row[idx] = 3
        elif 64 < row[idx]:
            row[idx] = 4

def fare_bands(input_file):
    idx = input_file[0].index("Fare")
    for row in input_file[1:]:
        if row[idx] <= 7.91:
            row[idx] = 0
        elif 7.91 < row[idx] <= 15.9:
            row[idx] = 1
        elif 15.9 < row[idx] <= 31:
            row[idx] = 2
        elif 31 < row[idx]:
            row[idx] = 3

def titles(input_data):
    titles = []
    idx = input_data[0].index("Name")
    for row in input_data[1:]:
        title = row[idx].split(",")[1]
        title = title[1:title.find(".")+1]
        titles.append(title)
        row[idx] = title

    title_count = {}
    for title in titles:
        if title not in title_count:
            title_count[title] = 1
        else:
            title_count[title] += 1
    return title_count

# LOADING AND EXAMINATION OF DATA
titanik_train_set = data_load(r"C:\Users\milav\Desktop\train.csv")
titanik_test_set = data_load(r"C:\Users\milav\Desktop\test.csv")

data_type_convert(titanik_train_set)
data_type_convert(titanik_test_set)

# DATA PREPROCESSING 
columns_to_delete = ["PassengerId", "Ticket", "Cabin", "SibSp", "Parch"]
column_remove(titanik_train_set, columns_to_delete)    
column_remove(titanik_test_set, columns_to_delete[1:])    

curent_values_sex = ["male", "female"]
new_values_sex = [0, 1]
label_encoding(titanik_train_set, "Sex", curent_values_sex, new_values_sex)
label_encoding(titanik_test_set, "Sex", curent_values_sex, new_values_sex)

# Filling "Embarked" column in train data set with most comon value "S"
curent_values_emb = ["S", "C", "Q", ""]
new_values_emb = [0, 1, 2, 0]
label_encoding(titanik_train_set, "Embarked", curent_values_emb, new_values_emb)
label_encoding(titanik_test_set, "Embarked", curent_values_emb, new_values_emb)

# Filling "Age" column with median values and bining 
all_median_train = [[1,2,3], [1,2,3]]
all_median_test = [[1,2,3], [1,2,3]]

for x in range(2):
    for y in range(3):
        all_median_train[x][y] = column_median_2(titanik_train_set, "Age", "Sex", "Pclass", x, y+1)
        all_median_test[x][y] = column_median_2(titanik_test_set, "Age", "Sex", "Pclass", x, y+1)

age_col_fill(titanik_train_set, all_median_train)
age_col_fill(titanik_test_set, all_median_test)

age_bands(titanik_train_set)
age_bands(titanik_test_set)

# Filling one missing column value "Fare" in test data set and bining
for row in titanik_test_set[1:]:
    idx = titanik_test_set[0].index("Fare")
    if row[idx] == "":
        row[idx] = 35.62781

fare_bands(titanik_train_set)
fare_bands(titanik_test_set)

# Titles analysis in "Name" column 
titles_train = titles(titanik_train_set)
titles_test = titles(titanik_test_set)
for title in titles_train:
    print(f"{title:<13} {titles_train[title]}")
print()
for title in titles_test:
    print(f"{title:<13} {titles_test[title]}")
print() 

rare = ["Jonkheer.", "the Countess.", "Capt.", "Col.", "Sir.", "Lady.", "Major.", "Dr.", "Rev.", "Don.", "Dona."]
comon = ["Mlle.", "Miss.", "Ms.", "Mme.", "Mrs.", "Mr.", "Master."]

label_encoding(titanik_train_set, "Name", rare, [0]*len(rare))
label_encoding(titanik_train_set, "Name", comon, [1,1,1,2,2,3,4])
label_encoding(titanik_test_set, "Name", rare, [0]*len(rare))
label_encoding(titanik_test_set, "Name", comon, [1,1,1,2,2,3,4])

# Moving column "Survived" to the end
for row in titanik_train_set:
    row.append(row.pop(0))

# Moving column "PassengerId" to the end
for row in titanik_test_set:
    row.append(row.pop(0))

data_head(titanik_train_set)

# NAIVE BAYES MODEL TRAINING
print("NAIVE BAYES MODEL TRAINIG ACCURACY")
test_data_set = []
train_data_set = []
n = int((len(titanik_train_set) - 1) * 0.3)
idx_test = random.sample(range(1, len(titanik_train_set)-1), n)
for idx in range(1, len(titanik_train_set)-1):
    if idx in idx_test:
        test_data_set.append(titanik_train_set[idx])
    else:
        train_data_set.append(titanik_train_set[idx])

all_predictions = []
for x in range(len(test_data_set)):
    prediction = nb_predict(test_data_set[x], train_data_set)
    all_predictions.append(prediction)
accuracy_score_nb = nb_accuracy_score(test_data_set, all_predictions)
print(f"Accuracy: {accuracy_score_nb} %")
print()

# k-fold-cross-validation
print("K-FOLD-CROSS-VALIDATION - NAIVE BAYES")    
accuracy = round(nb_k_fold_cross_val(titanik_train_set, 10, nb_predict), 2)
print(f"Mean accuracy: {accuracy} % ")
print()  

#KNN MODEL TRAINIG
print("K-FOLD-CROSS-VALIDATION - KNN")
k_accuracy = []
for K in range(1, 31):
    accuracy = round(k_fold_cross_val(titanik_train_set, K, 10), 2)
    k_accuracy.append(accuracy)
    print(f"K = {K:<8} Mean accuracy: {accuracy} % ")
print("Sum. accuracy", round(sum(k_accuracy)/len(k_accuracy), 2))
print()

#PREDICTIONS FOR TEST DATA SET USING NAIVE BAYES MODEL
test_set = titanik_test_set[1:]
train_set = titanik_train_set[1:]
all_predictions = []
for x in range(len(test_set)):
    prediction = nb_predict(test_set[x], train_set)
    all_predictions.append(prediction)
    print(f"PassengerId: {test_set[x][-1]}     Predicted: {prediction}")
