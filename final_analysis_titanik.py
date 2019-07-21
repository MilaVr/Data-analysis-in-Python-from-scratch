import csv
import random
import math
from copy import deepcopy
from data_prevew import data_shape, data_head, data_tail, data_info, data_load
from data_prep import data_type_convert, label_encoding, one_hot_encoding, column_remove, median,column_median_2
from data_analysis import categorical_analysis, numerical_analysis
from data_normalisation import mean, std, data_norm
from knn import eucli_dist, k_neighbors, predict, accuracy_score
from k_fold_cross_val import k_fold_data_split, k_fold_cross_val

def age_col_fill(input_data, all_median):
    for row in input_data[1:]:
        idx_age = input_data[0].index("Age")
        idx_sex = input_data[0].index("Sex")
        idx_pclass = input_data[0].index("Pclass")

        sex = row[idx_sex]
        pclass = row[idx_pclass]
        
        if row[idx_age] == "":
            row[idx_age] = all_median[sex][pclass - 1]

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
titanik_train_set = data_load("train.csv")
titanik_test_set = data_load("test.csv")

data_shape(titanik_train_set)
data_head(titanik_train_set)
data_tail(titanik_train_set)

data_shape(titanik_test_set)
data_head(titanik_test_set)
data_tail(titanik_test_set)

data_type_convert(titanik_train_set)
data_type_convert(titanik_test_set)

data_info(titanik_train_set)
data_info(titanik_test_set)

# Analysis of input data set
categorical_analysis(titanik_train_set, "Sex", "Survived")
categorical_analysis(titanik_train_set, "Pclass", "Survived")
categorical_analysis(titanik_train_set, "SibSp", "Survived")
categorical_analysis(titanik_train_set, "Parch", "Survived")
categorical_analysis(titanik_train_set, "Embarked", "Survived")

numerical_analysis(titanik_train_set, "Age", "Survived", 10)
numerical_analysis(titanik_train_set, "Fare", "Survived", 20)

# DATA PREPROCESSING 
columns_to_delete = ["PassengerId", "Ticket", "Cabin", "SibSp", "Parch"]
column_remove(titanik_train_set, columns_to_delete)    
column_remove(titanik_test_set, columns_to_delete[1:])    

one_hot_encoding(titanik_train_set, "Sex")
one_hot_encoding(titanik_test_set, "Sex")

curent_values_sex = ["male", "female"]
new_values_sex = [0, 1]
label_encoding(titanik_train_set, "Sex", curent_values_sex, new_values_sex)
label_encoding(titanik_test_set, "Sex", curent_values_sex, new_values_sex)

# Filling "Embarked" column in train data set with most comon value "S"
for row in titanik_train_set[1:]:
    idx = titanik_train_set[0].index("Embarked")
    if row[idx] == "":
         row[idx] = "S"
 
one_hot_encoding(titanik_train_set, "Embarked")
one_hot_encoding(titanik_test_set, "Embarked")

# Filling "Age" column with median values 
all_median_train = [[1,2,3], [1,2,3]]
all_median_test = [[1,2,3], [1,2,3]]

for x in range(2):
    for y in range(3):
        all_median_train[x][y] = column_median_2(titanik_train_set, "Age", "Sex", "Pclass", x, y+1)
        all_median_test[x][y] = column_median_2(titanik_test_set, "Age", "Sex", "Pclass", x, y+1)

age_col_fill(titanik_train_set, all_median_train)
age_col_fill(titanik_test_set, all_median_test)

data_norm(titanik_train_set, "Age")
data_norm(titanik_test_set, "Age")

# Filling one missing column value "Fare" in test data set
for row in titanik_test_set[1:]:
    idx = titanik_test_set[0].index("Fare")
    if row[idx] == "":
        row[idx] = 35.62781

data_norm(titanik_train_set, "Fare")
data_norm(titanik_test_set, "Fare")

columns_to_delete = ["Sex", "Embarked"]
column_remove(titanik_train_set, columns_to_delete)    
column_remove(titanik_test_set, columns_to_delete[1:])    

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

one_hot_encoding(titanik_train_set, "Name")
one_hot_encoding(titanik_test_set, "Name")

column_remove(titanik_train_set, ["Name"])
column_remove(titanik_test_set, ["Name"])

# Moving column "Survived" to the end
for row in titanik_train_set:
    row.append(row.pop(0))

# Moving column "PassengerId" to the end
for row in titanik_test_set:
    row.append(row.pop(0))

# KNN MODEL TRAINING
test_data_set = []
train_data_set = []
n = int((len(titanik_train_set) - 1) * 0.3)
idx_test = random.sample(range(1, len(titanik_train_set)-1), n)
for idx in range(1, len(titanik_train_set)-1):
    if idx in idx_test:
        test_data_set.append(titanik_train_set[idx])
    else:
        train_data_set.append(titanik_train_set[idx])

for K in range(1, 31):
    all_predictions = []
    k_accuracy = []
    for x in range(len(test_data_set)):
        neighbors = k_neighbors(test_data_set[x][:-1], train_data_set, K)
        prediction = predict(neighbors)
        all_predictions.append(prediction)
    accuracy = accuracy_score(test_data_set, all_predictions)
    k_accuracy.append(accuracy)
    print(f"K = {K:<8} accuracy: {accuracy} % ")
print(sum(k_accuracy)/len(k_accuracy))
print()

k_accuracy = []
for K in range(1, 31):
    accuracy = round(k_fold_cross_val(titanik_train_set, K, 10), 2)
    k_accuracy.append(accuracy)
    print(f"K = {K:<8} Mean accuracy: {accuracy} % ")
print(sum(k_accuracy)/len(k_accuracy))
print()

# PREDICTIONS FOR TEST DATA SET
K = 8
test_set = titanik_test_set[1:]
train_set = titanik_train_set[1:]
all_predictions = []
for x in range(len(test_set)):
    neighbors = k_neighbors(test_set[x][:-1], train_set, K)
    prediction = predict(neighbors)
    all_predictions.append(prediction)
    print(f"PassengerId: {test_set[x][-1]}     Predicted: {prediction}")
