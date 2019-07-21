from copy import deepcopy

def data_type_convert(input_data):
    """Convertes the numerical data from input data set to int and float.
    """
    for row in input_data[1:]:
        for col_idx in range(len(row)):
            try:
                row[col_idx] = float(row[col_idx])
                if row[col_idx] == int(row[col_idx]):
                    row[col_idx] = int(row[col_idx])
                else:
                    pass
            except ValueError:
                pass
   
def column_remove(input_data, columns):
    """Removes the entire column from the input data set.

    Parameters:
    
    columns (list): Names of columns to be removed
    """
    for row in input_data[1:]:
        fieldnames = deepcopy(input_data[0])
        for column in columns:
            row.pop(fieldnames.index(column))
            fieldnames.remove(column)

    for fieldname in columns:
        input_data[0].remove(fieldname)

def label_encoding(input_data, column, curent_values, new_values):
    """Converting categorical feature values by assigning each uniqe value an integer value.
    
    Parameters: 

    column (str): Name of column to be converted \n
    curent_values (list): Curent column values \n
    new_values (list): New labels (int) for column values
    """
    idx = input_data[0].index(column)
    for row in input_data[1:]:    
        for c_v, n_w in zip(curent_values, new_values):
            if row[idx] == c_v:
                row[idx] = n_w
            elif row[idx] == "":
                pass

def one_hot_encoding(input_data, column):
    """Label encoded feature is removed and a new binary feature is added for each unique feature (label) value.
    """
    idx = input_data[0].index(column)
    categorical_values = [row[idx] for row in input_data[1:]] 
    categorical_values = set(categorical_values)
    for c_value in categorical_values:
        input_data[0].append(c_value)
    for row in input_data[1:]:
        for _ in range(len(categorical_values)):
            row.append("")

    for row in input_data[1:]:
        for c_value in categorical_values:
            idx1 = input_data[0].index(c_value)
            if row[idx] == c_value:
                row[idx1] = 1
            else:
                row[idx1] = 0

def median(values):
    values = sorted(values)
    if len(values) // 2 != 0:
        return values[len(values) // 2]
    else:
        return round(((values[len(values) // 2] + values[len(values) // 2 - 1]) / 2), 2)

def column_median(input_data, column):
    """Returns a median value for feature column.
    """
    idx = input_data[0].index(column)
    col_values = [row[idx] for row in input_data[1:] if row[idx] != ""]
    return median(col_values)

def column_median_1(input_data, column, column1, value1):
    """Returns a median value for feature column across sets of column1 
        feature combinations (value1). 
    """
    idx = input_data[0].index(column)
    idx1 = input_data[0].index(column1)
    col_values = [row[idx] for row in input_data[1:] if (row[idx] != "" and row[idx1] == value1)]
    return median(col_values)
  
def column_median_2(input_data, column, column1, column2, value1, value2):
    """Returns a median value for feature column across sets of column1 and column2 
        feature combinations (value1, value2). 
    """
    idx = input_data[0].index(column)
    idx1 = input_data[0].index(column1)
    idx2 = input_data[0].index(column2)
    col_values = [row[idx] for row in input_data[1:] if (row[idx] != "" and row[idx1] == value1 and row[idx2] == value2)]
    return median(col_values)


       

        

        

    








   