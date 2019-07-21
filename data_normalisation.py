import math

def mean(values):
    """Returns a mean of a set of data values.
    """
    return sum(values)/len(values)

def std(values, mean_value):
    """Returns a standard deviation of a set of data values.
    """
    N = len(values)
    sum_sqr = 0
    for x in values:
        sum_sqr += (x - mean_value)**2
    return math.sqrt(sum_sqr/(N-1))

def data_norm(input_data, column):
    """Z-score normalization of feature values.
    """
    idx = input_data[0].index(column)
    col_values = [row[idx] for row in input_data[1:]]
    col_mean = mean(col_values)
    col_std = std(col_values, col_mean)
    for row in input_data[1:]:
        row[idx] = round((row[idx] - col_mean) / col_std, 4) 

