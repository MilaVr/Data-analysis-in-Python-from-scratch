
def categorical_analysis(input_data, column1, column2, value2=1):
    """Gives correlation between input feature (column1) and the output feature (column2, value2).  
    """
    idx1 = input_data[0].index(column1)
    idx2 = input_data[0].index(column2)
   
    feature_count = {}
    for row in input_data[1:]:
        if row[idx1] != "":
            if row[idx1] not in feature_count:
                if row[idx2] == value2:
                    feature_count[row[idx1]] = [1, 1]
                else:
                    feature_count[row[idx1]] = [0, 1]
            else:
                if row[idx2] == value2:
                    feature_count[row[idx1]][0] += 1
                    feature_count[row[idx1]][1] += 1
                else:
                    feature_count[row[idx1]][1] += 1
    
    feature_count = sorted(feature_count.items())

    print(f"{column1:<8} {column2:<8}")
    for f_c in feature_count:
       print(f"{f_c[0]:<8} {f_c[1][0]:<3.0f} / {f_c[1][1]:<8.0f} {round(f_c[1][0] / f_c[1][1], 6):<1.6f}")
    print()
    
def numerical_analysis(input_data, column1, column2, bins, value2=1):
    """Splits the input feature values (column1) into range bands (bins) and gives a correlation 
       with the output feature (column2, value2).
    """
    idx1 = input_data[0].index(column1)
    idx2 = input_data[0].index(column2)

    col_values = []
    for row in input_data[1:]:
        if row[idx1] != "" and row[idx1] not in col_values:
            col_values.append(row[idx1])
    col_values = sorted(col_values)
    step = len(col_values) // bins

    data_bins = []
    for i in range(0, len(col_values), step):
        if len(data_bins) < bins:
            data_bins.append(col_values[i:i+step])
        else:
            data_bins[-1] += col_values[i:i+step]
    
    bins_range = [(i[0], i[-1]) for i in data_bins]
 
    feature_count = {}
    for row in input_data[1:]:
        for data_bin, bin_range in zip(data_bins, bins_range): 
            if row[idx1] != "" and row[idx1] in data_bin:
                if bin_range not in feature_count:
                    if row[idx2] == value2:
                        feature_count[bin_range] = [1, 1]
                    else:
                        feature_count[bin_range] = [0, 1]
                else:
                    if row[idx2] == value2:
                        feature_count[bin_range][0] += 1
                        feature_count[bin_range][1] += 1
                    else:
                        feature_count[bin_range][1] += 1
    
    feature_count = sorted(feature_count.items())
 
    print(f"{column1} bands             {column2:<8}")
    for f_c in feature_count:
       print(f"{f_c[0][0]:<6.2f} - {f_c[0][1]:<11.2f} {f_c[1][0]:<3.0f} / {f_c[1][1]:<8.0f} {round(f_c[1][0] / f_c[1][1], 6):<1.6f}")
    print()
   


    