import csv

def data_load(file_name):
    with open(file_name) as csv_file:
        input_data = csv.reader(csv_file)
        return list(input_data)

def data_shape(input_data):
    """Prints the list of feature column names and shape of input data (nuber of inputs / number of features).
    """
    print("Columns, values:")
    print(input_data[0]) 
    print()
    print("Data shape:")
    print(len(input_data[1:]), "/", len(input_data[0]))
    print()

def data_head(input_data, num_rows = 5):
    """Prints the first n rows of the input data set.

    Parameters:

    num_row (int): Default value 5, number of rows to be printed
    """
    first_row = [str(data) for data in input_data[0]]
    print(" | ".join(first_row))
    print()
    for row in input_data[1:num_rows+1]:
        row_to_string = [str(data) for data in row]
        print(" | ".join(row_to_string))
    print()

def data_tail(input_data, num_rows = 5):
    """Prints the last n rows of the input data set.

    Parameters:
    
    num_row (int): Default value 5, number of rows to be printed
    """
    first_row = [str(data) for data in input_data[0]]
    print(" | ".join(first_row))
    print()
    for row in input_data[-num_rows:]:
        row_to_string = [str(data) for data in row]
        print(" | ".join(row_to_string))
    print()

def data_info(input_data):
    """Prints the statistical information about the input data set.
    """
    data_dict = {}
    fieldnames = input_data[0]
    for row in input_data[1:]:    
        for col_idx in range(len(fieldnames)):
            if row[col_idx] != "":
                if fieldnames[col_idx] in data_dict:
                    data_dict[fieldnames[col_idx]].append(row[col_idx])
                else:
                    data_dict[fieldnames[col_idx]] = [row[col_idx]]
    
    all_data_stat = []
    num_data_stat = []
    str_data_stat = []
    for column in data_dict:
        missing_data = len(input_data[1:]) - len(data_dict[column])
        full_data = len(input_data[1:]) - missing_data
        data_percent = full_data/len(input_data[1:]) * 100

        data_type = [type(x) for x in data_dict[column]]
        data_type = set(data_type)

        all_data_stat.append((column, full_data, data_percent, data_type))
    
        if str not in data_type:
            mean = sum(data_dict[column])/len(data_dict[column])
            max_value = max(data_dict[column])
            min_value = min(data_dict[column])   

            num_data_stat.append((column, full_data, mean, max_value, min_value))
        else:
            count_data = {}
            uniqe = 0
            for data in data_dict[column]:
                if data not in count_data:
                    count_data[data] = 1
                else:
                    count_data[data] += 1
            uniqe = len(count_data)
            count_data = sorted(count_data.items(), key = lambda key_value: key_value[1])
            top = count_data[-1][0]
            freq = count_data[-1][1]

            str_data_stat.append((column, full_data, uniqe, top, freq))

    print(f"Range Index: {len(input_data[1:])} entries")
    print(f"Data columns (total {len(fieldnames)} columns):")
    for data in all_data_stat:
        print(f"{data[0]:<15} {data[1]:<6} non-null   {data[2]:6.2f} %   {data[3]}")
    print()

    head_num = ["", "count", "mean", "max", "min"]
    print(f"{head_num[0]:<15} {head_num[1]:<10} {head_num[2]:<15} {head_num[3]:<13} {head_num[4]:<8}")
    for data in num_data_stat:
        print(f"{data[0]:<15} {data[1]:<10} {data[2]:<15f} {data[3]:<13} {data[4]:<8}")
    print()

    head_str = ["", "count", "uniqe", "top", "freq"]
    print(f"{head_str[0]:<15} {head_str[1]:<10} {head_str[2]:<10} {head_str[3]:<23} {head_str[4]:<8}")
    for data in str_data_stat:
        print(f"{data[0]:<15} {data[1]:<10} {data[2]:<10} {data[3]:<23} {data[4]:<8}")
    print()



    


