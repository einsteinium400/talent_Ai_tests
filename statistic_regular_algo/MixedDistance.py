import ast
import math
import re


def extract_year(input_str):
    # Regular expression pattern to match YYYY-MM
    pattern = re.compile(r'^(\d{4})-(\d{2})$')

    # Try to match the pattern
    match = pattern.match(input_str)

    if match:
        # Extract the matched year and ignore the month
        year = match.group(1)
        return year


# Test the function with different inputs
inputs = ["1994-09", "2021", ""]
for input_str in inputs:
    result = extract_year(input_str)
    print(f"Input: {input_str}, Year: {result}")


def MixedDistance(u, v, type_values, parameters):
    distance = 0
    results = []
    for i in range(len(u)):
        # if type is categorical
        if type_values[i] == "categoric":
            if v[i] != u[i] and v[i] != "" and u[i] != "":
                distance += 1
            # results.append(1)
        # if type is numeric
        if type_values[i] == "numeric":
            val = (float(u[i]) - float(v[i])) ** 2 if u[i] != '' and v[i] != '' else 0
            if not math.isnan(val):
                distance += val
            # results.append(u[i] - v[i])

            # distance += abs(u[i] - v[i])
        if type_values[i] == "list":

            u_list = ast.literal_eval(u[i])
            v_list = ast.literal_eval(v[i])

            #  adapt according to average list length

            if (len(u_list) < parameters["avg_list_len"][i]):
                u_list.extend(["missing_val"] * (parameters["avg_list_len"][i] - len(u_list)))

            if len(u_list) > parameters["avg_list_len"][i]:
                u_list = u_list[:parameters["avg_list_len"][i]]

            if len(v_list) < parameters["avg_list_len"][i]:
                v_list.extend(["missing_val"] * (parameters["avg_list_len"][i] - len(v_list)))

            if len(v_list) > parameters["avg_list_len"][i]:
                v_list = v_list[:parameters["avg_list_len"][i]]

            for j in range(len(v_list)):
                if v_list[j] == u_list[j] and u_list[j] != "missing_val":
                    distance += 1 / len(v_list)
    return distance, results
