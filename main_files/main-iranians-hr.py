import sys

from avivit_res_talentai.general_algos.Preprocess import *
import csv
import numpy as np
from avivit_res_talentai.statistic_regular_algo.Statistic_dot_product import Statistic_dot_product
from avivit_res_talentai.statistic_regular_algo.Statistic_list_frequency import Statistic_list_frequency
from avivit_res_talentai.statistic_regular_algo.Statistic_intersection import Statistic_intersection
from avivit_res_talentai.statistic_regular_algo.MixedDistance import MixedDistance

# Save the reference to the original sys.stdout
original_stdout = sys.stdout

# Specify the file path where you want to redirect the prints
output_file_path = '../logger_of_dana.txt'

# Open the file in write mode, this will create the file if it doesn't exist
output_file = open(output_file_path, 'w')

# Redirect sys.stdout to the file
# sys.stdout = output_file


import numpy as np
import re

def convert_month_year_to_year(array_list):
    # Regular expression pattern to match YYYY-MM
    pattern = re.compile(r'^(\d{4})-(\d{2})$')

    for arr in array_list:
        # Check if the array has at least 35 elements (to access index 33 and 34)
        if len(arr) > 34:
            # Process index 33
            value_33 = arr[33]
            if isinstance(value_33, str) and value_33:
                match_33 = pattern.match(value_33)
                if match_33:
                    arr[33] = int(match_33.group(1))
                else:
                    arr[33] = ""

            # Process index 34
            value_34 = arr[34]
            if isinstance(value_34, str) and value_34:
                match_34 = pattern.match(value_34)
                if match_34:
                    arr[34] = int(match_34.group(1))
                else:
                    arr[34] = ""

    return array_list

# Example usage:
# Create a list of NumPy arrays



types_list = ['categoric', 'categoric', 'categoric', 'categoric', 'numeric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'list', 'categoric', 'categoric', 'categoric', 'list', 'list',
              'categoric', 'categoric', 'categoric', 'numeric', 'categoric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list', 'categoric',
              'categoric', 'categoric', 'categoric', 'numeric', 'list', 'list', 'list']

i = 0

with open('../datasets/employes_flat_version.csv', 'r', encoding='utf-8') as csvfile:  # employes.csv
    # Create a CSV reader object
    csv_data = []
    csvreader = csv.reader(csvfile)
    i += 1
    # Iterate through each row in the CSV file

    for row in csvreader:
        # print(i)
        # print(row)
        # Append each row as a list to the csv_data list
        csv_data.append(row)

vectors = [np.array(f, dtype=object) for f in csv_data]
# model.calc_min_max_dist(vectors)
#vectors=convert_month_year_to_year(vectors)


hp, k = preProcess(vectors, types_list, Statistic_dot_product, 9, 9)

print("making model of dot")
model = KMeansClusterer(num_means=k,
                        distance=Statistic_dot_product,
                        repeats=16,
                        type_of_fields=types_list,
                        hyper_params=hp)
# print("before")
# model.calc_min_max_dist(vectors)

model.cluster_vectorspace(vectors)

print("done making model")

model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()
print("min distance is", model.min_dist)
print("max distance is", model.max_dist)


##################################################################3

print("######################3making model of dot product")

hp, k = preProcess(vectors, types_list, Statistic_dot_product, 9, 9)
model = KMeansClusterer(num_means=k,
                        distance=Statistic_dot_product,
                        repeats=20,
                        type_of_fields=types_list,
                        hyper_params=hp)
# print("before")
# model.calc_min_max_dist(vectors)


model.cluster_vectorspace(vectors)

print("done making model")

model.calc_min_max_dist(vectors)
print("min distance is", model.min_dist)
print("max distance is", model.max_dist)
model.get_wcss()
model.calc_distance_between_clusters()

exit()
##########################################################3

print("######################3making model of intersection")

hp, k = preProcess(vectors, types_list, Statistic_intersection, 9, 9)
model = KMeansClusterer(num_means=k,
                        distance=Statistic_intersection,
                        repeats=4,
                        type_of_fields=types_list,
                        hyper_params=hp)

model.cluster_vectorspace(vectors)

print("done making model")

#model.calc_min_max_dist(vectors)
model.calc_distance_between_clusters()

model.get_wcss()
print("min distance is", model.min_dist)
print("max distance is", model.max_dist)

# Close the file to ensure changes are saved
output_file.close()

# print("done making model")
# model.cluster_vectorspace(vectors)
# print("finish cluster")
# model.createClusterJson()
# print(model._model_json_info)
# model.calc_min_max_dist(vectors)
#
# print("wcss is", model.get_wcss())
