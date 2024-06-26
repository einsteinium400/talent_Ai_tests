import sys

from avivit_res_talentai.general_algos.Preprocess_for_hr import *
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

# Redirect sys.stdout to the file, comment this out if no need for print
# sys.stdout = output_file


import numpy as np
import re


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
        # Append each row as a list to the csv_data list
        csv_data.append(row)

vectors = [np.array(f, dtype=object) for f in csv_data]
print("hi")

hp, k = preProcess(vectors, types_list, Statistic_intersection, 9, 9)
# in order to run this you need to comment out the part that refers to one hot vector in kmeansclusterer

print("making model of dot")
model = KMeansClusterer(num_means=k,
                        distance=Statistic_intersection,
                        repeats=8,
                        type_of_fields=types_list,
                        hyper_params=hp)

# print(hp["list_freq_dict"])

model.cluster_vectorspace(vectors)

print("done making model")

# model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()
exit()

##################################################################

print("######################3making model of Statistic_intersection ")
hp, k = preProcess(vectors, types_list, Statistic_intersection, 9, 9)
model = KMeansClusterer(num_means=k,
                        distance=Statistic_intersection,
                        repeats=5,
                        type_of_fields=types_list,
                        hyper_params=hp)
print(hp["frequencies"])
model.cluster_vectorspace(vectors)

print("done making model")

model.get_wcss()
model.calc_distance_between_clusters()
exit()
##########################################################3

