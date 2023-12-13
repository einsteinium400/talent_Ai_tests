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


types_list = ['categoric', 'categoric', 'categoric', 'categoric', 'numeric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'list', 'categoric', 'categoric', 'categoric', 'list', 'list',
              'categoric', 'categoric', 'categoric', 'numeric', 'categoric',
              'categoric', 'categoric', 'categoric', 'categoric', 'categoric',
              'categoric', 'categoric', 'categoric', 'list', 'categoric',
              'categoric', 'categoric', 'categoric', 'numeric', 'list', 'list', 'list']

i = 0

with open('../datasets/employes_flat_version_slim.csv', 'r', encoding='utf-8') as csvfile:  # employes.csv
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

hp, k = preProcess(vectors, types_list, MixedDistance, 9, 9)

print("making model of Statistic_list_frequency")
model = KMeansClusterer(num_means=k,
                        distance=MixedDistance,
                        repeats=1,
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

exit()
##################################################################3

print("######################3making model of Statistic_dot_product")

hp, k = preProcess(vectors, types_list, Statistic_dot_product, 9, 9)
model = KMeansClusterer(num_means=k,
                        distance=Statistic_dot_product,
                        repeats=4,
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

model.calc_min_max_dist(vectors)
model.get_wcss()
model.calc_distance_between_clusters()
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
