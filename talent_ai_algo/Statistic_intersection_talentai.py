import ast

import numpy as np
import math


def Statistic_intersection(u, v, type_values, parameters):
    # print("started statistic")

    # print(parameters)
    distance = 0
    results = []

    def f_freq(z, theta1, betha, theta2, gamma):
        if z <= theta1:
            return 1
        if theta1 < z <= theta2:
            return 1 - betha * (z - theta1)
        if z > theta2:
            return 1 - betha * (theta2 - theta1) - gamma * (z - theta2)

    def calculate_union(one_hot_vector1, one_hot_vector2):
        if len(one_hot_vector1) != len(one_hot_vector2):
            raise ValueError("Input vectors must have the same length")

        union_result = [0] * len(one_hot_vector1)
        for i in range(len(one_hot_vector1)):
            union_result[i] = one_hot_vector1[i] or one_hot_vector2[i]

        return union_result

    betha = parameters["betha"]
    theta1 = parameters["theta1"]
    theta2 = parameters["theta2"]
    theta = parameters["theta"]
    gamma = parameters["gamma"]
    # print("u is: ", u)
    # print("v is: ", v)



    for i in range(len(v)):

        # catrgorical handle
        try:
            if type_values[i] == "categoric":
                # if attributes are same
                if float(u[i]) == float(v[i]):
                    results.append(0)
                # attributes are not the same - calculate max{f(|vak|), dfr(vi, ui), theta)
                else:
                    specific_domain_size = parameters["domain sizes"][i]
                    f_v_ak = f_freq(specific_domain_size, theta1, betha, theta2, gamma)
                    fr_u = float(u[i])
                    fr_v = float(v[i]) #parameters["frequencies"][str(i)][str((v[i]))] if v[i]!="" else 1
                    m_fk = parameters["minimum_freq_of_each_attribute"][str(i)]
                    d_fr = (abs(fr_u - fr_v) + m_fk) / max(fr_u, fr_v)
                    results.append(abs(max(d_fr, theta, f_v_ak)))
                    distance += pow(max(d_fr, theta, f_v_ak), 2)
        except Exception as e:
            print("error!!!!!", e)
            print("v is", v)
            print("i is", i)
            print("type values is", type_values, len(type_values))


        # numberic handle
        if type_values[i] == "numeric":
            # print()
            # print(f'i:{i}, u[i]:{u[i]}, v[i]:{v[i]}')
            try:
                if u[i] != '' and v[i] != '':
                    results.append(abs(np.float64(u[i]) - np.float64(v[i])))
                    distance += pow(np.float64(u[i]) - np.float64(v[i]), 2)

            except Exception as e:
                print(e)
                print(u[i])
                print(i)
                print(v[i])
                exit()

        if type_values[i] == "list":
            # create one hot vector
            u_list = ast.literal_eval(u[i])
            v_list = ast.literal_eval(v[i])

            ##### intersection
            one_hot_vec_u = [1 if word in u_list else 0 for word in parameters["one_hot_vector_prep"][i]]
            one_hot_vec_v = [1 if word in v_list else 0 for word in parameters["one_hot_vector_prep"][i]]


            # Calculate the intersection using element-wise AND
            intersection = [a & b for a, b in zip(one_hot_vec_u, one_hot_vec_v)]
            union = calculate_union(one_hot_vec_u, one_hot_vec_v)

            distance += 1 if sum(union) == 0 else 1 - sum(intersection) / sum(union)



    distance = math.sqrt(distance)
    # print("ended statistic")

    return distance, results