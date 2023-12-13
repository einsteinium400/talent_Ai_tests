import ast


def MixedDistance(u, v, type_values, parameters):
    distance = 0
    results = []
    for i in range(len(u)):
        # if type is categorical
        if type_values[i] == "categoric":
            if v[i] != u[i]:
                distance += 1
               # results.append(1)
        # if type is numeric
        if type_values[i] == "numeric":
            distance += (float(u[i]) - float(v[i])) ** 2 if u[i]!='' and v[i]!='' else 0
            #results.append(u[i] - v[i])

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
                if v_list[j] == u_list[j] and u_list[j]!= "missing_val":
                    distance += 1/len(v_list)
    return distance, results
