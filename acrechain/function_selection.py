import pickle
import os
import numpy as np

from acrechain import definitions


def getFunctions(sensor):


    function_feature_mapping_path = os.path.join(definitions.indexes_folder, "function_indexes_mapping.pickle")


    with open(function_feature_mapping_path, 'rb') as f:
        function_feature_indexes_mapping = pickle.load(f)
        #print(type(function_feature_indexes_mapping))



    important_features_indexes = os.path.join(definitions.indexes_folder, "indexes_0.4_percent_healthy_3.0s_model_1.0hz_reduced.pickle")
    with open(important_features_indexes, 'rb') as f:
        feature_indexes_mapping = pickle.load(f)



    #print("function_feature_indexes_mapping", function_feature_indexes_mapping)
    #print(feature_indexes_mapping)
    #print(len(feature_indexes_mapping))

    n_futures = len(feature_indexes_mapping)




    if sensor == "back":
        lower_back = []
        for i in feature_indexes_mapping:
            if i <= 68:
                lower_back.append(i)
        #print("lower_back", lower_back)
        feature_indexes_mapping = lower_back

    elif sensor == "thigh":
        thigh = []
        for i in feature_indexes_mapping:
            if i > 68:
                thigh.append(i % 69)
        #print("thigh", thigh)
        feature_indexes_mapping = thigh



    #print("Feature to index mapping", feature_indexes_mapping)

    for key in function_feature_indexes_mapping.keys():
        index_range = function_feature_indexes_mapping[key]
        lower_bound = index_range[0]
        upper_bound = index_range[1]
        indexes = []
        for i in range(lower_bound, upper_bound + 1, 1):
            indexes.append(i)
        function_feature_indexes_mapping[key] = indexes
    #print(function_feature_indexes_mapping)


    feature_indexes_function_mapping = {}
    for k, v in function_feature_indexes_mapping.items():
        new_k = tuple(v)
        feature_indexes_function_mapping[new_k] = k

    #print("Feature indexes function mapping", feature_indexes_function_mapping)

    #Functions covering the most important features
    functions = {}

    for index in feature_indexes_mapping:
       keys = feature_indexes_function_mapping.keys()
       for index_set in keys:
           #print(index_set)
           if int(index) in index_set:
               #print("index_set[0]", index_set[0])
               #print("index_set[-1]", index_set[-1])
               if (feature_indexes_function_mapping[index_set] not in functions):
                   if(index_set[0] == 0):
                       functions[feature_indexes_function_mapping[index_set]] = [index]
                   else:
                       functions[feature_indexes_function_mapping[index_set]] = [(index + index_set[0]) % (index_set[0])]

               else:
                   if(index_set[0] == 0):
                       functions[feature_indexes_function_mapping[index_set]].append((index))
                   else:
                       functions[feature_indexes_function_mapping[index_set]].append((index + index_set[0])%(index_set[0]))
    #print(functions)
    #for function, function_index in functions.keys():
    #    print(function_index)
    return functions

#print(getFunctions("back"))
#print(getFunctions("thigh"))

