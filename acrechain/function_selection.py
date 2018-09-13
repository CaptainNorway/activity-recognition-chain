import pickle
import os
import numpy as np
import collections

from acrechain import definitions


def getFunctions(sensor, feature_importances, print_stats = False):

    print("")
    print("Finding functions to calculate for ", sensor, " sensor")
    print("Feature importances inputed: ", feature_importances)
    print("Number of features inputed: ", len(feature_importances))



    function_feature_mapping_path = os.path.join(definitions.indexes_folder, "function_indexes_mapping.pickle")


    with open(function_feature_mapping_path, 'rb') as f:
        function_feature_indexes_mapping = pickle.load(f)
        #print(type(function_feature_indexes_mapping))



    #print("function_feature_indexes_mapping", function_feature_indexes_mapping)
    #print(feature_indexes_mapping)
    #print(len(feature_indexes_mapping))



    print("Function_feature_mapping: ", function_feature_indexes_mapping)

    feature_indexes = []

    if sensor == "back":
        lower_back = []
        for feature in feature_importances:
            if feature[0] <= 68:
                lower_back.append(feature[0])
        print("lower_back", lower_back)
        feature_indexes = lower_back

    elif sensor == "thigh":
        thigh = []
        for feature in feature_importances:
            if feature[0] > 68:
                thigh.append(feature[0] % 69)
        print("thigh", thigh)
        feature_indexes = thigh


    print("Feature indexes", feature_indexes)

    print("len feature_indexes: ", len(feature_indexes))
    if(len(feature_indexes) == 0):
        return []

    for key in function_feature_indexes_mapping.keys():
        index_range = function_feature_indexes_mapping[key]
        lower_bound = index_range[0]
        upper_bound = index_range[1]
        indexes = []
        for i in range(lower_bound, upper_bound + 1, 1):
            indexes.append(i)
        function_feature_indexes_mapping[key] = indexes
    #print(function_feature_indexes_mapping)

    print("Function_feature_mapping 2: ", function_feature_indexes_mapping)

    feature_indexes_function_mapping = {}
    for k, v in function_feature_indexes_mapping.items():
        new_k = tuple(v)
        feature_indexes_function_mapping[new_k] = k

    print("feature_indexes_function_mapping: ", feature_indexes_function_mapping)

    #print("Feature indexes function mapping", feature_indexes_function_mapping)

    #Functions covering the most important features

    functions = collections.OrderedDict()

    keys = feature_indexes_function_mapping.keys()
    for feature_index in feature_indexes:
        for index_set in keys:
           #print(index_set)
           if feature_index in index_set:
               #print("index_set[0]", index_set[0])
               #print("index_set[-1]", index_set[-1])
               if (feature_indexes_function_mapping[index_set] not in functions):
                   if(index_set[0] == 0):
                       functions[feature_indexes_function_mapping[index_set]] = [feature_index]
                   else:
                       functions[feature_indexes_function_mapping[index_set]] = [(feature_index + index_set[0]) % (index_set[0])]

               else:
                   if(index_set[0] == 0):
                       functions[feature_indexes_function_mapping[index_set]].append((feature_index))
                   else:
                       functions[feature_indexes_function_mapping[index_set]].append((feature_index + index_set[0])%(index_set[0]))


    #Transform into ordered list
    functions_list = []


    #print(functions)
    #for function, function_index in functions.keys():
    #    print(function_index)

    # Number of features extracted, should match the number of features inputed, when added to
    # the corresponding number for the other sensor

    number_of_functions_to_calculate = len(functions.keys())
    number_of_features_extracted = 0
    for function in functions.keys():
        number_of_features_extracted += len(functions[function])

    if(print_stats):
        print("Number of functions to calculate: ", number_of_functions_to_calculate)
        print("Number of features extracted: ", number_of_features_extracted)
        print("Functions to be calculated: ", functions)
        print("")
        list = []
        #for func


    return functions

#print(getFunctions("back"))
#print(getFunctions("thigh"))

