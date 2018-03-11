import pickle
import os
import numpy as np

from acrechain import definitions

function_future_mapping_path = os.path.join(definitions.indexes_folder, "function_indexes_mapping.pickle")


with open(function_future_mapping_path, 'rb') as f:
    function_feature_indexes_mapping = pickle.load(f)



important_features_indexes = os.path.join(definitions.indexes_folder, "indexes_0.4_percent_healthy_3.0s_model_1.0hz_reduced.pickle")
with open(important_features_indexes, 'rb') as f:
    feature_indexes_mapping = pickle.load(f)



for key in function_feature_indexes_mapping.keys():
    index_range = function_feature_indexes_mapping[key]
    lower_bound = index_range[0]
    upper_bound = index_range[1]
    indexes = []
    for i in range(lower_bound, upper_bound + 1, 1):
        indexes.append(i)
    function_feature_indexes_mapping[key] = indexes



feature_indexes_function_mapping = {}
for k, v in function_feature_indexes_mapping.items():
    new_k = tuple(v)
    feature_indexes_function_mapping[new_k] = k

#print(feature_indexes_function_mapping)

#Functions covering the most important features
functions = []

for index in feature_indexes_mapping:
   keys = feature_indexes_function_mapping.keys()
   for index_set in keys:
       if int(index) in index_set:
           if (feature_indexes_function_mapping[index_set] not in functions):
               functions.append(feature_indexes_function_mapping[index_set])



test_array_1 = np.array([0,1,2,3,4,5,6])
indexes = [1,2,5]
test_array_1 = test_array_1[indexes]
print(test_array_1)