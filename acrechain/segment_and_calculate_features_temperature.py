from collections import Counter

import numpy as np
import scipy.stats
import pickle
import acrechain.definitions as definitions
import acrechain.function_selection
import os


# Functions that were kept from the original segment_and_calculate_features.py

def most_frequent_value(array):
    if len(array.shape) > 1:
        most_common = []
        for column in array.T:
            counts = Counter(column)
            top = counts.most_common(1)[0][0]
            most_common.append(top)

        return np.array(most_common)

    counts = Counter(array)
    top = counts.most_common(1)[0][0]
    return np.array([top])


def median(array):
    return np.median(array)


def skewness(array):
    return scipy.stats.skew(array, axis=0)


def maxmin_range(array):
    return np.max(array, axis=0) - np.min(array, axis=0)

def mean(array):
    return np.mean(array)


def standard_deviation(array):
    return np.std(array)

# Temperature functions

def max(array):

    return np.max(array)

def min(array):
    return np.min(array)

# The variation in temperature from the warmest to the coldest
def max_min_delta(array):
    max_temp = max(array)
    min_temp = min(array)
    return(max_temp-min_temp)


def first_last_delta(array):
    temperature_first_sample_in_window = array[0]
    temperature_last_sample_in_window = array[-1]
    return(temperature_last_sample_in_window-temperature_first_sample_in_window)

def segment_acceleration_and_calculate_features(sensor_data, samples_pr_window = 50, overlap=0.0,
                                                remove_sign_after_calculation=True):
    #print("len sensor data: ", sensor_data.shape)
    functions = [
        max,
        min,
        max_min_delta,
        first_last_delta,
    ]

    #window_samples = int(sampling_rate * window_length)
    window_samples = samples_pr_window

    #print("Windows samples ", window_samples)
    step_size = int(round(window_samples * (1.0 - overlap)))


    all_features = []


    for window_start in np.arange(0, sensor_data.shape[0], step_size):
        #print("Window start: ", window_start, "Sensor_data.shape[0]: ", sensor_data.shape[0], step_size)
        window_start = int(round(window_start))
        window_end = window_start + int(round(window_samples))
        if window_end > sensor_data.shape[0]:
            break
        window = sensor_data[window_start:window_end]

        #print("Window", window)
        extracted_features = []
        # print("Windows start: ")
        index_of_function = 0
        n = 0
        for func in functions:
            value = func(window)
            extracted_features.append(value)



        all_features.append(np.hstack(extracted_features))


    one_large_array = np.vstack(all_features)

    if remove_sign_after_calculation:
        np.absolute(one_large_array, one_large_array)

    print(one_large_array.shape)

    return one_large_array


def segment_labels(label_data, sampling_rate=1, window_length=120, overlap=0.0, samples_pr_window = 50):
    #window_samples = int(sampling_rate * window_length)
    window_samples = samples_pr_window
    step_size = int(round(window_samples * (1.0 - overlap)))

    labels = []

    for window_start in np.arange(0, label_data.shape[0], step_size):
        window_start = int(round(window_start))
        window_end = window_start + int(round(window_samples))
        if window_end > label_data.shape[0]:
            break
        window = label_data[window_start:window_end]
        #print(window)
        top = find_majority_activity(window)
        labels.append(top)

    return np.array(labels)


def find_majority_activity(window):
    sensor_labels_list = window.tolist()
    labels_without_list = []
    for sensor_label in sensor_labels_list:
        labels_without_list.append(sensor_label[0])
    counts = Counter(labels_without_list)
    top = counts.most_common(1)[0][0]
    return top
