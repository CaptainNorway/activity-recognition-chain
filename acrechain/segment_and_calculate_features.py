from collections import Counter

import numpy as np
import scipy.stats
import pickle
import acrechain.definitions as definitions
import acrechain.function_selection
import os



def generate_all_integer_combinations(stop_integer):
    combinations_of_certain_length = dict()
    combinations_of_certain_length[1] = {(j,) for j in range(stop_integer)}

    for i in range(2, stop_integer + 1):
        combinations_of_certain_length[i] = set()
        for t in combinations_of_certain_length[i - 1]:
            largest_element = t[-1]
            for j in range(largest_element + 1, stop_integer):
                l = list(t)
                l.append(j)
                new_combination = tuple(l)
                combinations_of_certain_length[i].add(new_combination)

    values = [sorted(list(combinations_of_certain_length[j])) for j in range(2, stop_integer + 1)]

    reduced_list = []
    for value in values:
        reduced_list += value

    return reduced_list


def peak_acceleration(array):
    return max(np.linalg.norm(array, axis=1))


def max_and_mins(array):  # Something should be done about this, so it's the largest deviation or something
    return np.hstack(array.max())


def means_and_std_factory(absolute_values=False):
    def means_and_std(a):
        if absolute_values:
            a = abs(a)
        #print("mean_and_std_factory: ", np.hstack((np.mean(a, axis=0), np.std(a, axis=0))).shape)
        return np.hstack((np.mean(a, axis=0), np.std(a, axis=0)))
    return means_and_std


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
    print("!!!!!!!!!!!!!!!")
    print(np.array([top]))
    return np.array([top])


def column_product_factory(column_indices):
    def columns_product(array):
        transposed = np.transpose(array)[[column_indices]]  # Transpose for simpler logic

        product = np.ones(transposed.shape[1])
        for row in transposed:
            product *= row

        product = np.transpose(product)

        return np.array([product.mean(), product.std()])

    return columns_product


def crossing_rate_factory(type='zero'):
    def crossing_rate(array):
        if type is 'zero':
            if len(array.shape) > 1:
                means = np.zeros(array.shape[1])
            else:
                means = 0
        elif type is 'mean':
            means = np.average(array, axis=0)

        crossings = []
        for i in range(1, array.shape[0]):
            crossings.append(np.abs(np.sign(array[i] - means) - np.sign(array[i - 1] - means)))

        final_crossing_rate = np.sum(crossings, axis=0) / (array.shape[0] - 1)

        return final_crossing_rate

    return crossing_rate


def root_square_mean(array):
    squares = np.square(array)
    means = np.average(squares, axis=0)
    square_roots = np.sqrt(means)

    return square_roots


def energy(array):
    means = np.average(array, axis=0)
    calibrated_values = array - means
    squared = np.square(calibrated_values)
    axis_by_axis_energy = np.sqrt(np.average(squared, axis=0))

    average_energy = np.average(axis_by_axis_energy)

    return average_energy


def median(array):
    return np.median(array, axis=0)


def pearson_correlation(array):
    coefs = np.corrcoef(array, rowvar=0)

    results = []

    for i in range(array.shape[1]):
        for j in range(i + 1, array.shape[1]):
            j_ = coefs[i, j]

            if np.isnan(j_):
                j_ = 0

            results.append(j_)

    return np.array(results)


def skewness(array):
    return scipy.stats.skew(array, axis=0)


def maxmin_range(array):
    return np.max(array, axis=0) - np.min(array, axis=0)


def interquartile_range(array):
    q75, q25 = np.percentile(array, (75, 25), axis=0)
    return q75 - q25


def magnitude_avg_and_std(array):
    magnitude = np.linalg.norm(array, axis=1)
    #print("Fdddddd", np.average(magnitude))
    #print("DDDDDDDD", np.average(magnitude
    return np.average(magnitude), np.std(magnitude)


def frequency_domain_factory(sample_rate):
    def frequency_domain_features(array):
        fourier_transform = np.fft.rfft(array, axis=0)
        frequency_powers = np.abs(fourier_transform)

        means = np.mean(frequency_powers, axis=0)
        stds = np.std(frequency_powers, axis=0)

        max_power = np.max(frequency_powers, axis=0)
        median_power = np.median(frequency_powers, axis=0)

        sample_spacing = 1 / sample_rate
        frequencies = np.fft.rfftfreq(array.shape[0], sample_spacing)

        spectral_centroid = np.sum(frequency_powers * frequencies[:, np.newaxis], axis=0) / np.sum(frequency_powers,
                                                                                                   axis=0)

        for i in range(len(spectral_centroid)):
            if not np.isfinite(spectral_centroid[i]):
                spectral_centroid[i] = 0

        dominant_frequencies_indices = np.argmax(frequency_powers, axis=0)
        dominant_frequencies = np.array([frequencies[i] for i in dominant_frequencies_indices])

        frequency_power_squares = np.square(frequency_powers, np.zeros_like(frequency_powers)) / frequency_powers.shape[
            0]
        p_i = frequency_power_squares / np.sum(frequency_power_squares, axis=0)

        entropy = scipy.stats.entropy(p_i)

        for i in range(len(entropy)):
            if not np.isfinite(entropy[i]):
                entropy[i] = 0

        return np.hstack([means, stds, max_power, median_power, spectral_centroid, dominant_frequencies, entropy])

    return frequency_domain_features


def segment_acceleration_and_calculate_features(sensor_data, function_dict, sampling_rate, window_length=3.0, overlap=0.0,
                                                remove_sign_after_calculation=True, print_stats = False):
    #print("len sensor data: ", sensor_data.shape)
    functions = [
        means_and_std_factory(False),
        means_and_std_factory(True),
        peak_acceleration,
        crossing_rate_factory('zero'),
        crossing_rate_factory('mean'),
        root_square_mean,
        energy,
        median,
        skewness,
        pearson_correlation,
        maxmin_range,
        interquartile_range,
        magnitude_avg_and_std,
        frequency_domain_factory(sampling_rate),
    ]



    if len(sensor_data.shape) > 1:
        column_index_combinations = generate_all_integer_combinations(sensor_data.shape[1])

        functions += [column_product_factory(t) for t in column_index_combinations]





    window_samples = int(sampling_rate * window_length)
    #print("Windows samples ", window_samples)
    step_size = int(round(window_samples * (1.0 - overlap)))


    all_features = []


    print("Function dict passed to segment acceleration and calculate features: ", function_dict)
    iteration = 0
    for window_start in np.arange(0, sensor_data.shape[0], step_size):
        #print("Window start: ", window_start, "Sensor_data.shape[0]: ", sensor_data.shape[0], step_size)
        window_start = int(round(window_start))
        window_end = window_start + int(round(window_samples))
        if window_end > sensor_data.shape[0]:
            break
        window = sensor_data[window_start:window_end]

        #print("Window", window)
        #print("Step size ", step_size)
        #print("Window ", window)
        # extracted_features = [func(window) for func in functions]
        extracted_features = []
        # print("Windows start: ")
        index_of_function = 0
        n = 0



        if(not type(function_dict) == int):

            function_number_in_function_dict = 0
            for function, index in function_dict.keys():
                len_extracted_feature_before = len(extracted_features)
                value = functions[index](window)
                if (isinstance(value, float)):
                    extracted_features.append(value)
                    extracted_value = value
                    n += 1
                else:
                    n += value.__len__()
                    #print("Value: ", value, value.__len__(), type(value))
                    if(type(value) == tuple):
                        value = np.asarray(value)
                    feature_indexes = function_dict[(function, index)]
                    #print(feature_indexes)
                    extracted_value = value[feature_indexes]
                    value = extracted_features.append(value[feature_indexes])
                    # print("/Window end", n)
                if(print_stats and iteration < 1):
                    #print(function, index, value, "value_index: ", function_dict[(function, index)], "Value addded: ", extracted_value)
                    print("Function number: ", function_number_in_function_dict, " that was calculated is function: ", function)
                function_number_in_function_dict +=1


            #print (function, " number of features: ", str(len(extracted_features)-len_extracted_feature_before))
            all_features.append(np.hstack(extracted_features))

        else:
            for func in functions:
                len_extracted_feature_before = len(extracted_features)
                value = func(window)
                if (isinstance(value, float)):
                    n += 1
                else:
                    n += value.__len__()

                extracted_features.append(value)

                    # print("/Window end", n)

            # print (function, " number of features: ", str(len(extracted_features)-len_extracted_feature_before))
            all_features.append(np.hstack(extracted_features))

        iteration += 1
    #print("Len all_features: ", len(all_features))
    # print("number of features ", len(all_features))
    one_large_array = np.vstack(all_features)
    # print("")
    if remove_sign_after_calculation:
        np.absolute(one_large_array, one_large_array)
    #print(one_large_array.shape)
    return one_large_array


def segment_labels(label_data, sampling_rate=1, window_length=3.0, overlap=0.0):
    window_samples = int(sampling_rate * window_length)
    step_size = int(round(window_samples * (1.0 - overlap)))

    labels = []

    for window_start in np.arange(0, label_data.shape[0], step_size):
        window_start = int(round(window_start))
        window_end = window_start + int(round(window_samples))
        if window_end > label_data.shape[0]:
            break
        window = label_data[window_start:window_end]
        top = find_majority_activity(window)
        labels.append(top)

    return np.array(labels)


def find_majority_activity(window):
    counts = Counter(window)
    top = counts.most_common(1)[0][0]
    return top
