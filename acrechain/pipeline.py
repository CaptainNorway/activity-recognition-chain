import numpy as np
import pickle
import itertools
import os
from time import time

import pandas.errors
import pandas as pd
from acrechain import definitions
from acrechain.conversion import timesync_from_cwa
from acrechain.segment_and_calculate_features import segment_acceleration_and_calculate_features
from acrechain.function_selection import getFunctions

project_root = os.path.dirname(os.path.abspath(__file__))


model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
model_folder_reduced_features = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_reduced_features")
indexes_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexes")
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
hunt4_data_folder = os.path.join(project_root, "DATA_HUNT4")


data_paths = {
    1: os.path.join(data_folder, "001"),
    2: os.path.join(data_folder, "002"),
    3: os.path.join(data_folder, "003"),
    4: os.path.join(data_folder, "004"),
    5: os.path.join(data_folder, "005"),
    6: os.path.join(data_folder, "006"),
    8: os.path.join(data_folder, "008"),
    9: os.path.join(data_folder, "009"),
    10: os.path.join(data_folder, "010"),
    11: os.path.join(data_folder, "011"),
    12: os.path.join(data_folder, "012"),
    13: os.path.join(data_folder, "013"),
    14: os.path.join(data_folder, "014"),
    15: os.path.join(data_folder, "015"),
    16: os.path.join(data_folder, "016"),
    17: os.path.join(data_folder, "017"),
    18: os.path.join(data_folder, "018"),
    19: os.path.join(data_folder, "019"),
    20: os.path.join(data_folder, "020"),
    21: os.path.join(data_folder, "021"),
    22: os.path.join(data_folder, "022"),
}

model_paths = {
    100: os.path.join(model_folder, "healthy_3.0s_model_100.0hz.pickle"),
    50: os.path.join(model_folder, "healthy_3.0s_model_50.0hz.pickle"),
    25: os.path.join(model_folder, "healthy_3.0s_model_25.0hz.pickle"),
    20: os.path.join(model_folder, "healthy_3.0s_model_20.0hz.pickle"),
    10: os.path.join(model_folder, "healthy_3.0s_model_10.0hz.pickle"),
    5: os.path.join(model_folder, "healthy_3.0s_model_5.0hz.pickle"),
    4: os.path.join(model_folder, "healthy_3.0s_model_4.0hz.pickle"),
    2: os.path.join(model_folder, "healthy_3.0s_model_2.0hz.pickle"),
    1: os.path.join(model_folder, "healthy_3.0s_model_1.0hz.pickle"),
}

model_reduced_feature_paths = {
    1: os.path.join(model_folder_reduced_features, "healthy_3.0s_model_1.0hz_reduced.pickle")
}

indexes_paths = {
    1: os.path.join(indexes_folder, "indexes_0.4_percent_healthy_3.0s_model_1.0hz_reduced.pickle")
}


models = dict()
models_reduced_features = dict()
indexes = dict()

#for hz in model_paths:
#    with open(model_paths[hz], "rb") as f:
#       models[hz] = pickle.load(f)

#for hz in model_reduced_feature_paths:
#    with open(model_reduced_feature_paths[hz], "rb") as f:
 #       models_reduced_features[hz] = pickle.load(f)

for hz in indexes_paths:
    with open(indexes_paths[hz], "rb") as f:
        indexes[hz] = pickle.load(f)


window_length = 3.0
overlap = 0.0


def complete_end_to_end_prediction(back_cwa, thigh_cwa, end_result_path, sampling_frequency, reduced_feature_set, minutes_to_read_in_a_chunk=15):
    #a = time()
    #back_csv_path, thigh_csv_path, time_csv_path = timesync_from_cwa(back_cwa, thigh_cwa)

    #back_csv_path = os.path.join(data_paths[6], "006_LOWERBACK.csv")
    #thigh_csv_path = os.path.join(data_paths[6], "006_THIGH.csv")

    #b = time()
    #print("TIME: Conversion and sync:", format(b - a, ".2f"), "s")

    if (reduced_feature_set):
        c = time()
        predictions = load_csv_and_extract_features(back_csv_path, thigh_csv_path, sampling_frequency,
                                                    minutes_to_read_in_a_chunk, reduced_feature_set = True)
        d = time()
        print("TIME: Feature extraction and prediction with reduced feature set:", format(d - c, ".2f"), "s")


    else:
        a = time()
        predictions = load_csv_and_extract_features(back_csv_path, thigh_csv_path, sampling_frequency,
                                                    minutes_to_read_in_a_chunk, reduced_feature_set = False)
        b = time()
        print("TIME: Feature extraction and prediction:", format(b - a, ".2f"), "s")
        time_stamp_skip = int(sampling_frequency * window_length * (1.0 - overlap))
    #a = time()



    #with open(time_csv_path, "r") as t:
    #    time_stamp_lines = [_.strip() for _ in itertools.islice(t, 0, None, time_stamp_skip)]

    #output_lines = [tsl + ", " + str(pred) + "\n" for tsl, pred in zip(time_stamp_lines, predictions)]

   # with open(end_result_path, "w") as ef:
    #    ef.writelines(output_lines)
    #b = time()
    #print("TIME: Writing to disk:", format(b - a, ".2f"), "s")

    #for tmp_file in [back_csv_path, thigh_csv_path, time_csv_path]:
    #    os.remove(tmp_file)




def load_csv_and_extract_features(back_csv_path, thigh_csv_path, sampling_frequency, feature_importances, RFC_model_with_feature_count_features, minutes_to_read_in_a_chunk, print_stats = False):

    feature_count = len(feature_importances)

    number_of_samples_in_a_window = int(sampling_frequency * window_length)
    number_of_windows_to_read = int(round(minutes_to_read_in_a_chunk * 60 / window_length))
    number_of_samples_to_read = number_of_samples_in_a_window * number_of_windows_to_read

    if(print_stats):


        print("Number of samples in a window: ", number_of_samples_in_a_window)
        print("Number of windows to read: ", number_of_windows_to_read)
        print("Number of samples to read", number_of_samples_to_read)

    window_start = 0

    predictions = []


    sum_time_extract_windows_csv = 0
    sum_time_calculate_features = 0
    sum_time_predict = 0
    while True:
        try:
            a = time()
            this_back_window = pd.read_csv(back_csv_path, skiprows=window_start, nrows=number_of_samples_to_read,
                                           delimiter=",", header=None).as_matrix()
            this_thigh_window = pd.read_csv(thigh_csv_path, skiprows=window_start, nrows=number_of_samples_to_read,
                                            delimiter=",", header=None).as_matrix()



            print(this_thigh_window.size, this_thigh_window.shape)

            b = time()
            sum_time_extract_windows_csv += b-a

            window_start += number_of_samples_to_read


            back_functions = getFunctions("back", feature_importances, print_stats)
            thigh_functions = getFunctions("thigh", feature_importances, print_stats)



            c = time()


            if (not len(back_functions) == 0):
                back_features = segment_acceleration_and_calculate_features(this_back_window, back_functions,
                                                                        sampling_rate=sampling_frequency,
                                                                        window_length=window_length, overlap=overlap, print_stats=print_stats)

                #print("back_features", back_features)
                print("shape back_features", back_features.shape)

            if(not len(thigh_functions) == 0):
                thigh_features = segment_acceleration_and_calculate_features(this_thigh_window, thigh_functions,
                                                                         sampling_rate=sampling_frequency,
                                                                         window_length=window_length, overlap=overlap, print_stats=print_stats)

                #print("thigh_features", thigh_features)
                print("thigh features shape", thigh_features.shape)

            if(not len(back_functions) == 0 and not len(thigh_functions) == 0):
                boths_features = np.hstack((back_features, thigh_features))
            else:
                if(len(back_functions) == 0):
                    boths_features = thigh_features
                else:
                    boths_features = back_features


            print("Features fed into RFC: ", boths_features.shape)
            print(boths_features)

            d = time()

            sum_time_calculate_features += d-c



            a = time()
            this_windows_predictions = RFC_model_with_feature_count_features.predict(boths_features)
            print("Tree has ", RFC_model_with_feature_count_features.n_features_, " features!")
            b = time()
            sum_time_predict += b-a


            predictions.append(this_windows_predictions)
            #print("len(predictions): ", len(predictions))
        except pandas.errors.EmptyDataError:  # There are no more lines to read
            break

    predictions = np.hstack(predictions)
    print("TIME: Extract windows from CSV files: ", format(sum_time_extract_windows_csv, ".2f"), "s")
    print("TIME: Calculate features: ", format(sum_time_calculate_features, ".2f"), "s")
    print("TIME: Predict: ", format(sum_time_predict, ".2f"), "s")

    time_statistics = {}
    time_statistics["1.Extracting windows from CSVs"] = sum_time_extract_windows_csv
    time_statistics["2.Calculate features for all windows"] = sum_time_calculate_features
    time_statistics["3.Predicting"] = sum_time_predict


    return predictions, time_statistics







if __name__ == "__main__":

    cwa_1 = os.path.join(hunt4_data_folder, "4000006/4000006-28025_2018-04-19_B.cwa")
    cwa_2 = os.path.join(hunt4_data_folder, "4000006/4000006-31954_2018-04-19_T.cwa")

    reduced_feature_set = False
    sampling_frequency = 100
    keep_rate = int(round(100/sampling_frequency))


    output_path = os.path.join(hunt4_data_folder, "timestamped_predictions_test.csv")

    print(project_root)
    print(hunt4_data_folder)
    print(output_path)

    print(cwa_1)
    print(cwa_2)

    back_csv_path, thigh_csv_path, time_csv_path = timesync_from_cwa(cwa_1, cwa_2)


    #complete_end_to_end_prediction(cwa_1, cwa_2, output_path, sampling_frequency, reduced_feature_set, minutes_to_read_in_a_chunk=60)



    #back_csv_path = os.path.join(data_paths[6], "006_LOWERBACK.csv")
    #thigh_csv_path = os.path.join(data_paths[6], "006_THIGH.csv")

    #predictions = load_csv_and_extract_features(back_csv_path, thigh_csv_path, 100, 15, 1)
    #print(len(predictions))
    #for v in predictions:
    #   print(v)
