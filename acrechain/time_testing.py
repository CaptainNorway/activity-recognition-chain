from collections import Counter

import warnings
import os
import numpy as np
import pickle
import json


from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from time import time
from acrechain import load_accelerometer_csv, load_label_csv, segment_acceleration_and_calculate_features, \
    segment_labels, definitions
from acrechain.function_selection import getFunctions
from acrechain.pipeline import load_csv_and_extract_features
import pprint as pp

# This module traines models with various number of feature (feature_count) and measures the time
# it takes for those models to extract windows from csv, calculate the features for the windows,
# and predicting the activities, for the new data


#Config

warnings.filterwarnings('ignore')

train_overlap = 0.8
n_jobs = -1
window_length = 3.0


model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
model_folder_reduced_features = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_reduced_features")
indexes_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indexes")
data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")



feature_importances_paths = {
    100: os.path.join(definitions.feature_importance_folder, "100.0hz_final_feature_importances_sorted_descending_importance.pickle")
}

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


feature_importances_dict = {}

for hz in feature_importances_paths:
    with open(feature_importances_paths[hz], "rb") as f:
        feature_importances_dict[hz] = pickle.load(f)


#List of all activities in the labeled data-set.
label_to_number_dict = {
    "none": 0,
    "walking": 1,
    "running": 2,
    "shuffling": 3,
    "stairs (ascending)": 4,
    "stairs (descending)": 5,
    "standing": 6,
    "sitting": 7,
    "lying": 8,
    "transition": 9,
    "lie_sit": 911,
    "lie_stand": 912,
    "lie_walk": 913,
    "sit_stand": 921,
    "sit_lie" : 922,
    "sit_walk" : 923,
    "stand_lie": 931,
    "stand_sit": 932,
    "stand_walk": 933,
    "walk_lie": 941,
    "walk_sit": 942,
    "walk_stand": 943,
    "bending": 10,
    "picking": 11,
    "undefined": 12,
    "cycling": 13,
    "cycling (stand)": 14,
    "heel drop": 15,
    "vigorous activity": 16,
    "non-vigorous activity": 17,
    "Transport(sitting)": 18,
    "Commute(standing)": 19,
    "lying (prone)": 20,
    "lying (supine)": 21,
    "lying (left)": 22,
    "lying (right)": 23,
}

number_to_label_dict = dict([(label_to_number_dict[l], l) for l in label_to_number_dict])

# The activities we are relabeling
relabel_dict = {
    # 3: 9,
    4: 1,
    5: 1,
    11: 10,
    14: 13,
    20: 8,
    21: 8,
    22: 8,
    23: 8
}
keep_set = {1, 2, 6, 7, 8, 10, 13}




def get_files(path):

    csvs_we_are_looking_for = ["LOWERBACK", "THIGH", "labels"]

    subject_files = []
    for r, ds, fs in os.walk(path):
        found_csvs = [False] * len(csvs_we_are_looking_for)

        for f in fs:
            print("f", f)
            for i, csv_string in enumerate(csvs_we_are_looking_for):
                if csv_string in f:
                    #path_string = os.path.join(r, f)
                    #corrected_path_string = path_string.replace(".icloud", "")
                    #second_corrected_path_string = corrected_path_string.replace("/.", "/")
                    #print("Corrected path string", corrected_path_string)
                    #print("Second corrected path string", second_corrected_path_string)
                    found_csvs[i] = os.path.join(r, f)

        if False not in found_csvs:
            subject_files.append(found_csvs)

    subject_files.sort()

    subject_ids = [os.path.basename(os.path.dirname(s)) for s, _, _ in subject_files]

    #print("subject ids: ", subject_ids)
    #print("subject files", subject_files)




    return subject_files





def find_majority_activity(window):
    counts = Counter(window)
    top = counts.most_common(1)[0][0]
    return top


def load_features_and_labels(lb_file, th_file, label_file, raw_sampling_frequency, keep_rate, keep_transitions, feature_importances):
    print("Loading features and labels")
    print("Loading", lb_file, "and", th_file)

    lb_data, th_data = load_accelerometer_csv(lb_file), load_accelerometer_csv(th_file)

    shape_before_resampling = lb_data.shape

    print("Shape input data: ", lb_data.shape, th_data.shape, "for ", lb_file, " and ", th_file)
    lb_data_resampled, th_data_resampled = [], []

    if keep_rate > 1:
        print("Resampling data with window size", keep_rate)
        end_of_data = lb_data.shape[0]
        for window_start in range(0, end_of_data, keep_rate):
            window_end = min((window_start + keep_rate), end_of_data)
            average_of_lb_window = np.average(lb_data[window_start:window_end], axis=0)
            average_of_th_window = np.average(th_data[window_start:window_end], axis=0)
            lb_data_resampled.append(average_of_lb_window)
            th_data_resampled.append(average_of_th_window)

        lb_data, th_data = np.vstack(lb_data_resampled), np.vstack(th_data_resampled)
        shape_after_resampling = lb_data.shape
        print("Before resampling:", shape_before_resampling, "After resampling:", shape_after_resampling)

    resampled_sampling_frequency = raw_sampling_frequency / keep_rate

    print("Segmenting and calculating features for", lb_file, "and", th_file)

    back_function = getFunctions("back", feature_importances)
    thigh_function = getFunctions("thigh", feature_importances)

    a = time()
    lb_windows = segment_acceleration_and_calculate_features(lb_data, back_function, sampling_rate=resampled_sampling_frequency, window_length=window_length,
                                                             overlap=train_overlap)
    b = time()
    th_windows = segment_acceleration_and_calculate_features(th_data, thigh_function, sampling_rate=resampled_sampling_frequency,  window_length=window_length,
                                                             overlap=train_overlap)
    c = time()



    time_statistics_segment_and_calculate_features= {}

    time_segment_and_calculate_features_lower_back = b-a
    time_segment_and_calculate_features_thigh = c-b
    time_segment_and_calculate_features_both_sensors = c-a

    time_statistics_segment_and_calculate_features["Lower back sensor"] = time_segment_and_calculate_features_lower_back
    time_statistics_segment_and_calculate_features["Thigh sensor"] = time_segment_and_calculate_features_thigh
    time_statistics_segment_and_calculate_features["Both sensors"] = time_segment_and_calculate_features_both_sensors


    print("Shape lb windows: ", lb_windows.shape, " for ", lb_file, " and ", th_file)
    print("Shape th windows: ", th_windows.shape)


    features = np.hstack([lb_windows, th_windows])
    print("Shape of features combined for both sensors: ", features.shape)


    #Relabel activities in the label file

    print("Loading", label_file)
    label_data = load_label_csv(label_file)
    print("Relabeling", label_file)


    for k in relabel_dict:
        np.place(label_data, label_data == k, [relabel_dict[k]])

    if (keep_transitions):
        #Introduce the five different transition types
        transitionDict = {(8, 7): 911, (8, 6): 912, (8, 1): 913, (7, 6): 921, (7, 8): 922, (7, 1): 923, (6, 8): 931,
                          (6, 7): 932, (6, 1): 933, (1, 8): 941, (1, 7): 942, (1, 6): 943}
        prevValue = -1
        value = -1
        preTransitionActivity = -1
        postTransitionActivity = -1
        transitionCounter = 0
        for i in range(len(label_data)):
            if (value == -1):
                value = label_data[i]
            else:
                prevValue = value
                value = label_data[i]
                if (value == 9):
                    if(transitionCounter == 0):
                        preTransitionActivity = prevValue
                    transitionCounter += 1
                else:
                    if(prevValue == 9):
                        postTransitionActivity = value
                        #print("preTransitionActivity ",preTransitionActivity)
                        #print("postTransitionActivity ",postTransitionActivity)
                        if((preTransitionActivity, postTransitionActivity) in transitionDict):
                            transition_type = transitionDict.get((preTransitionActivity, postTransitionActivity))
                        else:
                            transition_type = 9
                        for j in range(transitionCounter):
                            label_data[i-transitionCounter+j] = transition_type
                        transitionCounter = 0
        transitions = [911, 912, 913, 921, 922, 923, 931, 932, 933, 941, 942, 943]
        for i in transitions:
            keep_set.add(i)



    #Resample lab data
    if keep_rate > 1:
        print("Resampling label data with window size", keep_rate)
        end_of_label_data = len(label_data)

        label_data_resampled = []

        for window_start in range(0, end_of_label_data, keep_rate):
            window_end = min(window_start + keep_rate, end_of_label_data)

            label_data_resampled.append(find_majority_activity(label_data[window_start:window_end]))

        label_data = np.hstack(label_data_resampled)

        print("Before resampling:", end_of_label_data, "After resampling:", len(label_data))

    print("Segmenting", label_file)

    lab_windows = segment_labels(label_data, window_length=window_length, overlap=train_overlap,
                                 sampling_rate=resampled_sampling_frequency)

    print("Removing unwanted activities from", label_file)
    #print("Lab windows", lab_windows)
    #print("Type ", type(lab_windows))

    indices_to_keep = [i for i, a in enumerate(lab_windows) if a in keep_set]
    features = features[indices_to_keep]
    lab_windows = lab_windows[indices_to_keep]



    return features, lab_windows, time_statistics_segment_and_calculate_features





#Trains a model with the top feature_count features by feature importance
def train_with_feature_count(feature_importances, subject_windows):

    print("Subject windows shape: ", type(subject_windows))

    feature_count = len(feature_importances)

    a = time()

    train_X, train_y = zip(*subject_windows)
    train_X, train_y = np.vstack(train_X), np.hstack(train_y)

    print("Shape of training set for this model: ", train_X.shape)


    indexes = getFeatureIndexes(feature_importances)
    train_X = getFeatures(train_X, indexes)

    print("Indexes of features to be used for training the model")
    print("Shape of training set X, after deleting the unecessary features: ", train_X.shape)
    print("Shape of labels y: ", train_y.shape)

    forest_with_feature_count_features = RFC(n_estimators=50, class_weight="balanced", random_state=0, n_jobs=-1).fit(train_X,
                                                                                                        train_y)
    b = time ()

    print("Time spent training model with ", feature_count, " features: ", b-a, "s")

    time_statistics_model_training = b-a

    return time_statistics_model_training, forest_with_feature_count_features





#Returns the indexes of the features with feature importance values greater than the feature importance for the feature below the top percentage#
def getFeatureIndexes(feature_importances):
    feature_count = len(feature_importances)
    feature_indexes = []
    for i in range(0, feature_count):
        feature_indexes.append(feature_importances[i][0])
    print("Features indexes extracted: ", feature_indexes)
    print("Number of features ", len(feature_indexes))
    return feature_indexes



#Returns the features of a data set which corresponding to the feature indexes#
def getFeatures(data, indexes):
    #print("Data set shape before reshape: ", data.shape)
    data = data[:, indexes]
    #print("Data set shape afer reshape ", data.shape)
    return data


def save_statistics(statistics, path):
    with open(path, "w") as f:
        json.dump(statistics, f, sort_keys=True, indent=4, separators=(',', ': '))



if __name__ == "__main__":

    #The models to train/create and run the time tests on

    start = 2
    end = 2
    step = 1
    models_to_create = (start, end, step)

    sampling_frequency = 100
    keep_rate = 1
    keep_transitions = 0

    feature_importances = feature_importances_dict[sampling_frequency]
    subject_files = get_files(data_folder)

    print(subject_files)

    back_csv_path = os.path.join(data_paths[6], "006_LOWERBACK.csv")
    thigh_csv_path = os.path.join(data_paths[6], "006_THIGH.csv")




    feature_count = 138

    print("Loading features and labels for the TFL datasets with n_features = ", feature_count)

    subject_windows_and_time_statistics = Parallel(n_jobs=n_jobs)(
        delayed(load_features_and_labels)(lb_file, th_file, label_file, sampling_frequency, keep_rate, keep_transitions, feature_importances)
        for
        lb_file, th_file, label_file in subject_files)

    subject_windows = []
    time_statistics = []
    for subject in subject_windows_and_time_statistics:
        subject_windows.append(subject[:2])
        time_statistics.append(subject[-1])

    print("Len subject windows: ", len(subject_windows))

    print("Time statistics for loading features and labels for the TFL dataset: ", time_statistics)

    print("Finished loading features and labels for the total TFL dataset")


    print("Starting time testing")
    feature_counts = [count for count in range(start, end + 1, step)]
    print("Testing with the following feature counts: ", feature_counts)



    #The statistics for both training all the models and predicting with all the
    time_statistics_all_models = {}
    for feature_count in feature_counts:
        model_statistics = {}
        feature_importances_feature_count = feature_importances[:feature_count]

        print("Training RFC model with the top ", feature_count, " features and the following feature importances: ", feature_importances_feature_count)
        time_statistics_model_training, RFC_model_with_feature_count_features = train_with_feature_count(feature_importances_feature_count, subject_windows)
        feature_count_string = "00" + str(feature_count) + " features"
        model_statistics["1.Time to train model"] = round(time_statistics_model_training,3)

        print("Time statistics for model with: ", feature_count, " features")

        print("Running prediction on the HUNT4 sample subject")
        # Prediction phase
        # 3 phases:
        #1: load csv

        #Run 3 iterations:
        average_statistics = {}

        iterations = 3

        for i in range(iterations):
            print("Running prediction iteration ", i, " on feature count ", feature_count)
            print("Loading CSVs: \n",  "back_csv_path: ", back_csv_path, "\n", "thigh_csv_path: ", thigh_csv_path)

            predictions, time_statistics_predictions = load_csv_and_extract_features(back_csv_path, thigh_csv_path, sampling_frequency, feature_importances_feature_count, RFC_model_with_feature_count_features, minutes_to_read_in_a_chunk=60, print_stats = True)


            average_statistics[i] = time_statistics_predictions

        sum_extract_windows = 0
        sum_calculate_feature = 0
        sum_predict = 0
        for key, value in average_statistics.items():
            sum_extract_windows += value["1.Extracting windows from CSVs"]
            sum_calculate_feature += value["2.Calculate features for all windows"]
            sum_predict += value["3.Predicting"]

        average_extract_windows = sum_extract_windows/iterations
        average_calculate_features = sum_calculate_feature/iterations
        average_predict = sum_predict/iterations

        average_statistics = {}
        average_statistics["1.Extracting windows from CSVs"] = round(average_extract_windows,3)
        average_statistics["2.Calculate features for all windows"] = round(average_calculate_features,3)
        average_statistics["3.Predicting"] = round(average_predict,3)


        model_statistics["2.Prediction pipeline"] = average_statistics

        time_statistics_all_models[feature_count_string[-12:]] = model_statistics


        print("Finished with training and predicting for the model with the ", feature_count, " highest feature importance features")






    pp.pprint(time_statistics_all_models)

    # Saving statistics
    print("Saving statistics")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistics")
    path = os.path.join(path, "100.0hz_time_statistics_training_and_predicting_" + "(" + str(start) + "," + str(end) + "," + str(step) + ").json")
    save_statistics(time_statistics_all_models, path)

