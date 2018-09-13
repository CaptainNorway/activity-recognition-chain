import os

from acrechain.conversion import timesync_from_cwa
from acrechain import segment_and_calculate_features_temperature
from acrechain import statistics_helpers as sh
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import collections
import pandas as pd



data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA_SNT")
plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PLOTS")


data_paths = {
    "P1_S1": os.path.join(data_folder, "P1_S1"),
    "P1_S2": os.path.join(data_folder, "P1_S2"),
    "P2_S1": os.path.join(data_folder, "P2_S1"),
    "P2_S2": os.path.join(data_folder, "P2_S2"),
}






def transform_ADC_to_celsius(ADC_reading):
    temperature_celsius = (ADC_reading*300/1024)-50




def transform_cwa_to_csv(protocol, subject):
    path_string = "P" + protocol + "_S" + subject
    if (subject == "1"):
        name = "atle"
    elif (subject == "2"):
        name = "vegar"

    back_cwa = os.path.join(data_paths[path_string], "P" + protocol + "_" + name + "_B.cwa")
    thigh_cwa = os.path.join(data_paths[path_string], "P" + protocol + "_" + name + "_T.cwa")

    print(back_cwa)
    print(thigh_cwa)

    output_path = os.path.join(data_paths[path_string], "timestamped_predictions.csv")


    back_csv_path, thigh_csv_path, time_csv_path = timesync_from_cwa(back_cwa, thigh_cwa)



def read_temperature_file(protocol, subject):
    path_string = "P" + protocol + "_S" + subject
    if (subject == "1"):
        name = "atle"
    elif (subject == "2"):
        name = "vegar"



    back_temperature_path= os.path.join(data_paths[path_string], "P" + protocol + "_" + name + "_B_temp_corrected.csv")
    thigh_temperature_path = os.path.join(data_paths[path_string], "P" + protocol + "_" + name + "_T_temp_corrected.csv")
    labels_path = os.path.join(data_paths[path_string], "P" + protocol + "_" + name + "_labels.csv")

    if(protocol == "2"):
        if (name == "atle"):
            delimitter = ";"
        else:
            delimitter = ","
    else:
        delimitter = ","


    print("Reading sensor file for ", path_string)
    back_temperature_readings = pd.read_csv(back_temperature_path,
                                           delimiter=delimitter, header=None).as_matrix([1])
    thigh_temperature_reading = pd.read_csv(thigh_temperature_path,
                                           delimiter=delimitter, header=None).as_matrix([1])
    labels = pd.read_csv(labels_path, delimiter=";", header=None).as_matrix([0])




    print(back_temperature_readings.shape)
    print(thigh_temperature_reading.shape)
    print(labels.shape)

    #print(back_temperature_readings)
    #print(thigh_temperature_reading)
    #print(labels)



    return back_temperature_readings, thigh_temperature_reading, labels





def train_no_wear_time_model(back_sensor_readings, thigh_sensor_readings, label_data, samples_pr_window, train_overlap):

    # Calculate features
    back_features = segment_and_calculate_features_temperature.segment_acceleration_and_calculate_features(back_sensor_readings, samples_pr_window=samples_pr_window, overlap=train_overlap)
    thigh_features = segment_and_calculate_features_temperature.segment_acceleration_and_calculate_features(thigh_sensor_readings, samples_pr_window = samples_pr_window, overlap=train_overlap)

    print(back_features)
    print(thigh_features)

    print("Shape back_features: ", back_features.shape)
    print("Shape thigh features: ", thigh_features.shape)

    label_windows = segment_and_calculate_features_temperature.segment_labels(label_data, samples_pr_window = samples_pr_window,  window_length=window_length, overlap=train_overlap)

    print(label_windows)
    print("Label_windows[14]: ", label_windows[14])

    boths_features = np.hstack((back_features, thigh_features))

    print(boths_features)
    print("Shape boths features: ", boths_features.shape)

    train_X = boths_features
    train_y = label_windows

    RFC_classifier = RFC(n_estimators=50, class_weight="balanced", random_state=0, n_jobs=-1).fit(train_X, train_y)

    print(RFC_classifier.feature_importances_)
    print(RFC_classifier)

    return  RFC_classifier



def train_no_wear_time_model_multiple_subjects(subjects, samples_pr_window, train_overlap):

    print("Training models for the following subjects (protocol, subject): \n", subjects)


    all_features = []
    all_labels = []

    for subject in subjects:
        print("Training subject ", subject[1], " on protocol ", subject[0])
        back_sensor_readings, thigh_sensor_readings, labels = read_temperature_file(subject[0], subject[1])
        # Calculate features
        back_features = segment_and_calculate_features_temperature.segment_acceleration_and_calculate_features(back_sensor_readings, samples_pr_window=samples_pr_window, overlap=train_overlap)
        thigh_features = segment_and_calculate_features_temperature.segment_acceleration_and_calculate_features(thigh_sensor_readings, samples_pr_window = samples_pr_window, overlap=train_overlap)

        label_windows = segment_and_calculate_features_temperature.segment_labels(labels, samples_pr_window = samples_pr_window,  window_length=window_length, overlap=train_overlap)


        boths_features = np.hstack((back_features, thigh_features))

        print("Shape back_features: ", back_features.shape)
        print("Shape thigh features: ", thigh_features.shape)
        print("Shape boths features: ", boths_features.shape)
        print("Shape labels: ", labels.shape)

        all_features.append(boths_features)
        all_labels.append(label_windows)

    #print("All features: ", all_features)

    train_X = np.concatenate(all_features)
    train_y = np.concatenate(all_labels)
    print("All features shape: ", train_X.shape)
    print("All labels shape: ", train_y.shape)

    RFC_classifier = RFC(n_estimators=100, class_weight="balanced", random_state=0, n_jobs=-1).fit(train_X, train_y)

    #print(RFC_classifier.feature_importances_)
    #print(RFC_classifier)

    return  RFC_classifier


# Subject wise cross validation
def test(subjects, samples_pr_window, train_overlap):

    print("Running test on the following protocols and subjects: \n", subjects)

    accuracy_all_subjects = {}
    all_test_y = []
    all_predictions = []


    for subject in subjects:
        print("Testing subject ", subject[1], " with protocol ", subject[0])


        subjects_to_train = subjects.copy()
        subjects_to_train.remove(subject)
        subject_to_predict = subject



        RFC_model = train_no_wear_time_model_multiple_subjects(subjects_to_train, samples_pr_window, train_overlap)


        # Get reading for testing subject
        back_temperature_readings, thigh_temperature_readings, labels = read_temperature_file(subject_to_predict[0], subject_to_predict[1])


        # Extract features

        back_features = segment_and_calculate_features_temperature.segment_acceleration_and_calculate_features(
            back_temperature_readings, samples_pr_window=samples_pr_window, overlap=0.0)
        thigh_features = segment_and_calculate_features_temperature.segment_acceleration_and_calculate_features(
            thigh_temperature_readings, samples_pr_window=samples_pr_window, overlap=0.0)

        label_windows = segment_and_calculate_features_temperature.segment_labels(labels, samples_pr_window = samples_pr_window, overlap=0.0)

        boths_features = np.hstack((back_features, thigh_features))

        print("Shape back_features: ", back_features.shape)
        print("Shape thigh features: ", thigh_features.shape)
        print("Shape labels: ", label_windows.shape)
        print("Shape boths_features: ", boths_features.shape)



        test_X = boths_features
        test_y = label_windows
        print("Test y: ", test_y)

        predictions = RFC_model.predict(test_X)

        all_test_y.append(test_y)
        all_predictions.append(predictions)

        print("Predictions: ", predictions)
        accuracy = accuracy_score(test_y, predictions)
        print("\n\n")
        print("Subject tested on: ", subject[1], ", protocol: ", subject[0])
        print("Accuracy: ", accuracy)
        print("\n\n")
        subject = tuple(subject)
        accuracy_all_subjects[subject] = accuracy

        print(confusion_matrix(test_y, predictions))


    all_test_y_concatenated = np.concatenate(all_test_y)
    all_predictions_concatenated = np.concatenate(all_predictions)

    counter1 = collections.Counter(all_test_y_concatenated)
    counter2 = collections.Counter(all_predictions_concatenated)

    print("All ground thruth labels: ", counter1)
    print("All prediction labels: ", counter2)


    print()
    print("Confusion matrix across all predictions: \n")
    confusion_matrix_ = confusion_matrix(all_test_y_concatenated,all_predictions_concatenated)

    y_true = pd.Series(all_test_y_concatenated)
    y_pred = pd.Series(all_predictions_concatenated)

    print()
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print()

    accuracy_accross_all_subjects = accuracy_score(all_test_y_concatenated, all_predictions_concatenated)
    print("Accuracy across all subjects: ", accuracy_accross_all_subjects)
    print("Accuracy for each subject (protocol, subject_id) (subject-wise cross validation): \n", accuracy_all_subjects)
    return accuracy_all_subjects, accuracy_accross_all_subjects, confusion_matrix_, y_true, y_pred

if __name__ == "__main__":

    train_overlap = 0.8
    window_length = 120 # The length of the window in seconds
    sampling_frequency = 50 # The sampling frequency of the original data
    keep_rate = 1
    temperature_reading_rate = 120 #There are 120 samples between every temperature reading

    samples_pr_second = 1/(temperature_reading_rate/sampling_frequency)
    samples_pr_window = int(window_length*samples_pr_second)
    print(samples_pr_second)
    print(samples_pr_window)
    print(samples_pr_window*samples_pr_second)

    #protocol = "1"
    #subject = "1"
    #back_temperature_readings, thigh_temperature_readings, labels = read_temperature_file(protocol,subject)
    #RFC_model = train_no_wear_time_model(back_temperature_readings, thigh_temperature_readings, labels, samples_pr_window, train_overlap)

    subjects = [["1", "1"], ["1", "2"], ["2", "1"], ["2", "2"]]


    accuracy_all_subjects, accuracy_across_all_subjects, confusion_matrix_, y_true, y_pred = test(subjects, samples_pr_window, train_overlap)

    save_path = os.path.join(plots_folder, "confusion_matrix_no_wear_time_cross_validation_all_subjects.png")
    title = "Confusion matrix"
    classes = ["A", "B", "T"]
    print("Generating confusion matrix")
    sh.generate_and_save_confusion_matrix(y_true, y_pred, classes, save_path, title)
    sh.generate_statistics_simple(y_true, y_pred)