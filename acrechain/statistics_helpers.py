from __future__ import division, print_function

import itertools
import json
from collections import defaultdict

import os

import numpy as np

import matplotlib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


matplotlib.use("Agg")  # Set non-interactive background. Must precede pyplot import.
import matplotlib.pyplot as plt
import pandas as pd




def save_confusion_matrix_image(matrix, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.summer,
                                save_path="./whatever.png"):
    """
    Modified version of http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    row_sums = matrix.sum(axis=1)
    shade_matrix = (matrix.transpose() / row_sums).transpose()

    plt.clf()
    plt.imshow(shade_matrix, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.tick_params(labelsize=15)
    plt.xticks(tick_marks, classes, rotation=45, fontsize = 15)
    plt.yticks(tick_marks, classes, fontsize = 15)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, matrix[i, j],
                 horizontalalignment="center",
                 color="black", fontsize=15)

    plt.tight_layout()
    plt.ylabel('True label\n', fontsize = 20)
    plt.xlabel('\nPredicted label', fontsize = 20)
    plt.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 8.5)
    plt.subplots_adjust(right = 0.9)
    plt.savefig(save_path, dpi=120)


def specificity_score(y_true, y_pred):
    occurring_classes = sorted(list(set(y_true)))

    specificity_list = []

    for c in occurring_classes:
        binary_y_true = [True if label == c else False for label in y_true]
        binary_y_pred = [True if label == c else False for label in y_pred]

        # This step is inspired by http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
        tn, fp, _, _ = confusion_matrix(binary_y_true, binary_y_pred).ravel()

        specificity_list.append(tn / (tn + fp))

    return specificity_list



def generate_statistics_simple(y_true, y_pred):
    # Statistics for all subjects in the dataset:

    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F score: ", f_score)
    print("Support: ", support)
    print("Specificity: ", specificity)







def generate_statistics(y_true, y_pred, number_to_class_name_dict, feature_count, indexes):
    # Statistics for all subjects in the dataset:

    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F score: ", f_score)
    print("Support: ", support)
    print("Specificity: ", specificity)

    # Statistics for each activity:

    occurring_classes = sorted(list(set(y_true)))
    classes = [number_to_class_name_dict[c] for c in occurring_classes]

    recall_dict = dict([(c, s) for c, s in zip(classes, recall)])
    precision_dict = dict([(c, s) for c, s in zip(classes, precision)])
    f_score_dict = dict([(c, s) for c, s in zip(classes, f_score)])
    support_dict = dict([(c, s) for c, s in zip(classes, support)])
    specificity_dict = dict([(c, s) for c, s in zip(classes, specificity)])
    d = {"model_feature_count": feature_count,
         "indexes": indexes,
         "accuracy": accuracy,
         "recall": recall_dict,
         "precision": precision_dict,
         "specificity": specificity_dict,
         "f_score": f_score_dict,
         "support": support_dict,

         }

    # print(d)
    for key in d.keys():
        #    print("Key: ", key)
        if (isinstance(d[key], dict)):
            for value in d[key].keys():
                # print(value)
                # print(type(value))
                # print(type(d[key][value]))
                if (isinstance(d[key][value], np.int64)):
                    d[key][value] = int(d[key][value])
                    # d[key][key]

    # print(d)
    return d


def generate_and_save_statistics_json(y_true, y_pred, number_to_class_name_dict, save_path):
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F score: ", f_score)
    print("Support: ", support)
    print("Specificity: ", specificity)

    print(os.path.split(save_path)[1], accuracy)

    occurring_classes = sorted(list(set(y_true)))

    classes = [number_to_class_name_dict[c] for c in occurring_classes]

    save_statistics_json(accuracy, classes, f_score, precision, recall, save_path, support, specificity)


def save_statistics_json(accuracy, classes, f_score, precision, recall, save_path, support, specificity):
    recall_dict = dict([(c, s) for c, s in zip(classes, recall)])
    precision_dict = dict([(c, s) for c, s in zip(classes, precision)])
    f_score_dict = dict([(c, s) for c, s in zip(classes, f_score)])
    support_dict = dict([(c, s) for c, s in zip(classes, support)])
    specificity_dict = dict([(c, s) for c, s in zip(classes, specificity)])
    d = {
        "recall": recall_dict,
        "precision": precision_dict,
        "specificity": specificity_dict,
        "f_score": f_score_dict,
        "support": support_dict,
        "accuracy": accuracy
    }

    print("Statistics!")
    print(d)
    # with open(save_path, "w") as f:
    #    json.dump(d, f)


def generate_and_save_confusion_matrix(y_true, y_pred, class_names, save_path, title=""):
    matrix = confusion_matrix(y_true, y_pred)
    original_labels = set(y_true)
    all_occurring_classes = set(y_pred) | original_labels
    # print ("All occuring classes: !!!!!!!", all_occurring_classes)
    #class_names = [number_to_label_dict[x] for x in sorted(list(all_occurring_classes))]
    save_confusion_matrix_image(matrix, class_names, save_path=save_path, title=title)


def f_score(tp, fp, fn, beta):
    beta_squared = beta * beta
    score = (1 + beta_squared) * tp / ((1 + beta_squared) * tp + beta_squared * fn + fp)
    return score


def print_accuracies(statistic_folder):
    subject_set = set()

    for _, _, files in os.walk(statistic_folder):
        this_folders_subject_names = {os.path.splitext(f)[0] for f in files}
        subject_set |= this_folders_subject_names

    subject_list = sorted(subject_set, key=lambda x: x.lower())

    for k in ["general_population", "adaptation", "best_individual"]:
        print()
        print(k)
        print("\t" + "\t".join(subject_list))
        for root, dirs, files in os.walk(statistic_folder):
            if k in root:
                sub_test_name = os.path.split(os.path.split(root)[0])[1]
                print(sub_test_name, end="\t")
                for s in subject_list:
                    f = os.path.join(root, s + ".json")
                    if os.path.exists(f):
                        with open(f, "r") as g:
                            print(json.load(g)["accuracy"], end="\t")
                    else:
                        print("", end="\t")

                print()


def average_dicts(dicts):
    all_results = dict()

    for d in dicts:
        for score_type in d:
            score_data = d[score_type]
            if type(score_data) == float:
                try:
                    all_results[score_type].append(score_data)
                except KeyError:
                    all_results[score_type] = [score_data]
            else:
                try:
                    _ = all_results[score_type]
                except KeyError:
                    all_results[score_type] = dict()
                for activity in score_data:
                    activity_score = score_data[activity]
                    try:
                        all_results[score_type][activity].append(activity_score)
                    except KeyError:
                        all_results[score_type][activity] = [activity_score]

    for k in all_results:
        if type(all_results[k]) == list:
            all_results[k] = sum(all_results[k]) / len(all_results[k])
        else:
            sub_dict = all_results[k]
            for s_k in sub_dict:
                sub_dict[s_k] = sum(sub_dict[s_k]) / len(sub_dict[s_k])

    return all_results


def _individualfactory():
    return defaultdict(list)


def _strategyfactory():
    return defaultdict(_individualfactory)


def _subtestfactory():
    return defaultdict(_strategyfactory)


def walk_and_make_average_for_all_tests(path):
    test_dict = defaultdict(_subtestfactory)

    for r, d, fs in os.walk(path):
        if "_sensors" in r: continue
        if os.path.basename(r) in ["best_individual", "general_population", "mix_in", "adaptation"]:
            strategy = os.path.basename(r)
            parent_folder = os.path.dirname(r)
            sub_test_name = os.path.basename(parent_folder)
            test_name = os.path.basename(os.path.dirname(parent_folder))[15:]
            for file_path in fs:
                print(file_path)
                individual_name = os.path.splitext(file_path)[0]
                with open(os.path.join(r, file_path), "r") as f:
                    my_dict = json.load(f)
                test_dict[test_name][sub_test_name][strategy][individual_name].append(my_dict)

    print("finished this")

    def walk(node, outpath):
        for key, item in node.items():
            if type(item) is not list:
                walk(item, os.path.join(outpath, key))
            else:
                a = average_dicts(item)
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                with open(os.path.join(outpath, key + ".json"), "w") as f:
                    json.dump(a, f)

    walk(test_dict, os.path.join(VAGESHAR_ROOT, "average_statistics"))


def convert_key_to_boolean_array(string):
    order = [["lb"], ["lt", "at"], ["rt", "ut"], ["lw", "aw"], ["rw", "uw"]]

    boolean_array = []
    for l in order:
        for substring in l:
            if substring in string:
                boolean_array.append(True)
                break
        else:
            boolean_array.append(False)

    return boolean_array


def convert_boolean_array_to_string_array(array, true_string="x", false_string=" "):
    return [true_string if e else false_string for e in array]


def output_tabular_for_statistic_folder(path):
    """

    :param path: Path to a test folder
    :return:
    """
    order = ("general_population", "mix_in", "adaptation", "best_individual")

    score_dict = dict()

    for sub_test in os.listdir(path):
        sub_test_scores = []
        for approach in order:
            with open(os.path.join(path, sub_test, approach, "overall.json"), "r") as f:
                accuracy = json.load(f)["accuracy"]
            accuracy_as_percentage_string = str(round(accuracy * 100, 2))
            sub_test_scores.append(accuracy_as_percentage_string)

        score_dict[sub_test] = sub_test_scores

    rows = []
    for sub_test in score_dict:
        rows.append([sub_test] + score_dict[sub_test])

    rows.sort(key=lambda x: x[1], reverse=True)

    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}")
    print("\t\\hline")
    # print("\t", " & ".join(["lb", "lt", "rt", "lw", "rw", "LOSO", "Mix-in", "MP", "SP"]), "\\\\")
    print("\t", " & ".join(["lb", "at", "ut", "aw", "uw", "LOSO", "Mix-in", "MP", "SP"]), "\\\\")
    print("\t\\hline")
    for r in rows:
        xs = convert_boolean_array_to_string_array(convert_key_to_boolean_array(r[0]))
        print("\t", " & ".join(xs + r[1:]), "\\\\")
    print("\t\\hline")
    print("\\end{tabular}")


def make_tabular_with_difference(path, path2):
    """

    :param path: Path to a test folder
    :return:
    """
    score_dict = get_score_dict(path)
    score_dict2 = get_score_dict(path2)

    rows = make_rows_from_score_dict(score_dict)
    rows2 = make_rows_from_score_dict(score_dict2)

    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}")
    print("\t\\hline")
    # print("\t", " & ".join(["lb", "lt", "rt", "lw", "rw", "LOSO", "Mix-in", "MP", "SP"]), "\\\\")
    print("\t", " & ".join(["lb", "at", "ut", "aw", "uw", "LOSO", "Mix-in", "MP", "SP"]), "\\\\")
    print("\t\\hline")
    for r1, r2 in zip(rows, rows2):
        xs = convert_boolean_array_to_string_array(convert_key_to_boolean_array(r1[0]))

        print("\t", " & ".join(xs + row_difference(r1[1:], r2[1:])), "\\\\")
    print("\t\\hline")
    print("\\end{tabular}")


def row_difference(array1, array2):
    return ["%0.2f (%0.3f)" % (float(a), round(float(a) - float(b), 2)) for a, b in zip(array1, array2)]


def make_rows_from_score_dict(score_dict):
    rows = []
    for sub_test in score_dict:
        rows.append([sub_test] + score_dict[sub_test])
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def get_score_dict(path):
    order = ("general_population", "mix_in", "adaptation", "best_individual")
    score_dict = dict()
    for sub_test in os.listdir(path):
        sub_test_scores = []
        for approach in order:
            with open(os.path.join(path, sub_test, approach, "overall.json"), "r") as f:
                accuracy = json.load(f)["accuracy"]
            accuracy_as_percentage_string = str(round(accuracy * 100, 2))
            sub_test_scores.append(accuracy_as_percentage_string)

        score_dict[sub_test] = sub_test_scores
    return score_dict


# USed primarilyfor plotting the activity distribution from a labeled set of activities
def plot_data_distribution(labels):
    data_frame = pd.DataFrame(labels["006"])

    # fig = plt.figure
    # ploting = data_frame.plot.hist()
    # fig = ploting.get_figure()
    # fig.savefig("histogram_006.png")

    plt.hist(labels["006"])
    plt.title("Gaussian Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    fig = plt.gcf()
    fig.savefig("histogram_006.png")


    # activity_counts = {}
    #  for activity in labels:
    #      if activity in activity_counts.keys():
    #          activity_counts[activity] += 1
    #      else:
    #          activity_counts[activity] = 1
    #  return activity_counts


if __name__ == "__main__":
    walk_and_make_average_for_all_tests(os.path.join(VAGESHAR_ROOT, "loso_statistics"))

    """
    for i in range(5):
        make_tabular_with_difference(
            os.path.join(VAGESHAR_ROOT, "average_statistics", str(i + 1) + "_sensors_affected"),
            os.path.join(VAGESHAR_ROOT, "average_statistics", str(i + 1) + "_sensors_no_stairs")
        )
    """
