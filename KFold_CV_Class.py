#%%
import os
import random as random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def MOFIT_CLASS(y_test, Prediction):
    """The MOFIT function takes in the Ytest values and predictions from a classifictation model
    It then calculates accuracy Precision Recall and F1 Score and returns them all in a dictionary for
    each level of the response variable
    """
    d = {}
    y_test.reset_index(drop=True, inplace=True)
    Prediction.reset_index(drop=True, inplace=True)

    for val in np.unique(y_test):
        true_pos = sum(((y_test == val) & (Prediction == val)))
        false_pos = sum(((y_test != val) & (Prediction == val)))
        false_neg = sum(((y_test == val) & (Prediction != val)))
        mc_rate = sum(Prediction != y_test) / len(y_test)

        if (true_pos + false_pos) > 0:
            Precision = true_pos / (true_pos + false_pos)
        else:
            Precision = float("NaN")
        if (true_pos + false_neg) > 0:
            Recall = true_pos / (true_pos + false_neg)
        else:
            Recall = float("NaN")
        if np.isnan(Precision) or np.isnan(Recall) or (Precision + Recall) <= 0:
            F1 = float("Nan")
        else:
            try:
                F1 = 2 * float((Precision * Recall)) / float((Precision + Recall))
            except:
                print(f"Precision : {Precision} Recall: {Recall}")
        mofs = {
            "MisClass": mc_rate,
            "Precision": Precision,
            "Recall": Recall,
            "F1": F1,
            "Confusion Matrix": confusion_matrix(y_test, Prediction),
        }

        d[str(val)] = mofs
    return d


# Build Rep Function
def rep(pattern, n):
    elist = []
    if isinstance(pattern, float) or isinstance(pattern, int):
        elist = [pattern for x in range(n)]
        return elist
    else:
        if type(pattern) == str:
            elist = [pattern for x in range(n)]
            return elist
        else:
            elist = [pattern for x in range(n)]
            elist = sum(elist, [])
            return elist


class Kfold_CV_Classifier:
    """
    The Kfold Classifer Class takes in an k value a sk-learn classification model a dataframe a response variable
    and a string which is the name of the model being used

    The Run CV method Runs the Kfold cross validation and returns a dictionary for each level of the response and the
    potential measures of fit

    the Plot Metric method takes in which level of the response and which measure of fit to analyze and plots the m
    measure of fit over the k values

    The mean mofit method returns the mean measure of fit and the standard deviation of the metric chosen over the k
    folds

    the sum confusion matrix method sums the confusion matrix over the k folds
    """

    def __init__(self, k, model, data, response, name):
        self.k = k
        self.model = model
        self.data = data
        self.response = response
        self.met_list = []
        self.name = name

    def run_CV(self, seed=-28):
        nums = rep(list(range(self.k)), int(np.ceil(self.data.shape[0] / self.k)))
        if seed > 0:
            random.seed(seed)

        self.data["index"] = random.sample(nums, self.data.shape[0])

        for i in range(self.k):
            X_test = self.data[self.data["index"] == i].drop(
                [self.response, "index"], axis=1
            )
            X_train = self.data[self.data["index"] != i].drop(
                [self.response, "index"], axis=1
            )

            Y_test = self.data[self.data["index"] == i][self.response]
            Y_train = self.data[self.data["index"] != i][self.response]

            self.model.fit(X_train, Y_train)
            prediction = pd.Series(self.model.predict(X_test))

        self.met_list.append(MOFIT_CLASS(Y_test, prediction))

    def plot_metric(self, Category, Metric):
        metric_list = []
        ks = []
        for val in range(self.k):
            if not np.isnan(self.met_list[val][Category][Metric]):
                metric_list.append(self.met_list[val][Category][Metric])
                ks.append(val)
            else:
                pass

            plt.figure(figsize=(10, 6))
            plt.plot(
                ks,
                metric_list,
                color="green",
                linestyle="dashed",
                marker="o",
                markerfacecolor="red",
                markersize=10,
            )
            plt.title(f"{Metric} vs. K Value")
            plt.xlabel("K")
            plt.ylabel(Metric)
            plt.title(self.name)
            plt.show()


def mean_mofit(self, Category, Metric):
    mean_list = []
    for val in range(self.k):
        if not np.isnan(self.met_list[val][Category][Metric]):
            mean_list.append(self.met_list[val][Category][Metric])
        else:
            pass
        mean_val = np.mean(mean_list)
        std_val = np.std(mean_list)
        print(
            f"Category: {Category}\nMetric: {Metric}\nValue: {mean_val}\nStd:{std_val}"
        )


def Sum_Confusion(self, Category):
    matrix_list = []
    for val in range(self.k):
        matrix_list.append(self.met_list[val][Category]["Confusion Matrix"])
