import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn import metrics
from kmodes.kmodes import KModes


class UserRecommendation:
    def __init__(self, user_info_path=None, user_ratings_path=None):
        self.user_info_path = user_info_path
        self.user_ratings_path = user_ratings_path
        self.user_info = None
        self.user_ratings = None
        self.number_of_clusters = 0

    def read_csv_data(self, info_columns, ratings_columns, info_sep="|", ratings_sep="\t"):
        self.user_info = pd.read_csv(self.user_info_path, sep=info_sep, names=info_columns)
        self.user_ratings = pd.read_csv(self.user_ratings_path, sep=ratings_sep, names=ratings_columns)

    def matching_dissimilarity(self, a, b):
        return np.sum(a != b)

    # def silhouette_analysis(self, df):
    #     return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))

    def set_optimal_k_clusters(self):
        k_start = 25
        k_end = 35
        score = []
        cost = []
        fit_time = []
        for k in range(k_start, k_end):
            print("K clusters = " + str(k))
            kmode = KModes(n_clusters=k, init="random", n_init=5, n_jobs=-1, verbose=0)

            start = time.perf_counter()
            cluster_labels = kmode.fit_predict(self.user_info)
            end = time.perf_counter()
            s_score = metrics.silhouette_score(self.user_info, cluster_labels, metric=self.matching_dissimilarity)
            print(s_score)
            score.append(s_score)
            cost.append(kmode.cost_)
            fit_time.append(end - start)

        np_cost = np.array([cost])
        np_fit_time = np.array([fit_time])
        cost_normalized = (np_cost - np.amin(np_cost)) / (np.amax(np_cost) - np.amin(np_cost))
        fit_time_normalized = (np_fit_time - np.amin(np_fit_time)) / (np.amax(np_fit_time) - np.amin(np_fit_time))

        for i in range(cost_normalized.size):
            if cost_normalized[0][i] <= fit_time_normalized[0][i]:
                return i + 1, cost_normalized, fit_time_normalized, np_cost, np_fit_time
        return -1


if __name__ == "__main__":
    base_dataset_dir = "../dataset/ml-100k/"
    recom_module = UserRecommendation(user_info_path=base_dataset_dir + "u.user",
                                      user_ratings_path=base_dataset_dir + "u.data")
    recom_module.read_csv_data(['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                               ['user_id', 'item_id', 'rating', 'timestamp'], info_sep="|", ratings_sep="\t")

    # get optimal number of clusters
    optimal_n_cluster, cost_normalized, fit_time_normalized, np_cost, np_fit_time = recom_module.set_optimal_k_clusters()

    print(optimal_n_cluster)
    k_clusters = range(1, int(math.sqrt(recom_module.user_ratings.shape[0])) + 1)

    # show plots
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('No. of clusters')
    ax1.set_ylabel('distortion', color=color)
    ax1.plot(k_clusters, cost_normalized[0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('time (s)', color=color)
    ax2.plot(k_clusters, fit_time_normalized[0], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()