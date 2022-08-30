import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from kmodes.kmodes import KModes
from utils import silhouette_score, matching_dissimilarity
from surprise import SlopeOne
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


NUMBER_OF_USERS = 943
NUMBER_OF_MOVIES = 1682


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

    def set_optimal_k_clusters(self, k_start, k_end):
        score = []
        cost = []
        fit_time = []
        for k in range(k_start, k_end):
            print("K clusters = " + str(k))
            kmode = KModes(n_clusters=k, init="random", n_init=5, n_jobs=-1, verbose=0)
            start = time.perf_counter()
            cluster_labels = kmode.fit_predict(self.user_info)
            end = time.perf_counter()
            s_score = silhouette_score(self.user_info[["age", "gender", "occupation"]], cluster_labels, metric=matching_dissimilarity)
            score.append(s_score)
            cost.append(kmode.cost_)
            fit_time.append(end - start)

        return np.argmax(np.array(score)) + k_start, score, cost, fit_time

    def generate_virtual_ratings(self, k):
        virtual_rating = pd.DataFrame(columns=["user_id", "item_id", "rating"])

        for i in range(k):
            # users in cluster i
            users = self.user_info[self.user_info["cluster"] == i]

            # ratings of these users
            users_rating = self.user_ratings[self.user_ratings["user_id"].isin(users["user_id"])]

            # mean of item ratings in within this cluster
            users_mean_rating = users_rating.groupby(["item_id"], as_index=False)["rating"].mean()

            # add virtual user_id to dataframe
            users_mean_rating.insert(0, 'user_id', i)

            # TODO maybe we can optimise this part
            # add them to virtual_rating
            virtual_rating = pd.concat([virtual_rating, users_mean_rating])

        return virtual_rating


if __name__ == "__main__":
    base_dataset_dir = "../dataset/ml-100k/"
    recom_module = UserRecommendation(user_info_path=base_dataset_dir + "u.user",
                                      user_ratings_path=base_dataset_dir + "u.data")
    recom_module.read_csv_data(["user_id", "age", "gender", "occupation", "zip_code"],
                               ["user_id", "item_id", "rating", "timestamp"], info_sep="|", ratings_sep="\t")

    # get optimal number of clusters
    k_start = 26
    k_end = 27
    optimal_k, score, cost, fit_time = recom_module.set_optimal_k_clusters(k_start, k_end)
    print("Optimal K = " + str(optimal_k))

    # cluster with optimal k
    kmode = KModes(n_clusters=25, init="random", n_init=5, n_jobs=-1, verbose=0)
    cluster_labels = kmode.fit_predict(recom_module.user_info)
    recom_module.user_info['cluster'] = cluster_labels.tolist()
    virtual_rating = recom_module.generate_virtual_ratings(25)
    # print(virtual_rating)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(virtual_rating, reader)
    algo = SlopeOne()
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    exit(0)
    # show plots
    k_clusters = range(k_start, k_end)
    color = "k"
    marker = "."
    line = "-"
    fig, axes = plt.subplots(3, 1)

    ax1 = axes[0]
    # ax1.set_xlabel("No. of clusters")
    ax1.set_ylabel("distortion", color=color)
    # ax1.set_xticks(k_clusters)
    ax1.plot(k_clusters, cost, color=color, marker=marker, linestyle=line)

    ax2 = axes[1]
    # ax2.set_xlabel("No. of clusters")
    ax2.set_ylabel("time (s)", color=color)
    # ax2.set_xticks(k_clusters)
    ax2.plot(k_clusters, fit_time, color=color, marker=marker, linestyle=line)

    ax3 = axes[2]
    ax3.set_xlabel("No. of clusters")
    ax3.set_ylabel("Silhouette score")
    # ax3.set_xticks(k_clusters)
    ax3.plot(k_clusters, score, color=color, marker=marker, linestyle=line)

    fig.tight_layout()
    plt.show()
