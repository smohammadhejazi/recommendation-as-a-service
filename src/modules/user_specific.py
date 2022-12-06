import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from .module_base import ModuleBase
from kmodes.kmodes import KModes
from src.utils import silhouette_score, matching_dissimilarity
from surprise import WeightedSlopeOne
from surprise import Dataset
from surprise import Reader


class UserSpecific(ModuleBase):
    def __init__(self, user_rating, user_info, item_info, options=None):
        if options is None:
            options = {}
        ModuleBase.__init__(self, user_rating, options)
        self.user_info = user_info
        self.item_info = item_info
        self.algo = None

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
            s_score = silhouette_score(self.user_info[["age", "gender", "occupation"]], cluster_labels,
                                       metric=matching_dissimilarity)
            score.append(s_score)
            cost.append(kmode.cost_)
            fit_time.append(end - start)

        return np.argmax(np.array(score)) + k_start, score, cost, fit_time

    def generate_virtual_rating_count(self, k):
        virtual_rating = pd.DataFrame(columns=["user_id", "item_id", "rating_mean"])
        virtual_count = pd.DataFrame(columns=["user_id", "item_id", "rating_count"])

        for i in range(k):
            # users in cluster i
            users = self.user_info[self.user_info["cluster"] == i]

            # ratings and counts of these users
            users_rating = self.user_rating[self.user_rating["user_id"].isin(users["user_id"])]
            users_items = users_rating.groupby(["item_id"], as_index=False)
            res = users_items.agg({"rating": ["mean", "count"]})
            res.columns = list(map(''.join, res.columns.values))
            res = res.rename({"ratingmean": "rating_mean", "ratingcount": "rating_count"}, axis=1)

            # mean and count of item ratings in within this cluster
            item_mean = res[["item_id", "rating_mean"]]
            item_count = res[["item_id", "rating_count"]]

            # add virtual user_id to dataframes
            item_mean.insert(0, 'user_id', i)
            item_count.insert(0, 'user_id', i)

            # TODO maybe we can optimise this part
            # add them to virtual_rating and virtual count
            virtual_rating = pd.concat([virtual_rating, item_mean])
            virtual_count = pd.concat([virtual_count, item_count])

        return virtual_rating, virtual_count

    def fit(self):
        # get optimal number of clusters
        k_start = 26
        k_end = 27
        optimal_k, score, cost, fit_time = self.set_optimal_k_clusters(k_start, k_end)
        print("Optimal K = " + str(optimal_k))

        # cluster with optimal k
        kmode = KModes(n_clusters=25, init="random", n_init=5, n_jobs=-1, verbose=0)
        cluster_labels = kmode.fit_predict(self.user_info)
        self.user_info['cluster'] = cluster_labels.tolist()
        virtual_rating, virtual_count = self.generate_virtual_rating_count(25)

        # virtual ratings ready

        mean_reader = Reader(rating_scale=(1, 5))
        count_reader = Reader(rating_scale=(virtual_count["rating_count"].min(), virtual_count["rating_count"].max()))
        mean_data = Dataset.load_from_df(virtual_rating, mean_reader)
        count_data = Dataset.load_from_df(virtual_count, count_reader)

        mean_train_set = mean_data.build_full_trainset()
        count_train_set = count_data.build_full_trainset()

        self.algo = WeightedSlopeOne(count_train_set)
        self.algo.fit(mean_train_set)

    def recommend(self, user_id, item_id):
        pred = self.algo.predict(user_id, item_id)
        return pred

        # exit(0)
        # # show plots
        # k_clusters = range(k_start, k_end)
        # color = "k"
        # marker = "."
        # line = "-"
        # fig, axes = plt.subplots(3, 1)
        #
        # ax1 = axes[0]
        # # ax1.set_xlabel("No. of clusters")
        # ax1.set_ylabel("distortion", color=color)
        # # ax1.set_xticks(k_clusters)
        # ax1.plot(k_clusters, cost, color=color, marker=marker, linestyle=line)
        #
        # ax2 = axes[1]
        # # ax2.set_xlabel("No. of clusters")
        # ax2.set_ylabel("time (s)", color=color)
        # # ax2.set_xticks(k_clusters)
        # ax2.plot(k_clusters, fit_time, color=color, marker=marker, linestyle=line)
        #
        # ax3 = axes[2]
        # ax3.set_xlabel("No. of clusters")
        # ax3.set_ylabel("Silhouette score")
        # # ax3.set_xticks(k_clusters)
        # ax3.plot(k_clusters, score, color=color, marker=marker, linestyle=line)
        #
        # fig.tight_layout()
        # plt.show()
