import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from collections import defaultdict
from .module_base import ModuleBase
from ..utils import silhouette_score
from ..utils import matching_dissimilarity
from ..surprise import WeightedSlopeOne
from ..surprise import Dataset
from ..surprise import Reader


class UserSpecific(ModuleBase):
    def __init__(self, user_rating, user_info, item_info, options=None):
        if options is None:
            options = {}
        ModuleBase.__init__(self, user_rating, options)
        self.user_info = user_info
        self.item_info = item_info
        self.algo = None
        self.clusters_score = None
        self.clusters_cost = None
        self.clusters_fit_time = None
        self.k_start = None
        self.k_end = None
        self.optimal_k = options.get("k", None)
        self.manual_cluster = False if options.get("k", None) is None else True
        self.top_n = options.get("top_n", 10)

    def name_to_id(self, name):
        # csv reads ids as integer but we need string in inner_id
        movie = self.item_info[self.item_info["movie_title"] == name]
        return movie["movie_id"].item()

    def id_to_name(self, iid):
        # after converting to string, here we convert back
        movie = self.item_info[self.item_info["movie_id"] == int(iid)]
        return movie["movie_title"].item()

    def set_optimal_k_clusters(self, k_start, k_end):
        if self.verbose:
            print("Finding optimal cluster...")
        score = []
        cost = []
        fit_time = []
        for k in range(k_start, k_end):
            kmode = KModes(n_clusters=k, init="random", n_init=5, n_jobs=-1, verbose=0)
            start = time.perf_counter()
            cluster_labels = kmode.fit_predict(self.user_info)
            end = time.perf_counter()
            s_score = silhouette_score(self.user_info[["age", "gender", "occupation"]], cluster_labels,
                                       metric=matching_dissimilarity)
            score.append(s_score)
            cost.append(kmode.cost_)
            fit_time.append(end - start)

        self.clusters_score = score
        self.clusters_cost = cost
        self.clusters_fit_time = fit_time

        optimal_cluster = np.argmax(np.array(score)) + k_start

        if self.verbose:
            print("Optimal cluster k={} found.".format(optimal_cluster))

        return optimal_cluster

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

    def draw_clusters_graph(self, path=None):
        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        if self.manual_cluster:
            print("Error: Can't draw graph, finding optimal cluster was done manually.")
            return
        # show plots
        k_clusters = range(self.k_start, self.k_end)
        color = "k"
        marker = "."
        line = "-"
        fig, axes = plt.subplots(3, 1)

        ax1 = axes[0]
        # ax1.set_xlabel("No. of clusters")
        ax1.set_ylabel("distortion", color=color)
        # ax1.set_xticks(k_clusters)
        ax1.plot(k_clusters, self.clusters_cost, color=color, marker=marker, linestyle=line)

        ax2 = axes[1]
        # ax2.set_xlabel("No. of clusters")
        ax2.set_ylabel("time (s)", color=color)
        # ax2.set_xticks(k_clusters)
        ax2.plot(k_clusters, self.clusters_fit_time, color=color, marker=marker, linestyle=line)

        ax3 = axes[2]
        ax3.set_xlabel("No. of clusters")
        ax3.set_ylabel("Silhouette score")
        # ax3.set_xticks(k_clusters)
        ax3.plot(k_clusters, self.clusters_score, color=color, marker=marker, linestyle=line)

        fig.tight_layout()
        if path is not None:
            plt.savefig(path)
        plt.show()

    def set_top_n(self, predictions, n=10):
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        self.top_n = top_n

    def fit(self, k_start=1, k_end=2):
        if k_start >= k_end:
            raise ValueError("Error: k_start should be smaller than k_end")
        if self.verbose:
            print("Fitting the algorithm...")

        self.k_start = k_start
        self.k_end = k_end

        # get optimal number of clusters
        if self.optimal_k is None:
            self.optimal_k = self.set_optimal_k_clusters(k_start, k_end)

        # cluster with optimal k
        if self.verbose:
            print("Clustering with k={}...".format(self.optimal_k))

        kmode = KModes(n_clusters=self.optimal_k, init="random", n_init=5, n_jobs=-1, verbose=0)
        cluster_labels = kmode.fit_predict(self.user_info)

        if self.verbose:
            print("Clustering done.".format(self.optimal_k))
            print("Building tables...")

        self.user_info['cluster'] = cluster_labels.tolist()
        virtual_rating, virtual_count = self.generate_virtual_rating_count(self.optimal_k)

        # virtual ratings ready

        mean_reader = Reader(rating_scale=(1, 5))
        count_reader = Reader(rating_scale=(virtual_count["rating_count"].min(), virtual_count["rating_count"].max()))
        mean_data = Dataset.load_from_df(virtual_rating, mean_reader)
        count_data = Dataset.load_from_df(virtual_count, count_reader)

        mean_train_set = mean_data.build_full_trainset()
        count_train_set = count_data.build_full_trainset()

        mean_train_set.build_anti_testset()

        self.algo = WeightedSlopeOne(count_train_set)
        self.algo.fit(mean_train_set)

        data = Dataset.load_from_df(self.user_rating[["user_id", "item_id", "rating"]], mean_reader)
        train_set = data.build_full_trainset()
        test_set = train_set.build_anti_testset()
        predictions = self.algo.test(test_set)

        self.set_top_n(predictions, n=self.top_n)
        self.is_fit = True
        if self.verbose:
            print("Tables are built.")
            print("Fitting is done.")

    def predict_rating(self, user_id, item_id):
        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        prediction_rating_object = self.algo.predict(user_id, item_id)
        return prediction_rating_object

    def recommend(self, user_id, n=10):
        items = self.top_n[user_id]
        recommendation_table = []
        for i in range(n):
            entry = (items[i][0], self.id_to_name(items[i][0]), items[i][1])
            recommendation_table.append(entry)
        return recommendation_table
