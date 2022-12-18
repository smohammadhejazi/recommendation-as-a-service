"""
This file contains ColdStart class/module.
"""


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
    """
    This class recommends items based on user's previous ratings.
    First users are clustered using KMods, then using weighted slope one algorithm
    unrated items are predicted and sorted for final recommendation.
    """
    def __init__(self, user_rating, user_info, item_info, options=None):
        """
        :param user_rating: Users' ratings csv
        :param user_info: Users' info csv
        :param item_info: Items' ratings csv
        :param options: Options dictionary
        """

        if options is None:
            options = {}
        ModuleBase.__init__(self, user_rating=user_rating, user_info=user_info, item_info=item_info, options=options)
        self.algo = None
        self.clusters_score = None
        self.clusters_cost = None
        self.clusters_fit_time = None
        self.k_start = None
        self.k_end = None
        self.sorted_predictions = None
        self.optimal_k = options.get("k", None)
        self.manual_cluster = False if self.optimal_k is None else True

    def set_optimal_k_clusters(self, k_start, k_end):
        """
        Find optimal number of clusters between k_start and k_end
        :param k_start: start number to find optimal number of cluster
        :param k_end: end number to find optimal number of cluster
        :return: optimal number of clusters
        """
        if self.verbose:
            print("Finding optimal cluster between {} and {}".format(k_start, k_end))
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
        """
        This function clusters users into k different clusters. The return panda frames are used later
        in weighted slope one algorithm.
        :param k: number of clusters
        :return: two panda frames:
            virtual_rating: mean of item ratings of all the users in the cluster
            virtual_count: number of ratings to items of all users in the cluster
        """
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
        """
        Plots graph of fit_time, distortion and Silhouette score for different number of clusters.
        This function can be used when automatic clustering is on.
        :param path: path of the plot to be saved
        """
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

    def set_top_n(self, predictions):
        """
        Builds the prediction table of unrated items for all users
        :param predictions: predictions of unrated items provided by algo.test
        :return: prediction table
        """
        # First map the predictions to each user.
        sorted_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            sorted_predictions[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in sorted_predictions.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            sorted_predictions[uid] = user_ratings

        self.sorted_predictions = sorted_predictions

    def fit(self, k_start=None, k_end=None):
        """
        Fits the class and prepares the required things for recommend function.
        First clusters users using KModes, then fits the weighted slope one algorithm for later use.
        """

        if k_start is None:
            k_start = 2
        if k_end is None:
            k_end = int(self.user_info.shape[0] / 2)

        if not self.manual_cluster and k_start >= k_end:
            raise ValueError("Error: k_start should be smaller than k_end")
        if self.verbose:
            print("Fitting the algorithm...")

        self.k_start = k_start
        self.k_end = k_end

        # get optimal number of clusters
        if not self.manual_cluster:
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

        self.set_top_n(predictions)
        self.is_fit = True
        if self.verbose:
            print("Tables are built.")
            print("Fitting is done.")

    def predict_rating(self, user_id, item_id):
        """
        Predict the rating of an item
        :param user_id: specified user
        :param item_id: specified item
        :return: prediction object
        """

        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        prediction_rating_object = self.algo.predict(user_id, item_id)
        return prediction_rating_object

    def recommend(self, user_id, n):
        """
        Recommends n items based on user history
        :param user_id: specified user
        :param n: number of items to be recommended
        :return: list of n items
        """

        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        items = self.sorted_predictions[user_id]
        recommendation_table = []
        for i in range(n):
            entry = (items[i][0], self.id_to_name(items[i][0]), items[i][1])
            recommendation_table.append(entry)
        return recommendation_table
