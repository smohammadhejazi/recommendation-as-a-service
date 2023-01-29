"""
This file contains ColdStart class/module.
"""


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from kmodes.kmodes import KModes
from collections import defaultdict
from .module_base import ModuleBase
from ..utils import silhouette_score
from ..utils import matching_dissimilarity
from ..surprise import WeightedSlopeOne
from ..surprise import Dataset
from ..surprise import Reader
from ..surprise import Prediction


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
        self.user_cluster = None
        self.clusters_score = None
        self.clusters_cost = None
        self.clusters_fit_time = None
        self.k_start = options.get("k_start", None)
        self.k_end = options.get("k_end", None)
        self.sorted_predictions = None
        self.optimal_k = options.get("k", None)
        self.build_tables = options.get("build_tables", True)
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
        virtual_rating_count = pd.DataFrame(columns=["user_id", "item_id", "ratingmean", "ratingcount"])

        for i in range(k):
            # users in cluster i
            users = self.user_info[self.user_info["cluster"] == i]

            # ratings and counts of these users
            users_rating = self.user_rating[self.user_rating["user_id"].isin(users["user_id"])]
            res = users_rating.groupby(["item_id"], as_index=False).agg({"rating": ["mean", "count"]})
            res.columns = list(map(''.join, res.columns.values))
            # res = res.rename({"ratingmean": "rating_mean", "ratingcount": "rating_count"}, axis=1)

            # add virtual user_id to dataframes
            res.insert(0, 'user_id', i)

            # TODO maybe we can optimise this part
            # add them to virtual_rating_count
            virtual_rating_count = pd.concat([virtual_rating_count, res])

        return virtual_rating_count

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
        ax1.set_ylabel("Cost", color=color)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.plot(k_clusters, self.clusters_cost, color=color, marker=marker, linestyle=line)

        ax2 = axes[1]
        ax2.set_ylabel("Time (s)", color=color)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(k_clusters, self.clusters_fit_time, color=color, marker=marker, linestyle=line)

        ax3 = axes[2]
        ax3.set_ylabel("Silhouette Score")
        ax3.set_xlabel("No. of clusters")
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
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

    def fit(self, trainset=None):
        """
        Fits the class and prepares the required things for recommend function.
        First clusters users using KModes, then fits the weighted slope one algorithm for later use.
        """

        if self.k_start is None:
            self.k_start = 2
        if self.k_end is None:
            self.k_end = int(self.user_info.shape[0] / 2)

        if not self.manual_cluster and self.k_start >= self.k_end:
            raise ValueError("Error: k_start should be smaller than k_end")
        if self.verbose:
            print("Fitting the algorithm...")

        # get optimal number of clusters
        if not self.manual_cluster:
            self.optimal_k = self.set_optimal_k_clusters(self.k_start, self.k_end)

        # cluster with optimal k
        if self.verbose:
            print("Clustering with k={}...".format(self.optimal_k))

        kmode = KModes(n_clusters=self.optimal_k, init="random", n_init=5, n_jobs=-1, verbose=0)
        cluster_lables = kmode.fit_predict(self.user_info)
        self.user_info["cluster"] = cluster_lables.tolist()
        self.user_cluster = {}
        for idx, i in np.ndenumerate(cluster_lables):
            self.user_cluster[idx[0] + 1] = i
        virtual_rating_count = self.generate_virtual_rating_count(self.optimal_k)

        # virtual ratings ready
        rating_reader = Reader(rating_scale=(1, 5))
        count_reader = Reader(rating_scale=(virtual_rating_count["ratingcount"].min(), virtual_rating_count["ratingcount"].max()))
        rating_data = Dataset.load_from_df(virtual_rating_count[["user_id", "item_id", "ratingmean"]], rating_reader)
        count_data = Dataset.load_from_df(virtual_rating_count[["user_id", "item_id", "ratingcount"]], count_reader)
        rating_trainset = rating_data.build_full_trainset()
        count_trainset = count_data.build_full_trainset()

        self.algo = WeightedSlopeOne(count_trainset)
        self.algo.fit(rating_trainset)

        if not self.build_tables:
            return

        if self.verbose:
            print("Clustering done.".format(self.optimal_k))
            print("Building tables...")

        # data = Dataset.load_from_df(self.user_rating[["user_id", "item_id", "rating"]], rating_reader)
        # train_set = data.build_full_trainset()
        test_set = rating_trainset.build_anti_testset()
        predictions = self.test(test_set)

        self.set_top_n(predictions)
        self.is_fit = True
        if self.verbose:
            print("Tables are built.")
            print("Fitting is done.")

    def predict_rating(self, user_id, item_id, r_ui=None, clip=True, verbose=False):
        """
        Predict the rating of an item
        :param user_id: specified user
        :param item_id: specified item
        :param r_ui: the true rating
        :param clip: whether to clip the estimation into the rating scale.
        :param verbose: whether to print details of the prediction
        :return: prediction object
        """

        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        virtual_user = self.user_cluster[user_id]
        p_object = self.algo.predict(virtual_user, item_id, r_ui, clip, verbose)
        p_object = Prediction(virtual_user, p_object.iid, p_object.r_ui, p_object.est, p_object.details)
        return p_object

    def test(self, testset, verbose=False):
        """
        Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.
        :param testset: a test set, as returned by a cross-validation or by the build_testset() method
        :param verbose: whether to print details for each predictions
        :return: list of predictions.
        """

        # The ratings are translated back to their original scale.
        predictions = [
            self.algo.predict(uid, iid, r_ui_trans, verbose=verbose)
            for (uid, iid, r_ui_trans) in testset
        ]

        return predictions

    def recommend(self, user_id, n):
        """
        Recommends n items based on user history
        :param user_id: specified user
        :param n: number of items to be recommended
        :return: list of n items
        """

        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        items = self.sorted_predictions[self.user_cluster[user_id]]
        recommendation_table = []
        for i in range(n):
            entry = (items[i][0], self.id_to_name(items[i][0]), items[i][1])
            recommendation_table.append(entry)
        return recommendation_table
