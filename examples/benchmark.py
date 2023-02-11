import datetime
import random
import time

import numpy as np

from kabirrec import RecommendationService
from kabirrec.surprise import (
    Dataset,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    KNNBaseline,
    SlopeOne,
    Reader
)
from kabirrec.surprise.model_selection import cross_validate, KFold
from tabulate import tabulate

# The algorithms to cross-validate
algos = (
    SlopeOne(),
    KNNBasic(),
    KNNWithMeans(),
    KNNWithZScore(),
    KNNBaseline(),
)

# ugly dict to map algo names and datasets to their markdown links in the table
stable = "https://surprise.readthedocs.io/en/stable/"
LINK = {
    "SlopeOne": "[{}]({})".format(
        "Slope One",
        stable + "slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne",
    ),
    "WeightedSlopeOne": "[{}]({})".format(
        " Clustering With Weighted Slope One",
        "https://github.com/smohammadhejazi/recommendation-as-a-service",
    ),
    "KNNBasic": "[{}]({})".format(
        "KNN Basic",
        stable + "knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic",
    ),
    "KNNWithMeans": "[{}]({})".format(
        "KNN With Means",
        stable + "knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans",
    ),
    "KNNWithZScore": "[{}]({})".format(
        "KNN With Z-Score",
        stable + "knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithZScore",
    ),
    "KNNBaseline": "[{}]({})".format(
        "KNN Baseline",
        stable + "knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline",
    ),
    "ml-100k": "[{}]({})".format(
        "Movielens 100k", "https://grouplens.org/datasets/movielens/100k"
    ),
    "ml-1m": "[{}]({})".format(
        "Movielens 1M", "https://grouplens.org/datasets/movielens/1m"
    ),
}


# set RNG
np.random.seed(0)
random.seed(0)

recommendation_service = RecommendationService()
# # ml-100k
# data_set = "ml-100k"
# recommendation_service.read_csv_data(
#     user_info_path="../dataset/{}/u.user".format(data_set),
#     user_ratings_path="../dataset/{}/u.data".format(data_set),
#     item_info_path="../dataset/{}/u.item".format(data_set),
#     user_info_columns=["user_id", "age", "gender", "occupation", "zip_code"],
#     user_ratings_columns=["user_id", "item_id", "rating", "timestamp"],
#     item_info_columns=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown",
#                   "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
#                   "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi",
#                   "thriller", "war", "western"],
#     user_info_sep="|", user_ratings_sep="\t", item_sep="|"
# )

# ml-1m
data_set = "ml-1m"
recommendation_service.read_csv_data(
    user_info_path="../dataset/{}/users.dat".format(data_set),
    user_ratings_path="../dataset/{}/ratings.dat".format(data_set),
    item_info_path="../dataset/{}/movies.dat".format(data_set),
    user_info_columns=["user_id", "gender", "age", "occupation", "zip_code"],
    user_ratings_columns=["user_id", "item_id", "rating", "timestamp"],
    item_info_columns=["movie_id", "movie_title", "genre"],
    user_info_sep="::", user_ratings_sep="::", item_sep="::"
)

user_specific = recommendation_service.user_specific_module(options={"verbose": True,
                                                                     "k": 30,
                                                                     "k_start": 20,
                                                                     "k_end": 30,
                                                                     "build_tables": False})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_specific.user_rating[["user_id", "item_id", "rating"]], reader)
kf = KFold(random_state=0)  # folds will be the same for all algorithms.

table = []
# different algos benchmark
#################################
for algo in algos:
    out = cross_validate(algo, data, ["rmse", "mae"], kf)
    link = LINK[algo.__class__.__name__]
    mean_rmse = "{:.3f}".format(np.mean(out["test_rmse"]))
    mean_mae = "{:.3f}".format(np.mean(out["test_mae"]))
    mean_fit_time = str(datetime.timedelta(seconds=np.mean(out["fit_time"])))
    mean_test_time = str(datetime.timedelta(seconds=np.mean(out["test_time"])))

    # print current algo perf
    new_line = [link, mean_rmse, mean_mae, mean_fit_time, mean_test_time]
    print(tabulate([new_line], tablefmt="pipe"))
    table.append(new_line)
#################################


# just slope one
#################################
# algo = SlopeOne()
# out = cross_validate(algo, data, ["rmse", "mae"], kf)
# link = LINK[algo.__class__.__name__]
# mean_rmse = "{:.3f}".format(np.mean(out["test_rmse"]))
# mean_mae = "{:.3f}".format(np.mean(out["test_mae"]))
# mean_fit_time = str(datetime.timedelta(seconds=np.mean(out["fit_time"])))
# mean_test_time = str(datetime.timedelta(seconds=np.mean(out["test_time"])))
#
# # print current algo perf
# new_line = [LINK["SlopeOne"], mean_rmse, mean_mae, mean_fit_time, mean_test_time]
# print(tabulate([new_line], tablefmt="pipe"))
# table.append(new_line)
#################################

# weighted slope one Benchmark
#################################
out = cross_validate(user_specific, data, ["rmse", "mae"], kf)
mean_rmse = "{:.3f}".format(np.mean(out["test_rmse"]))
mean_mae = "{:.3f}".format(np.mean(out["test_mae"]))
mean_fit_time = str(datetime.timedelta(seconds=np.mean(out["fit_time"])))
mean_test_time = str(datetime.timedelta(seconds=np.mean(out["test_time"])))

# print current algo perf
new_line = [LINK["WeightedSlopeOne"], mean_rmse, mean_mae, mean_fit_time, mean_test_time]
print(tabulate([new_line], tablefmt="pipe"))
table.append(new_line)
#################################

# print all table
header = [LINK["ml-100k"], "RMSE", "MAE", "Fit Time", "Total Testing Time"]
print(tabulate(table, header, tablefmt="pipe"))
