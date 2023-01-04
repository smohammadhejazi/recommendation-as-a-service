"""
This file contains the RecommendationService class
"""


import pandas as pd
from .services.cold_start import ColdStart
from .services.similar_items import SimilarItems
from .services.user_specific import UserSpecific


class RecommendationService:
    """
    A class to control the usage of ColdStart, SimilarItems, UserSpecific modules.
    Read csv files of dataset and use each module generator to get access to services modules
    and use them to recommend items or predict ratings.
    """

    def __init__(self):
        self.data_loaded = False
        self.user_info = None
        self.user_ratings = None
        self.item_info = None
        self.cold_start = None
        self.similar_items = None
        self.user_specific = None

    def read_csv_data(self, user_info_path, user_ratings_path, item_info_path,
                      info_columns,  ratings_columns, item_columns,
                      info_sep, ratings_sep, item_sep):
        """
        Reads the csv files of dataset and saves it in the class

        :param user_info_path: csv file path of users' info
        :param user_ratings_path: csv file path of users' ratings to items
        :param item_info_path: csv file path of items' info
        :param info_columns: columns of users' info csv
        :param ratings_columns: columns of users' ratings csv
        :param item_columns: columns of items' info
        :param info_sep: csv separator of users' info csv
        :param ratings_sep: csv separator of users' ratings csv
        :param item_sep: csv separator of items' info
        """

        self.user_info = pd.read_csv(user_info_path, sep=info_sep, names=info_columns)
        self.user_ratings = pd.read_csv(user_ratings_path, sep=ratings_sep, names=ratings_columns)
        self.item_info = pd.read_csv(item_info_path, sep=item_sep, names=item_columns, encoding="ISO-8859-1")
        self.data_loaded = True

    def cold_start_module(self, options=None):
        """
        Generates the ColdStart module with the currently saved dataset

        :param options: options for the cold start module:
            verbose: True/False
        :return: returns the ColdStart object
        """

        if options is None:
            options = {}
        self.cold_start = ColdStart(self.user_ratings, self.item_info, options=options)
        return self.cold_start

    def similar_items_module(self, options=None):
        """
        Generates the SimilarItem module with the currently saved dataset

        :param options: options for the cold start module:
            verbose: True/False
        :return: returns the SimilarItem object
        """

        if options is None:
            options = {}
        self.similar_items = SimilarItems(self.user_ratings, self.item_info, options=options)
        return self.similar_items

    def user_specific_module(self, options=None):
        """
        Generates the UserSpecific module with the currently saved dataset

        :param options: options for the cold start module:
            verbose: True/False
            k: None/number of clusters if given then the process of finding optimal clusters won't happen.
            top_n: number of predicted items. A table of predicted ratings for n unrated items of users is created
                    for later querying.
        :return: returns the UserSpecific object
        """

        if options is None:
            options = {}
        self.user_specific = UserSpecific(self.user_ratings, self.user_info, self.item_info, options=options)
        return self.user_specific


if __name__ == "__main__":
    """
    Examples of how to use the library
    """

    recommendation_service = RecommendationService()

    recommendation_service.read_csv_data(
        user_info_path="../dataset/ml-100k/u.user",
        user_ratings_path="../dataset/ml-100k/u.data",
        item_info_path="../dataset/ml-100k/u.item",
        info_columns=["user_id", "age", "gender", "occupation", "zip_code"],
        ratings_columns=["user_id", "item_id", "rating", "timestamp"],
        item_columns=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown",
                      "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
                      "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi",
                      "thriller", "war", "western"],
        info_sep="|", ratings_sep="\t", item_sep="|"
    )

    cold_start = recommendation_service.cold_start_module(options={"verbose": True})
    cold_start.fit()
    items = cold_start.recommend(5)
    print(items.head(5))

    similar_items = recommendation_service.similar_items_module(options={"verbose": True})
    similar_items.fit()
    items = similar_items.recommend("Toy Story (1995)", n=5)
    for movie in items:
        print(movie)

    user_specific = recommendation_service.user_specific_module(options={"verbose": True,
                                                                         "k": None})
    user_specific.fit(26, 27)
    user_specific.draw_clusters_graph()
    prediction_rating = user_specific.recommend(1, 1)
    print(prediction_rating)
    prediction_rating = user_specific.recommend(1, 5)
    print(prediction_rating.est)

