import pandas as pd
from src.modules.cold_start import ColdStart
from src.modules.similar_items import SimilarItems
from src.modules.user_specific import UserSpecific


class RecommendationService:
    def __init__(self):
        self.user_info = None
        self.user_ratings = None
        self.item_info = None

    def read_csv_data(self, user_info_path, user_ratings_path, item_info_path,
                      info_columns,  ratings_columns, item_columns,
                      info_sep, ratings_sep, item_sep):
        self.user_info = pd.read_csv(user_info_path, sep=info_sep, names=info_columns)
        self.user_ratings = pd.read_csv(user_ratings_path, sep=ratings_sep, names=ratings_columns)
        self.item_info = pd.read_csv(item_info_path, sep=item_sep, names=item_columns, encoding="ISO-8859-1")

    def cold_start_module(self, options=None):
        if options is None:
            options = {}
        return ColdStart(self.user_ratings, options=options)

    def similar_items_module(self, options=None):
        if options is None:
            options = {}
        return SimilarItems(self.user_ratings, self.item_info, options=options)

    def user_specific_module(self, options=None):
        if options is None:
            options = {}
        return UserSpecific(self.user_ratings, self.user_info, self.item_info, options=options)


if __name__ == "__main__":
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
    items = similar_items.recommend("Toy Story (1995)", k=5)
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

