from kabirrec import RecommendationService


if __name__ == "__main__":
    recommendation_service = RecommendationService()

    data_set = "ml-100k"
    recommendation_service.read_csv_data(
        user_info_path="../dataset/{}/u.user".format(data_set),
        user_ratings_path="../dataset/{}/u.data".format(data_set),
        item_info_path="../dataset/{}/u.item".format(data_set),
        user_info_columns=["user_id", "age", "gender", "occupation", "zip_code"],
        user_ratings_columns=["user_id", "item_id", "rating", "timestamp"],
        item_info_columns=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown",
                      "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
                      "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi",
                      "thriller", "war", "western"],
        user_info_sep="|", user_ratings_sep="\t", item_sep="|"
    )

    similar_items = recommendation_service.similar_items_module(options={"algo": "KNNBasic", "verbose": True})
    similar_items.fit()
    items = similar_items.recommend("Toy Story (1995)", n=5)
    for movie in items:
        print(movie)
