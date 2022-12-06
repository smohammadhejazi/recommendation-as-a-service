from kabirrec.recommendation_service import RecommendationService


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

    user_specific = recommendation_service.user_specific_module(options={"verbose": True,
                                                                         "k": None})
    user_specific.fit(20, 30)
    user_specific.draw_clusters_graph()
    prediction_rating = user_specific.recommend(1, 1)
    print(prediction_rating)
    prediction_rating = user_specific.recommend(1, 5)
    print(prediction_rating.est)
