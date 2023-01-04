from kabirrec import RecommendationService


if __name__ == "__main__":
    recommendation_service = RecommendationService()

    recommendation_service.read_csv_data(
        user_info_path="../dataset/ml-100k/u.user",
        user_ratings_path="../dataset/ml-100k/u.data",
        item_info_path="../dataset/ml-100k/u.item",
        user_info_columns=["user_id", "age", "gender", "occupation", "zip_code"],
        user_ratings_columns=["user_id", "item_id", "rating", "timestamp"],
        item_columns=["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown",
                      "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
                      "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi",
                      "thriller", "war", "western"],
        user_info_sep="|", user_ratings_sep="\t", item_sep="|"
    )

    user_specific = recommendation_service.user_specific_module(options={"verbose": True,
                                                                         "k": None,
                                                                         "top_n": 10})
    user_specific.fit(20, 30)
    user_specific.draw_clusters_graph("../examples_output/user_specific_plot.png")
    prediction_rating = user_specific.recommend(1, 4)
    print(prediction_rating)
    prediction_rating = user_specific.predict_rating(1, 1)
    print(prediction_rating.est)
