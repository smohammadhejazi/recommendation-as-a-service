from kabirrec import RecommendationService


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
