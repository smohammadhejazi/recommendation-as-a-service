from user_recom import UserRecommendation

if __name__ == "__main__":
    base_dataset_dir = "../dataset/ml-100k/"
    recom_module = UserRecommendation(user_info_path=base_dataset_dir + "u.user",
                                      user_ratings_path=base_dataset_dir + "u.data",
                                      item_info_path=base_dataset_dir + "u.item")
    recom_module.read_csv_data(["user_id", "age", "gender", "occupation", "zip_code"],
                               ["user_id", "item_id", "rating", "timestamp"],
                               ["movie_id", "movie_title", "release_date", "video_release_date", "imdb_url", "unknown",
                                "action", "adventure", "animation", "children's", "comedy", "crime", "documentary",
                                "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi",
                                "thriller", "war", "western"],
                               info_sep="|", ratings_sep="\t", item_sep="|")

    
