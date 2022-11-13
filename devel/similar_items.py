from user_recom import UserRecommendation
from surprise import Dataset, KNNBasic


def name_to_id(name, items):
    movie = items[items["movie_title"] == name]
    return movie["movie_id"].item()


def id_to_name(iid, items):
    movie = items[items["movie_id"] == iid]
    return movie["movie_name"].item()


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

    # First, train the algorithm to compute the similarities between items
    data = Dataset.load_builtin("ml-100k")
    trainset = data.build_full_trainset()
    sim_options = {"name": "pearson_baseline", "user_based": False}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)

    # Get id of a movie
    toy_story_raw_id = name_to_id("Toy Story (1995)", recom_module.item_info)

    # Convert into inner id of the train set
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

    # Get the inner ids of the closest 10 movies
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

    # Convert inner ids to real ids
    toy_story_neighbors = (
        algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors
    )
    # Convert real ids to names
    toy_story_neighbors = (id_to_name(rid, recom_module.item_info) for rid in toy_story_neighbors)

    # Print them
    print("The nearest 10 movies are:")
    for movie in toy_story_neighbors:
        print(movie)
