from user_recom import UserRecommendation
from surprise import Dataset, KNNBasic


def name_to_id(name, items):
    movie = items[items["movie_title"] == name]
    return movie["movie_id"]


def id_to_name(iid, items):
    movie = items[items["movie_id"] == iid]
    return movie["movie_name"]


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

    print(name_to_id("Dead Man Walking (1995)", recom_module.item_info))
    exit(0)

    def read_item_names():

        file_name = base_dataset_dir + "u.item"
        rid_to_name = {}
        name_to_rid = {}
        with open(file_name, encoding="ISO-8859-1") as f:
            for line in f:
                line = line.split("|")
                rid_to_name[line[0]] = line[1]
                name_to_rid[line[1]] = line[0]

        return rid_to_name, name_to_rid


    # First, train the algorithm to compute the similarities between items
    data = Dataset.load_builtin("ml-100k")
    trainset = data.build_full_trainset()
    sim_options = {"name": "pearson_baseline", "user_based": False}
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)

    # Read the mappings raw id <-> movie name
    rid_to_name, name_to_rid = read_item_names()

    # Retrieve inner id of the movie Toy Story
    toy_story_raw_id = name_to_rid["Toy Story (1995)"]
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)

    # Retrieve inner ids of the nearest neighbors of Toy Story.
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)

    # Convert inner ids of the neighbors into names.
    toy_story_neighbors = (
        algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors
    )
    toy_story_neighbors = (rid_to_name[rid] for rid in toy_story_neighbors)

    print()
    print("The 10 nearest neighbors of Toy Story are:")
    for movie in toy_story_neighbors:
        print(movie)