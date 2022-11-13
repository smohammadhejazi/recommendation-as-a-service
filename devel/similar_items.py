from user_recom import UserRecommendation
from surprise import Dataset, KNNBasic

if __name__ == "__main__":
    # base_dataset_dir = "../dataset/ml-100k/"
    # recom_module = UserRecommendation(user_info_path=base_dataset_dir + "u.user",
    #                                   user_ratings_path=base_dataset_dir + "u.data")
    # recom_module.read_csv_data(["user_id", "age", "gender", "occupation", "zip_code"],
    #                            ["user_id", "item_id", "rating", "timestamp"], info_sep="|", ratings_sep="\t")

    data = Dataset.load_builtin("ml-100k")

    sim_options = {"name": "pearson_baseline", "user_based": False, "shrinkage": 0}
    algo = KNNBasic(sim_options=sim_options)
    uid = 1
    iid = 1
    pred = algo.predict(uid, iid, verbose=True)