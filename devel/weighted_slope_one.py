from surprise import WeightedSlopeOne
from surprise import Dataset
from surprise import Reader
from user_recom import UserRecommendation
from kmodes.kmodes import KModes

if __name__ == "__main__":
    base_dataset_dir = "../dataset/ml-100k/"
    recom_module = UserRecommendation(user_info_path=base_dataset_dir + "u.user",
                                      user_ratings_path=base_dataset_dir + "u.data")
    recom_module.read_csv_data(["user_id", "age", "gender", "occupation", "zip_code"],
                               ["user_id", "item_id", "rating", "timestamp"], info_sep="|", ratings_sep="\t")

    # cluster with optimal k
    kmode = KModes(n_clusters=25, init="random", n_init=5, n_jobs=-1, verbose=0)
    cluster_labels = kmode.fit_predict(recom_module.user_info)
    recom_module.user_info['cluster'] = cluster_labels.tolist()
    virtual_rating, virtual_count = recom_module.generate_virtual_rating_count(25)

    # virtual ratings ready
    mean_reader = Reader(rating_scale=(1, 5))
    count_reader = Reader(rating_scale=(virtual_count["rating_count"].min(), virtual_count["rating_count"].max()))
    mean_data = Dataset.load_from_df(virtual_rating, mean_reader)
    count_data = Dataset.load_from_df(virtual_count, count_reader)

    mean_train_set = mean_data.build_full_trainset()
    count_train_set = count_data.build_full_trainset()

    algo = WeightedSlopeOne(count_train_set)
    algo.fit(mean_train_set)
    uid = 1
    iid = 1
    pred = algo.predict(uid, iid, verbose=True)