import pandas as pd
from surprise import SlopeOne
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from user_recom import UserRecommendation

if __name__ == "__main__":
    base_dataset_dir = "../dataset/ml-100k/"
    recom_module = UserRecommendation(user_info_path=base_dataset_dir + "u.user",
                                      user_ratings_path=base_dataset_dir + "u.data")
    recom_module.read_csv_data(["user_id", "age", "gender", "occupation", "zip_code"],
                               ["user_id", "item_id", "rating", "timestamp"], info_sep="|", ratings_sep="\t")

    reader = Reader(rating_scale=(1, 5))

    ratings = recom_module.user_ratings[(recom_module.user_ratings["user_id"] == 1) &
                                        (recom_module.user_ratings["item_id"] == 1)]
    print(ratings)

    data = Dataset.load_from_df(recom_module.user_ratings[['user_id', 'item_id', 'rating']], reader)
    train_set = data.build_full_trainset()

    algo = SlopeOne()
    algo.fit(train_set)

    uid = 1
    iid = 1
    pred = algo.predict(uid, iid, verbose=True)
