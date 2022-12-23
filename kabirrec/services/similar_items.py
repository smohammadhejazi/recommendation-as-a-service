"""
This file contains ColdStart class/module
"""


from .module_base import ModuleBase
from ..surprise import Dataset
from ..surprise import KNNBasic
from ..surprise import KNNWithZScore
from ..surprise import KNNWithMeans
from ..surprise import KNNBaseline
from ..surprise import Reader


ALGOS = {
    "knnbasic": KNNBasic,
    "knnwithzscore": KNNWithZScore,
    "knnwithmeans": KNNWithMeans,
    "knnbaseline": KNNBaseline
}


class SimilarItems(ModuleBase):
    def __init__(self, user_rating, item_info, options=None):
        """
        :param user_rating: Users' ratings csv
        :param item_info: Items' info csv
        :param options: Options dictionary
        """

        if options is None:
            options = {}
        ModuleBase.__init__(self, user_rating=user_rating, item_info=item_info, options=options)

        self.algo_class = ALGOS[options.get("algo", "knnbasic").lower()]
        self.sim_options = options.get("sim_options", {"name": "pearson_baseline", "user_based": False})
        self.bsl_options = options.get("bsl_options", {})
        self.k = options.get("k", 40)
        self.min_k = options.get("min_k", 1)
        self.algo = self.algo_class(k=self.k, min_k=self.min_k,
                                    sim_options=self.sim_options, bsl_options=self.bsl_options,
                                    verbose=self.verbose)

    def fit(self):
        """
        Fits the class and prepares the required things for recommend function
        """

        if self.verbose:
            print("Fitting the algorithm...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.user_rating[["user_id", "item_id", "rating"]], reader)
        train_set = data.build_full_trainset()

        self.algo.fit(train_set)
        self.is_fit = True
        if self.verbose:
            print("Fitting is done.")

    def recommend(self, item, n=10):
        """
        Recommend k items similar to the item given
        :param item: item to find similar items to
        :param n: number of similar items
        :return: List of (id, names) of similar items
        """
        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        if self.verbose:
            print("Finding {} nearest items...".format(n))
        toy_story_raw_id = self.name_to_id(item)

        # Convert into inner id of the train set
        toy_story_inner_id = self.algo.trainset.to_inner_iid(toy_story_raw_id)

        # Get the inner ids of the closest 10 movies
        toy_story_neighbors_inner_ids = self.algo.get_neighbors(toy_story_inner_id, k=n)

        # Convert inner ids to real ids
        toy_story_neighbors_rids = (
            self.algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors_inner_ids
        )

        toy_story_neighbors = [(rid, self.id_to_name(rid)) for rid in toy_story_neighbors_rids]

        if self.verbose:
            print("{} nearest items found.".format(n))

        return toy_story_neighbors
