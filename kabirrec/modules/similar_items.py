from .module_base import ModuleBase
from ..surprise import Dataset
from ..surprise import KNNBasic
from ..surprise import Reader


class SimilarItems(ModuleBase):
    def __init__(self, user_rating, item_info, options=None):
        if options is None:
            options = {}
        ModuleBase.__init__(self, user_rating, options)
        self.item_info = item_info
        self.algo = None

    def name_to_id(self, name):
        # csv reads ids as integer but we need string in inner_id
        movie = self.item_info[self.item_info["movie_title"] == name]
        return movie["movie_id"].item()

    def id_to_name(self, iid):
        # after converting to string, here we convert back
        movie = self.item_info[self.item_info["movie_id"] == int(iid)]
        return movie["movie_title"].item()

    def fit(self):
        if self.verbose:
            print("Fitting the algorithm...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.user_rating[["user_id", "item_id", "rating"]], reader)
        train_set = data.build_full_trainset()

        sim_options = {"name": "pearson_baseline", "user_based": False}
        self.algo = KNNBasic(sim_options=sim_options)
        self.algo.verbose = self.verbose
        self.algo.fit(train_set)
        self.is_fit = True
        if self.verbose:
            print("Fitting is done.")

    def recommend(self, item, k=10):
        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        if self.verbose:
            print("Finding {} nearest items...".format(k))
        toy_story_raw_id = self.name_to_id(item)

        # Convert into inner id of the train set
        toy_story_inner_id = self.algo.trainset.to_inner_iid(toy_story_raw_id)

        # Get the inner ids of the closest 10 movies
        toy_story_neighbors_inner_ids = self.algo.get_neighbors(toy_story_inner_id, k=k)

        # Convert inner ids to real ids
        toy_story_neighbors_rids = (
            self.algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors_inner_ids
        )

        toy_story_neighbors = (self.id_to_name(rid) for rid in toy_story_neighbors_rids)

        if self.verbose:
            print("{} nearest items found.".format(k))

        return toy_story_neighbors
