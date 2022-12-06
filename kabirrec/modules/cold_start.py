from .module_base import ModuleBase


class ColdStart(ModuleBase):
    def __init__(self, user_rating, options=None):
        if options is None:
            options = {}
        ModuleBase.__init__(self, user_rating, options)
        self.popular_table = None

    def fit(self):
        if self.verbose:
            print("Ranking items...")
        self.popular_table = self.user_rating.groupby(["item_id"])["rating"].sum().reset_index(). \
            sort_values(by=["rating", "item_id"], ascending=[False, False])
        self.is_fit = True
        if self.verbose:
            print("All items are ranked.")

    def recommend(self, n=10):
        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        return self.popular_table.head(n)
