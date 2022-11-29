from .module_base import ModuleBase


class ColdStart(ModuleBase):
    def __init__(self, dataset, options={}):
        ModuleBase.__init__(self, dataset, options)
        self.popular_table = None

    def fit(self):
        self.popular_table = self.dataset.groupby(["item_id"])["rating"].sum().reset_index(). \
            sort_values(by=["rating", "item_id"], ascending=[False, False])

    def recommend(self, n=10):
        return self.popular_table.head(n)
