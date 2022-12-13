"""
This file contains ColdStart class/module
"""


from .module_base import ModuleBase


class ColdStart(ModuleBase):
    """
    This class represents ColdStart module which is used to build a table
    of all items sorted by their ratings from highest to lowest.
    """
    def __init__(self, user_rating, item_info, options=None):
        """
        :param user_rating: User ratings csv
        :param options: Options dictionary
        """

        if options is None:
            options = {}
        ModuleBase.__init__(self, user_rating=user_rating, item_info=item_info, options=options)
        self.popular_table = None

    def fit(self):
        """
        Fits the class and prepares the required things for recommend function
        """

        if self.verbose:
            print("Ranking items...")
        self.popular_table = self.user_rating.groupby(["item_id"])["rating"].sum().reset_index(). \
            sort_values(by=["rating", "item_id"], ascending=[False, False])
        self.is_fit = True
        if self.verbose:
            print("All items are ranked.")

    def recommend(self, n=10):
        """
        Recommends top n popular items
        :param n: number of popular items
        :return: Panda frame
        """

        if self.is_fit is False:
            raise ValueError("Algorithm is not fit.")
        result = []
        for index, row in self.popular_table.head(n).iterrows():
            result.append((row["item_id"], self.id_to_name(row["item_id"]), row["rating"]))
        return result
