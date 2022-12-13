"""
This file contains ModuleBase class.
"""


class ModuleBase:
    """
    This class is the base of all of the other services.
    """
    def __init__(self, user_rating=None, user_info=None, item_info=None, options=None):
        """
        :param user_rating: Users' ratings csv
        :param options: Options dictionary
        """

        self.user_rating = user_rating
        self.user_info = user_info
        self.item_info = item_info
        self.options = options
        self.verbose = self.options.get("verbose", False)
        self.is_fit = False

    def name_to_id(self, name):
        """
        Get a name of item and returns its id from items' info csv
        :param name: Name of item
        :return: Id of item
        """
        movie = self.item_info[self.item_info["movie_title"] == name]
        return movie["movie_id"].item()

    def id_to_name(self, iid):
        """
        Get a id of item and returns its name from items' info csv
        :param iid: Id of item
        :return: Name of item
        """
        movie = self.item_info[self.item_info["movie_id"] == int(iid)]
        return movie["movie_title"].item()
