"""
This file contains ModuleBase class.
"""


class ModuleBase:
    """
    This class is the base of all of the other services.
    """
    def __init__(self, user_rating, options):
        """
        :param user_rating: Users' ratings csv
        :param options: Options dictionary
        """

        self.user_rating = user_rating
        self.options = options
        self.verbose = self.options.get("verbose", False)
        self.is_fit = False
