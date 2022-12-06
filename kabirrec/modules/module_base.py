class ModuleBase:
    def __init__(self, user_rating, options):
        self.user_rating = user_rating
        self.options = options
        self.verbose = self.options.get("verbose", False)
        self.is_fit = False
