class BaseEngine:
    def __init__(self, verbose=True):
        """
        Base class for engine

        :param verbose: whether or not to print running messages
        :type verbose: bool
        """
        super().__init__()
        self.verbose = verbose

    def _print(self, *args, show_name=True):
        if self.verbose:
            if show_name:
                print('[{}]'.format(self.__class__.__name__), *args)
            else:
                print(*args)
