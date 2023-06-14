import abc


class BaseStrategy(abc.ABC):
    def __init__(self, args):
        self.args = args
        self.budget = args.budget
        self.name = "BaseStrategy"

    def propose_acquisition(self, data_module, classifier_module, **kwargs):
        # Returns tuple: (pool_inds_to_annotate, extra_outputs)
        #  pool_inds_to_annotate: numpy array of size self.budget, containing indices of pool points to annotate.
        #  extra_outputs: dictionary containing any other outputs.
        pass

    def __str__(self):
        return self.name
