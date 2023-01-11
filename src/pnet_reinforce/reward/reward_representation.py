class DataSolution(object):
    r'''
    A container to transport the reward information.
    '''
    def __init__(self) -> None:
        self.probability_of_error = None
        self.normalized_pr_error = None
        self.redundancy = None 
        self.normalized_redundancy = None 
        self.pondered_objective = None 
