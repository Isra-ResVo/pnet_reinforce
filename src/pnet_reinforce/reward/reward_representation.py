from reward.base import BaseReward
class DataSolution(BaseReward):
    r'''
    A container to transport the reward information.
    '''
    def __init__(self, reward_config) -> None:
        super().__init__(reward_config=reward_config)
        self.probability_of_error = None
        self.normalized_pr_error = None
        self.redundancy = None 
        self.normalized_redundancy = None 
        self.ponderate_objective = None 
