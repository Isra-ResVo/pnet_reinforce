import torch

from reward.base import RewardConfig, BaseReward
from reward.min_max import MaxMinError, MaxMinRedundancy
from reward.reward_representation import DataSolution
from generator.data_interface.data import DataRepresentation


class Reward(BaseReward):
    # this can be changed by any of the formulas given in reward.error_func module
    def __init__(self, reward_config: RewardConfig):
        super(Reward, self).__init__(reward_config=reward_config)
        self.reward_config = reward_config

    def main(self, data_object:DataRepresentation) -> DataSolution:

        reward = {}
        grouped_rewards = DataSolution(reward_config=self.reward_config)
        epsilon = 1e-35

        # prError for model selections
        size_subsets = (
            self.n_inferred - self.k_inferred
        ) + 1  #  subset size to have to fail before +2
        probability_of_error = self.error_formula(
            batch=data_object.batch,
            subsets=size_subsets,
            only_clouds=self.selected_clouds,
        )
        # reward['prError'] = pr_error
        grouped_rewards.probability_of_error = probability_of_error

        # min and max for normalization process
        max_min_prbobality_of_error = MaxMinError(reward_config=self.reward_config)
        maximum, minimum = max_min_prbobality_of_error.min_max_error(data_object=data_object)

        # print("probabilidad de error", pr_error)
        # print("maximum", maximum)
        # print("minimum", minimum)

        # Log probabilities
        # if self.config.log:
        #     logError = torch.log(pr_error)
        #     maximum = torch.log(maximum)
        #     minimum = torch.log(minimum)
        #     reward['normError'] = (logError - minimum)/(maximum - minimum + epsilon)
        # else:
        #     normError = (pr_error - minimum) / (maximum - minimum +epsilon)
        #     reward['normError'] = normError

        # normalized
        normalized_pr_error = (probability_of_error - minimum) / (
            maximum - minimum + epsilon
        )
        grouped_rewards.normalized_pr_error = normalized_pr_error
        # print("reward[normError]", reward["normError"])
        # print('valores normalizados de error de fallo', reward['normError'])

        # Redundancy and normalization
        redundancy = self.__redundancy()
        grouped_rewards.redundancy = redundancy

        max_min_redundancy_of_service = MaxMinRedundancy(
            reward_config=self.reward_config
        )
        maximum, minimum = max_min_redundancy_of_service.max_min_redundancy(
            len_elements=data_object.elements_length
        )

        normalized_redundancy = (redundancy - minimum) / (maximum - minimum + epsilon)
        grouped_rewards.normalized_redundancy = normalized_redundancy

        # Ponderate
        importance_of_objective_1 = self.config.wo[0]
        importance_of_objective_2 = self.config.wo[1]

        ponderate_objetive = (
            importance_of_objective_1 * normalized_pr_error
            + importance_of_objective_2 * normalized_redundancy
        )
        grouped_rewards.ponderate_objetive = ponderate_objetive
        # This ponderation was deactivate for not show improvent in the learnign
        # reward['ponderate'] = 10/(1-torch.log(reward['ponderate'])) # realizar cambios d


        return grouped_rewards

    def __redundancy(self):
        """
        This values are experimental values from real comfigurations. 
        This values were retrivied from previous literature
        """

        rewardR = []
        for (k_i, n_i) in zip(self.k_inferred, self.n_inferred):

            key = str(k_i.cpu().numpy()) + str(n_i.cpu().numpy())
            rewardR.append(
                torch.tensor(
                    self.redundancy_values.get(key, None),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        rewardR = torch.stack(rewardR, dim=0)

        return rewardR

    def extraction_time(self):

        rwd_time = []

        for (k_i, n_i) in zip(self.k_inferred, self.n_inferred):

            k = str(k_i.cpu().numpy())
            n = str(n_i.cpu().numpy())
            rwd_time.append(
                torch.tensor(self.time.get(k + n, 12345))
            )  # se castigan valores  soluciones no validas

        return torch.stack(rwd_time, dim=0)
