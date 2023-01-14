import torch
from reward.base import BaseReward, RewardConfig
from generator.data_interface.data import DataRepresentation
from evaluation.data_point import Point


class RedundancyReferences(BaseReward):
    def __init__(self, reward_config: RewardConfig):
        super(RedundancyReferences, self).__init__(reward_config)

    def add(
        self, point: Point, config, data_object: DataRepresentation, index
    ):
        # This funtion is for creating the values for plotting

        # agreagar index and kwargs
        mode = config.mode
        redundancy_all_values_of_element_batch = []

        if mode == "k_n":
            if config.variable_length:
                n = data_object.elements_length[index]
            else:
                n = point.elements_length

            limitKeys = ((n) * (n - 1)) / 2
            iterable = self.redundancy_values

        elif mode == "n":

            n = point.n_inferred
            iterable = [str(i) + str(n.item()) for i in range(2, n + 1)]

        elif mode == "k":
            if config.variable_length:
                n = data_object.elements_length[index]
            else:
                n = point.elements_length

            k = point.k_inferred
            iterable = [str(k.item()) + str(i) for i in range(k, n + 1)]

        # Just go thorugh all the dict and get the minimum and maximum
        for i, key in enumerate(iterable):
            val = self.redundancy_values[key]
            redundancy_all_values_of_element_batch.append(val)

            if i == 0:
                redundancy_minimum = val
                redundancy_maximum = val

            else:
                if redundancy_minimum > val:
                    redundancy_minimum = val
                if redundancy_maximum < val:
                    redundancy_maximum = val

            if mode == "k_n":
                if i == limitKeys - 1:
                    break

        redundancy_all_values_of_element_batch = torch.tensor(
            redundancy_all_values_of_element_batch, dtype=torch.float32
        )
        wildcard_var = redundancy_all_values_of_element_batch

        logredundancy = False
        if logredundancy:

            wildcard_var = torch.log(wildcard_var)
            redundancy_maximum = torch.log(
                torch.tensor(redundancy_maximum, dtype=torch.float32)
            )
            redundancy_minimum = torch.log(
                torch.tensor(redundancy_minimum, dtype=torch.float32)
            )

        epsilon = 1e-35
        redundancy_all_values_of_element_batch_normalized = (
            wildcard_var - redundancy_minimum
        ) / (redundancy_maximum - redundancy_minimum + epsilon)
        redundancy_degradation_wrt_minimum = point.redundancy / redundancy_minimum

        point.redundancy_all_values_of_element_batch_normalized = redundancy_all_values_of_element_batch_normalized
        point.redundancy_all_values_of_element_batch = redundancy_all_values_of_element_batch
        point.redundancy_degradation_wrt_minimum = redundancy_degradation_wrt_minimum
        point.redundancy_maximum = redundancy_minimum
        point.redundancy_minimum = redundancy_maximum
