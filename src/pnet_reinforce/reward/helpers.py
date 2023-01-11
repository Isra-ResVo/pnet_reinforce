import torch
from reward.base import BaseReward, RewardConfig


class HelperPlottingPoints(BaseReward):
    def __init__(self, reward_config: RewardConfig):
        super(HelperPlottingPoints, self).__init__(reward_config)

    def redundancyValsPlot(self, point, config, kwargs, index):
        # This funtion is for creating the values for plotting

        # agreagar index and kwargs
        mode = config.mode
        redundancy = []

        if mode == "k_n":
            if config.variable_length:
                n = kwargs["len_elements"][index]
            else:
                n = point["batchQntClouds"]

            limitKeys = ((n) * (n - 1)) / 2
            iterable = self.redundancy_values

        elif mode == "n":

            n = point["n_position"]
            iterable = [str(i) + str(n.item()) for i in range(2, n + 1)]

        elif mode == "k":
            if config.variable_length:
                n = kwargs["len_elements"][index]
            else:
                n = point["batchQntClouds"]

            k = point["k_position"]
            iterable = [str(k.item()) + str(i) for i in range(k, n + 1)]

        for i, key in enumerate(iterable):
            val = self.redundancy_values[key]
            redundancy.append(val)

            if i == 0:
                minimum = val
                maximum = val

            else:
                if minimum > val:
                    minimum = val
                if maximum < val:
                    maximum = val

            if mode == "k_n":
                if i == limitKeys - 1:
                    break

        redundancy = torch.tensor(redundancy, dtype=torch.float32)
        redundancy_original = redundancy

        logredundancy = False
        if logredundancy:

            redundancy = torch.log(redundancy)
            maximum = torch.log(torch.tensor(maximum, dtype=torch.float32))
            minimum = torch.log(torch.tensor(minimum, dtype=torch.float32))

        epsilon = 1e-35
        redundancy = (redundancy - minimum) / (maximum - minimum + epsilon)
        ratio = point["redundancy"] / minimum

        return redundancy, redundancy_original, ratio, minimum, maximum
